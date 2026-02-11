# main.pyimport logging
import logging
import math
import os
import argparse
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn import MSELoss, SmoothL1Loss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from typing import List

from src.model.utils import freeze_backbone
from src.data.dataset import PetBurstDataset
from src.data.utils import get_batch_mask
from src.data.utils import DistributedWeightedRandomSampler, expand_train_input, get_label_from_sample
from src.model.factory import create_loss, create_model
from src.model.utils import convert_targets_to_polar
from src.training.losses import VirtualCylinderAwareLoss
from src.utils.config import TrainCfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def worker_init_fn(_):
    import torch
    torch.set_num_threads(1)


def parse_args():
    # Pre-parse only the .env path so we can load it before reading os.environ
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('-e', '--env', dest='env_path', type=str, default=None,
                     help='Path to a .env file to load before reading environment variables')
    pre_args, _ = pre.parse_known_args()

    if pre_args.env_path:
        load_dotenv(dotenv_path=pre_args.env_path, override=True)
    else:
        load_dotenv()

    # break long env fetches for lint
    args = Namespace(
        train=os.getenv("TRAIN", None),
        val=os.getenv("VAL", None),
        logdir=os.getenv(
            "LOGDIR", "runs/pet_pointloc"
        ),
        bs=int(os.getenv("BS", TrainCfg.batch_size)),
        epochs=int(os.getenv("EPOCHS", TrainCfg.max_epochs)),
        lr=float(os.getenv("LR", TrainCfg.lr)),
        max_lr=float(os.getenv("MAX_LR", TrainCfg.onecycle_max_lr)),
        wd=float(os.getenv("WD", TrainCfg.weight_decay)),
        workers=int(os.getenv("WORKERS", TrainCfg.num_workers)),
        clip=float(os.getenv("CLIP", TrainCfg.clip_grad_norm)),
        no_amp=bool(int(os.getenv("NO_AMP", "1"))),
        freeze_backbone_params=int(
            os.getenv("FREEZE_BACKBONE_PARAMS", "0")
        ),
        project_to_shell=bool(int(os.getenv("PROJECT_TO_SHELL", "0"))),
        lambda_value=float(
            os.getenv("LAMBDA_VALUE", TrainCfg.lambda_value)
        ),
        normalisation_strategy=os.getenv(
            'NORMALISATION_STRATEGY', TrainCfg.normalisation_strategy
        ).lower(),
        model=os.getenv("MODEL", TrainCfg.model),
        seed=int(os.getenv("SEED", TrainCfg.seed)),
        loss=os.getenv("LOSS", "smooth").lower(),
        model_path=os.getenv("MODEL_PATH"),
        data_type=os.getenv("DATA_TYPE", "train").lower(),
        coordinate_system=os.getenv("COORDINATE_SYSTEM", "cartesian").lower(),
        transfer_learning=os.getenv("TRANSFER_LEARNING", "False") == "True",
        log_normalise=os.getenv("LOG_NORMALISE", "False") == "True",
        linear_scaling=os.getenv("LINEAR_SCALING", "False") == "True",
        skip_phase_1=os.getenv("SKIP_PHASE_1", "False") == "True",
        log_scale_factor=float(os.getenv("LOG_SCALE_FACTOR", "1.0")),
        phase_1_ratio=float(os.getenv("PHASE_1_RATIO", "0.2")),
        phase_2_ratio=float(os.getenv("PHASE_2_RATIO", "0.2"))
    )
    if args.train is None:
        raise ValueError("Environment variable TRAIN must be set")
    if args.val is None:
        raise ValueError("Environment variable VAL must be set")
    if args.data_type not in ["train", "trues"]:
        raise ValueError(f"DATA_TYPE must be 'train', 'trues', got: {args.data_type}")
    return args


def compute_class_balanced_weights(dataset) -> torch.Tensor:
    """
    Compute 1 / class_count weights based on event_type for each sample.
    Assumes binary labels {0,1} but will work for >2 as well.
    """
    labels: List[int] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.append(get_label_from_sample(sample))

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor)
    # avoid div by zero
    class_counts = torch.clamp(class_counts, min=1)

    # per-sample weight = 1 / count_of_its_class
    weights = 1.0 / class_counts[labels_tensor]
    return weights.to(torch.float32)


def make_loaders(
        train_dir: str,
        val_dir: str,
        bs: int,
        workers: int,
        balance_trues: bool = False,  # turn on class balancing for training
        data_type: str = "train",  # which directories to search: train or trues
        pattern: str = "*.pt"
):
    # Expand training .pt files (or whatever your helper does)
    train_list = expand_train_input(
        train_dir,
        data_type=data_type
    )

    train_ds = PetBurstDataset(
        train_list,
        regex_filter=pattern
    )
    val_ds = PetBurstDataset(
        val_dir,
        regex_filter=pattern,
        transform=None
    )

    using_ddp = dist.is_available() and dist.is_initialized()
    pin = torch.cuda.is_available()

    # ---------- TRAIN SAMPLER ----------
    if balance_trues:
        # Compute per-sample weights from event_type
        weights = compute_class_balanced_weights(train_ds)

        if using_ddp:
            # DDP-aware weighted sampler
            train_sampler = DistributedWeightedRandomSampler(
                weights=weights,
                num_samples=None,  # will default to ~len(dataset)/world_size per rank
                replacement=True,
            )
        else:
            # Single-process weighted sampler
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),  # one "epoch"
                replacement=True,
            )

        train_shuffle = False  # when using sampler, shuffle must be False
        train_drop_last = True
    else:
        # Original DDP sampler / plain shuffle
        train_sampler = (
            DistributedSampler(train_ds, shuffle=True, drop_last=True)
            if using_ddp
            else None
        )
        train_shuffle = (train_sampler is None)
        train_drop_last = True

    # ---------- VAL SAMPLER ----------
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, drop_last=False)
        if using_ddp
        else None
    )

    # ---------- LOADERS ----------
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin,
        drop_last=train_drop_last,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, workers // 2),
        pin_memory=pin,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    return train_loader, val_loader, train_ds


def train_one_epoch(
        model: nn.Module,
        criterion: SmoothL1Loss |
                   MSELoss |
                   VirtualCylinderAwareLoss,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler,
        scaler: GradScaler | None,
        device: torch.device,
        writer: SummaryWriter | None,
        epoch: int,
        args: Namespace,
        log_every: int = 50,
        use_amp: bool = True,
        clip: float = 1.0,
        filter_mode: str = "all"
):
    model.train()
    # Ensure different shuffles across epochs for DistributedSampler
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    # Running accumulators for logging every `log_every`
    running = {metric: 0.0 for metric in TrainCfg.training_metrics}
    steps_accumulated = 0
    t0 = time()

    for step, batch in enumerate(loader, start=1):
        burst = batch["burst"].to(device)
        target_coord = batch["target"].to(device)
        event_type = batch["event_type"].to(device)
        is_pe = batch["is_pe"].to(device)

        # Flag to track if this is a dummy step
        step_is_dummy = False

        mask = get_batch_mask(event_type, is_pe, filter_mode)

        if mask.sum() == 0:
            # CRITICAL FIX: Do NOT continue.
            # We must run a forward/backward pass to keep DDP in sync.
            step_is_dummy = True
        else:
            burst = burst[mask]
            target_coord = target_coord[mask]
            event_type = event_type[mask]

            # Geometry sanity check for trues data
            if filter_mode in ["trues", "trues_pe"]:
                r1_target = torch.sqrt(target_coord[:, 0]**2 + target_coord[:, 1]**2)
                r2_target = torch.sqrt(target_coord[:, 3]**2 + target_coord[:, 4]**2)

                # Detector is at ~235mm, filter out bad labels (0,0,0) closer than 200mm
                valid_mask = (r1_target > 200.0) & (r2_target > 200.0)

                if valid_mask.sum() == 0:
                    step_is_dummy = True
                else:
                    burst = burst[valid_mask]
                    target_coord = target_coord[valid_mask]
                    event_type = event_type[valid_mask]

        if args.log_normalise:
            # 1. Split channels
            counts = burst[:, :, 0:2, :, :]  # Inner/Outer (Counts)
            doi_map = burst[:, :, 2:3, :, :]  # DOI Map (Ratio)

            # 2. Log-Norm ONLY the counts to reveal the tails
            counts = torch.log1p(counts)

            # 3. Recombine
            burst = torch.cat([counts, doi_map], dim=2)

        if args.linear_scaling:
            # 1. Split channels
            counts = burst[:, :, 0:2, :, :]  # Inner/Outer (Counts)
            doi_map = burst[:, :, 2:3, :, :]  # DOI Map (Ratio)

            # 2. Log-Norm ONLY the counts to reveal the tails
            counts = counts / 100.0

            # 3. Recombine
            burst = torch.cat([counts, doi_map], dim=2)

        # Prepare Targets (Only if valid, otherwise placeholders)
        if not step_is_dummy:
            if args.coordinate_system == "polar":
                target_coord = convert_targets_to_polar(target_coord)
            target = torch.cat([target_coord, event_type.unsqueeze(-1)], dim=-1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            if step_is_dummy:
                # DUMMY PASS: Run model on full burst to generate graph
                # Multiply output by 0 to ensure 0 gradients
                # This satisfies DDP's need for synchronisation
                dummy_out = model(burst)
                total = dummy_out.sum() * 0.0
            else:
                # REAL PASS
                out = criterion(model(burst), target)
                total = out["total_loss"] if isinstance(out, dict) else out

        # Backward Pass (Happens for both Real and Dummy)
        if scaler is not None and use_amp:
            scaler.scale(total).backward()
            if clip is not None and clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if clip is not None and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Skip logging for dummy steps to avoid polluting metrics with zeros
        if step_is_dummy:
            continue

        # Update running metrics (detach to CPU scalars)
        running["total_loss"] += float(total.detach().item()) * math.floor(args.log_scale_factor) if args.log_normalise else float(total.detach().item())
        if isinstance(out, dict):
            for key in TrainCfg.training_metrics:
                if key in out:
                    running[key] += float(out[key].item())
        steps_accumulated += 1

        # Periodic logging
        if step % log_every == 0 and writer is not None:
            lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            global_step = epoch * len(loader) + step

            # Average over the last `log_every` steps
            avg = {
                k: v / max(steps_accumulated, 1)
                for k, v in running.items()
            }
            if isinstance(criterion, torch.nn.MSELoss):
                total_loss = math.sqrt(avg["total_loss"])
            else:
                total_loss = avg["total_loss"]

            writer.add_scalar(
                "train/loss", total_loss, global_step
            )
            writer.add_scalar(
                "train/lr", lr, global_step
            )

            # Log additional metrics if available
            for key in TrainCfg.training_metrics:
                if key in avg and avg[key] > 0:
                    writer.add_scalar(f"train/{key}", avg[key], global_step)

            logging.info(
                f"Epoch {epoch} Step {step}: "
                f"loss={total_loss:.3f} | "
                f"lr = {lr: .6f}"
            )

            # Reset window
            for k in running.keys():
                running[k] = 0.0
            steps_accumulated = 0

    dt = time() - t0
    print(f"Epoch {epoch} train done in {dt:.1f}s")


def validate(
        model: nn.Module,
        criterion: SmoothL1Loss |
                   MSELoss |
                   VirtualCylinderAwareLoss,
        loader: DataLoader,
        device: torch.device,
        writer: SummaryWriter | None,
        epoch: int,
        args: Namespace,
        filter_mode: str = "all"
):
    model.eval()

    # Accumulators (as tensors on a device for easy all_reduce)
    sums = {metric: torch.tensor(0.0, device=device) for metric in TrainCfg.training_metrics}
    n_samples = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in loader:
            burst = batch["burst"].to(device)
            target_coord = batch["target"].to(device)
            event_type = batch["event_type"].to(device)
            is_pe = batch["is_pe"].to(device)

            mask = get_batch_mask(event_type, is_pe, filter_mode)

            if mask.sum() == 0:
                continue
            else:
                burst = burst[mask]
                target_coord = target_coord[mask]
                event_type = event_type[mask]

                # Geometry sanity check for trues data
                if filter_mode in ["trues", "trues_pe"]:
                    r1_target = torch.sqrt(target_coord[:, 0]**2 + target_coord[:, 1]**2)
                    r2_target = torch.sqrt(target_coord[:, 3]**2 + target_coord[:, 4]**2)

                    # Detector is at ~235mm, filter out bad labels (0,0,0) closer than 200mm
                    valid_mask = (r1_target > 200.0) & (r2_target > 200.0)

                    if valid_mask.sum() == 0:
                        continue

                    burst = burst[valid_mask]
                    target_coord = target_coord[valid_mask]
                    event_type = event_type[valid_mask]

            if args.log_normalise:
                counts = burst[:, :, 0:2, :, :]  # Inner/Outer (Counts)
                doi_map = burst[:, :, 2:3, :, :]  # DOI Map (Ratio)
                counts = torch.log1p(counts)
                burst = torch.cat([counts, doi_map], dim=2)

            if args.linear_scaling:
                counts = burst[:, :, 0:2, :, :]  # Inner/Outer (Counts)
                doi_map = burst[:, :, 2:3, :, :]  # DOI Map (Ratio)

                # 2. Log-Norm ONLY the counts to reveal the tails
                counts = counts / 100.0

                # 3. Recombine
                burst = torch.cat([counts, doi_map], dim=2)

            if args.coordinate_system == "polar":
                target_coord = convert_targets_to_polar(target_coord)  # (B,4)
            target = torch.cat([target_coord, event_type.unsqueeze(-1)], dim=-1)  # (B,5)
            out = criterion(model(burst), target)
            loss_scalar = out["total_loss"] if isinstance(out, dict) else out
            batch_size = torch.tensor(float(burst.size(0)), device=device)
            sums["total_loss"] += loss_scalar.detach() * batch_size * math.floor(args.log_scale_factor) if args.log_normalise else loss_scalar.detach() * batch_size
            if isinstance(out, dict):
                for key in TrainCfg.training_metrics:
                    if key in out:
                        sums[key] += out[key].detach() * batch_size
            n_samples += batch_size

    # DDP: sum across ranks
    if dist.is_available() and dist.is_initialized():
        for k in sums.keys():
            dist.all_reduce(sums[k], op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)

    denom = torch.clamp(n_samples, min=1.0)
    avgs = {k: (v / denom).item() for k, v in sums.items()}

    if isinstance(criterion, torch.nn.MSELoss):
        total_loss = math.sqrt(avgs["total_loss"])
    else:
        total_loss = avgs["total_loss"]

    if writer is not None:
        writer.add_scalar("val/L1Loss", total_loss, epoch)
        # Log additional metrics if available
        for key in TrainCfg.training_metrics:
            if key in avgs and avgs[key] > 0:
                writer.add_scalar(f"val/{key}", avgs[key], epoch)

    # Rank-0 console summary
    if (
            (not dist.is_available())
            or (not dist.is_initialized())
            or dist.get_rank() == 0
    ):
        print(
            f"[EXPERIMENT] exp={args.logdir}"
            f"[VAL] epoch={epoch} "
            f'loss={total_loss:.3f} | '
        )
    return avgs["total_loss"]


def main():
    args = parse_args()
    print(f"Using seed: {args.seed}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = torch.cuda.is_available() and world_size > 1

    if use_ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=10)
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        rank = 0

    model = create_model(args.model, args).to(device)
    if rank == 0:
        print(f"Using '{args.model.lower()}' model")

    if use_ddp:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True
        )
    # Select loss
    criterion = create_loss(args)
    scaler = torch.amp.GradScaler(
        enabled=(not args.no_amp) and (device.type == "cuda")
    )

    # Logging (rank 0 only)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir) if rank == 0 else None
    if writer is not None:
        writer.add_text("hparams", str(vars(args)))

    best_val = float("inf")
    ckpt_dir = Path(args.logdir) / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Determine Phase Structure
    if args.transfer_learning:

        if args.skip_phase_1:
            print("Skipping Phase 1 as requested...")
            p1_epochs = 0
            p2_epochs = int(args.epochs * args.phase_2_ratio)
        else:
            # Split total epochs: Half for Phase 1 (Singles), Half for Phase 2 (All)
            p1_epochs = int(args.epochs * args.phase_1_ratio)
            p2_epochs = int(args.epochs * args.phase_2_ratio)

        p3_epochs = args.epochs - (p1_epochs + p2_epochs)

        phases = [
            {
                "name": "Phase 1 (Singles PE - Feature Learning)",
                "epochs": p1_epochs,
                "freeze_backbone": False,
                "filter_mode": "singles",
                "balance_trues": False,
                "lr_mult": 1.0
            },
            {
                "name": "Phase 2 (Trues PE + Replay)",
                "epochs": p2_epochs,
                "balance_trues": True,  # Sampler mixes 50/50
                "filter_mode": "trues",  # <--- FIXED: Allow Singles to pass through
                "freeze_backbone": True,  # <--- FIXED: Lock the 'Blob Detector' features
                "lr_mult": 0.1
            },
            {
                "name": "Phase 3 (All Data - Robustness)",
                "epochs": p3_epochs,
                "freeze_backbone": False,
                "balance_trues": True,  # Keep balanced to prevent 90% singles domination
                "filter_mode": "all",
                "lr_mult": 0.25
            }
        ]
    else:
        phases = [{
            "name": "Standard Training",
            "epochs": args.epochs,
            "freeze_backbone": False,
            "balance_trues": False,
            "filter_mode": "trues",
            "lr_mult": 1.0
        }]

    pattern = {
        "all": r".*\.pt",
        "trues": r"\d+_1_[01]\.pt",
        "singles": r"\d+_0_[01](?:_aug)?\.pt",
        "trues_pe": r"\d+_1_1\.pt",
        "singles_pe": r"\d+_0_1(?:_aug)?\.pt",
    }

    # Global epoch counter for logging continuity
    current_epoch = 1
    patience = int(os.getenv("PATIENCE", "0"))
    epochs_since_improve = 0

    for phase in phases:
        pattern = pattern.get(phase.get("filter_mode"))
        train_loader, val_loader, train_ds = make_loaders(
            args.train, args.val, args.bs, args.workers,
            balance_trues=phase["balance_trues"],
            data_type=args.data_type,
            pattern=pattern
        )

        if rank == 0:
            print(
                f"Device: {device} | World size: {world_size} | "
                f"Train steps/epoch(per-rank): {len(train_loader)} | "
                f"Val steps(per-rank): {len(val_loader)}"
            )

        if phase["epochs"] <= 0:
            continue

        if rank == 0:
            print(f"\n=== Starting {phase['name']} ===")
            print(f"Running for {phase['epochs']} epochs.")

        # 1. Apply Freezing Logic if needed (Phase 2)
        raw_model = model.module if use_ddp else model
        freeze_backbone(raw_model, freeze=phase["freeze_backbone"])

        # 2. Reset Optimiser/Scheduler for this phase-Only pass parameter that require gradients
        params_to_train = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(
            params_to_train, lr=args.lr, weight_decay=args.wd
        )

        total_steps = len(train_loader) * phase['epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr * phase['lr_mult'],
            epochs=phase['epochs'],
            steps_per_epoch=len(train_loader),  # explicit steps per epoch is safer
            anneal_strategy="cos",
            pct_start=0.3 if not phase["freeze_backbone"] else 0.1,  # Short warmup for fine-tune
            div_factor=10,
            final_div_factor=10,
            cycle_momentum=True
        )

        # 3. Epoch Loop for this Phase
        for _ in range(phase['epochs']):
            # Inform augmentation about the current epoch
            if hasattr(train_ds.transform, "set_epoch"):
                train_ds.transform.set_epoch(current_epoch)

            train_one_epoch(
                model=model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                writer=writer,
                epoch=current_epoch,  # Continuous epoch number
                args=args,
                log_every=50,
                use_amp=(not args.no_amp and device.type == "cuda"),
                clip=args.clip,
            )

            val_loss = validate(
                model=model,
                criterion=criterion,
                loader=val_loader,
                device=device,
                writer=writer,
                epoch=current_epoch,
                args=args,
            )

            # Save best
            if rank == 0:
                if current_epoch == args.epochs:
                    ckpt_path = (
                            ckpt_dir / f"last_epoch{current_epoch:03d}_loss{val_loss:.3f}.pt"
                    )
                    state_dict = (
                        model.module.state_dict() if use_ddp
                        else model.state_dict()
                    )
                    torch.save(
                        {
                            "epoch": current_epoch,
                            "model_state": state_dict,
                            "optimizer_state": optimizer.state_dict(),
                            "val_loss": best_val,
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    print(f"Saved last checkpoint: {ckpt_path}")

                if val_loss < best_val:
                    best_val = val_loss
                    epochs_since_improve = 0
                    ckpt_path = (
                            ckpt_dir / f"best_epoch{current_epoch:03d}_loss{best_val:.3f}.pt"
                    )
                    state_dict = (
                        model.module.state_dict() if use_ddp
                        else model.state_dict()
                    )
                    torch.save(
                        {
                            "epoch": current_epoch,
                            "model_state": state_dict,
                            "optimizer_state": optimizer.state_dict(),
                            "val_loss": best_val,
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    print(f"Saved checkpoint: {ckpt_path}")
                else:
                    epochs_since_improve += 1
                    if 0 < patience <= epochs_since_improve:
                        print(
                            "Early stopping at epoch "
                            f"{current_epoch} after {patience} epochs without improvement."
                        )
                        break

            if 0 < patience <= epochs_since_improve:
                break

            current_epoch += 1

        if 0 < patience <= epochs_since_improve:
            break

    if use_ddp:
        dist.barrier()  # Ensure all ranks finish loader creation

    if writer is not None:
        writer.close()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
