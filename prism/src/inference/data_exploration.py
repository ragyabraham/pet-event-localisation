import torch


def forensic_analysis(train_loader, test_loader):
    # Get a single batch from each
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    print("--- FORENSIC REPORT ---")

    # 1. Check Input Scale (Normalisation)
    print(f"Input Range [Train]: {x_train.min():.2f} to {x_train.max():.2f}")
    print(f"Input Range [Test]:  {x_test.min():.2f} to {x_test.max():.2f}")

    # 2. Check Target Scale (Geometry)
    print(f"Target Mean R [Train]: {torch.norm(y_train[:, :2], dim=1).mean():.2f} mm")
    print(f"Target Mean R [Test]:  {torch.norm(y_test[:, :2], dim=1).mean():.2f} mm")

    # 3. Check Channel Statistics (Did channel swap?)
    # Calculate mean intensity of Channel 0 vs Channel 1
    t_ch0 = x_train[:, 0].mean()
    t_ch1 = x_train[:, 1].mean()
    test_ch0 = x_test[:, 0].mean()
    test_ch1 = x_test[:, 1].mean()

    print(f"Ch0 Mean [Train vs Test]: {t_ch0:.2f} vs {test_ch0:.2f}")
    print(f"Ch1 Mean [Train vs Test]: {t_ch1:.2f} vs {test_ch1:.2f}")




def check_data_scale(dataset_name, dataloader):
    # Grab a single batch
    batch = next(iter(dataloader))
    burst = batch["burst"]

    print(f"\n--- DATA SCALE REPORT: {dataset_name} ---")
    print(f"Max Value (Brightest Pixel): {burst.max().item():.2f}")
    print(f"Mean Value (Active Pixels):  {burst[burst > 0].mean().item():.2f}")
    print(f"Min Value:                   {burst.min().item():.2f}")
    print(f"Data Type:                   {burst.dtype}")
    print("-" * 40)
