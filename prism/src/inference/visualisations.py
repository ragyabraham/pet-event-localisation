import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable # Crucial for alignment
import seaborn as sns

# Helper function to convert Cartesian to cylindrical coordinates for visualization
def cartesian_to_cylindrical(x, y, z):
    # Ensure inputs are tensors
    if not torch.is_tensor(x): x = torch.tensor(x)
    if not torch.is_tensor(y): y = torch.tensor(y)
    if not torch.is_tensor(z): z = torch.tensor(z)

    Z_HALF = 148.0  # mm
    W_THETA = 519   # pixels along theta
    H_Z = 104       # pixels along z

    """Convert Cartesian coordinates to cylindrical (theta, r, z)."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    two_pi = torch.as_tensor(2.0 * torch.pi, device=theta.device, dtype=theta.dtype)
    theta_wrapped = torch.where(theta < 0, theta + two_pi, theta)

    # Map theta (-pi, pi) -> (0, 519)
    theta_pix = (theta_wrapped / two_pi) * W_THETA

    # Map Z (-148, 148) -> (0, 104)
    z_half = torch.as_tensor(Z_HALF, device=z.device, dtype=z.dtype)
    z_pix = ((z + z_half) / (2.0 * z_half)) * H_Z

    return theta_pix, r, z_pix

def visualise_predictions(burst, pred_cart, target_cart, event_type, pred_class_prob, idx=0, output_directory='./'):
    """
    Publication-ready visualization with perfectly aligned colorbars and aspect ratios.
    """

    # --- PLOTTING STYLE ---
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '#D0D0D0',
        'grid.linestyle': ':',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'mathtext.fontset': 'cm' # Computer Modern for LaTeX math
    })

    # --- DATA PREP ---
    pred = pred_cart[idx].cpu().numpy() if torch.is_tensor(pred_cart) else pred_cart[idx]
    target = target_cart[idx].cpu().numpy() if torch.is_tensor(target_cart) else target_cart[idx]
    burst_img = burst[idx].cpu().numpy() if torch.is_tensor(burst) else burst[idx]

    evt_type_val = event_type[idx].item() if torch.is_tensor(event_type) else event_type[idx]
    class_prob_val = pred_class_prob[idx].item() if torch.is_tensor(pred_class_prob) else pred_class_prob[idx]

    # Permutation Invariance Logic
    p1 = pred[0:3]
    p2 = pred[3:6]
    t1 = target[0:3]
    t2 = target[3:6]

    dist_a = np.linalg.norm(p1 - t1) + np.linalg.norm(p2 - t2)
    dist_b = np.linalg.norm(p1 - t2) + np.linalg.norm(p2 - t1)

    if dist_b < dist_a:
        t1, t2 = t2.copy(), t1.copy()
        target_copy = target.copy()
        target[0:3] = target_copy[3:6]
        target[3:6] = target_copy[0:3]

    # Convert to Cylindrical for Scatter Plots
    pt1_th, pt1_r, pt1_z = cartesian_to_cylindrical(*p1)
    pt2_th, pt2_r, pt2_z = cartesian_to_cylindrical(*p2)
    tt1_th, tt1_r, tt1_z = cartesian_to_cylindrical(*t1)
    tt2_th, tt2_r, tt2_z = cartesian_to_cylindrical(*t2)

    # Calculate Errors
    err_p1 = np.linalg.norm(p1 - t1)
    err_p2 = np.linalg.norm(p2 - t2)
    type_str = "True Coincidence" if evt_type_val == 1 else "Single Event"
    pred_str = "True Coincidence" if class_prob_val > 0.5 else "Single Event"

    # --- FIGURE LAYOUT ---
    # Increased figure height slightly to accommodate aspect-ratio constraints
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.6])

    # Row 1: Heatmaps
    ax_hm1 = fig.add_subplot(gs[0, 0:2])
    ax_hm2 = fig.add_subplot(gs[0, 2:4])

    # Row 2: Geometric Views
    ax_cyl = fig.add_subplot(gs[1, 0:2]) # Theta-Z
    ax_top = fig.add_subplot(gs[1, 2])   # X-Y
    ax_side = fig.add_subplot(gs[1, 3])  # X-Z

    # Row 3: Statistics Table
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    # --- PLOT 1 & 2: HEATMAPS (With Aligned Colorbars) ---
    inner_sum = burst_img[0, :, :].sum(axis=0) if burst_img.ndim == 4 else burst_img[0]
    outer_sum = burst_img[1, :, :].sum(axis=0) if burst_img.ndim == 4 else burst_img[1]

    vmax = np.percentile(inner_sum[inner_sum > 0], 99) if inner_sum.max() > 0 else 1
    vmax = max(vmax, 5.0)

    for ax, img, title in zip([ax_hm1, ax_hm2], [inner_sum, outer_sum], ["Inner Layer Signal", "Outer Layer Signal"]):
        im = ax.imshow(img, cmap='inferno', aspect='equal', origin='lower', vmax=vmax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r"Transaxial Bin ($\theta$)")
        ax.set_ylabel(r"Axial Bin ($Z$)")

        # KEY CHANGE: make_axes_locatable aligns colorbar to the plot exactly
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Hits", rotation=270, labelpad=12)

    # --- PLOTTING HELPERS ---
    def plot_pair(ax, t_x, t_y, p_x, p_y):
        ax.scatter(t_x, t_y, c='none', edgecolors='#006400', s=120, linewidth=2, label='Ground Truth', zorder=3)
        ax.scatter(p_x, p_y, c='#8B0000', marker='x', s=100, linewidth=2, label='Prediction', zorder=4)
        ax.plot([t_x[0], p_x[0]], [t_y[0], p_y[0]], '--', c='gray', alpha=0.6, linewidth=1)
        ax.plot([t_x[1], p_x[1]], [t_y[1], p_y[1]], '--', c='gray', alpha=0.6, linewidth=1)

    # --- PLOT 3: Cylindrical View (Theta-Z) ---
    plot_pair(ax_cyl, [tt1_th, tt2_th], [tt1_z, tt2_z], [pt1_th, pt2_th], [pt1_z, pt2_z])
    ax_cyl.set_title("Unrolled View (Theta-Z)", fontweight='bold')
    ax_cyl.set_xlabel(r"$\theta$ (Bins)")
    ax_cyl.set_ylabel(r"$Z$ (Bins)")
    ax_cyl.set_xlim(0, 520)
    ax_cyl.set_ylim(0, 104)

    # KEY CHANGE: aspect='equal' prevents distortion here too
    ax_cyl.set_aspect('equal')
    ax_cyl.legend(loc='upper right', framealpha=0.9, edgecolor='black', fontsize=9)
    ax_cyl.grid(True)

    # --- PLOT 4: Top View (X-Y) ---
    plot_pair(ax_top, [t1[0], t2[0]], [t1[1], t2[1]], [p1[0], p2[0]], [p1[1], p2[1]])
    ax_top.add_patch(Circle((0, 0), 260, fill=False, linestyle='--', color='black', alpha=0.3))
    ax_top.set_title("Top View ($X-Y$)", fontweight='bold')
    ax_top.set_xlabel("$X$ (mm)")
    ax_top.set_ylabel("$Y$ (mm)")
    ax_top.set_aspect('equal')
    ax_top.grid(True)

    # --- PLOT 5: Side View (X-Z) ---
    plot_pair(ax_side, [t1[0], t2[0]], [t1[2], t2[2]], [p1[0], p2[0]], [p1[2], p2[2]])
    ax_side.set_title("Side View ($X-Z$)", fontweight='bold')
    ax_side.set_xlabel("$X$ (mm)")
    ax_side.set_ylabel("$Z$ (mm)")
    ax_side.set_ylim(-160, 160)
    ax_side.set_aspect('equal')
    ax_side.grid(True)

    # --- PLOT 6: STATISTICS TABLE ---
    col_labels = ["Metric", "Point 1", "Point 2", "Average / Info"]
    cell_text = [
        ["Euclidean Error", f"{err_p1:.2f} mm", f"{err_p2:.2f} mm", f"Mean: {(err_p1+err_p2)/2:.2f} mm"],
        ["Ground Truth XYZ", f"({t1[0]:.1f}, {t1[1]:.1f}, {t1[2]:.1f})", f"({t2[0]:.1f}, {t2[1]:.1f}, {t2[2]:.1f})", f"Type: {type_str}"],
        ["Predicted XYZ", f"({p1[0]:.1f}, {p1[1]:.1f}, {p1[2]:.1f})", f"({p2[0]:.1f}, {p2[1]:.1f}, {p2[2]:.1f})", f"Class: {pred_str} (P={class_prob_val:.2f})"],
    ]

    table = ax_stats.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style the table headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#333333')
        else:
            cell.set_edgecolor('#D0D0D0')

    # --- SAVE ---
    plt.savefig(f"{output_directory}/true_vs_predicted.pdf", dpi=300, bbox_inches='tight')
    print(f"Sample visualization saved to: {output_directory}/true_vs_predicted.pdf")
    plt.show()



def plot_error_distribution(errors, output_directory='./'):
    """
    Plots a publication-quality histogram of 3D errors with statistical annotations.
    """

    # --- STYLE CONFIGURATION ---
    plt.style.use('default')
    # Set Seaborn style for a clean, academic look
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.color': '#B0B0B0'
    })

    # --- DATA PREP ---
    # Ensure data is numpy array
    if torch.is_tensor(errors):
        data = errors.cpu().numpy()
    else:
        data = np.array(errors)

    # Calculate Statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    p95_val = np.percentile(data, 95)
    count = len(data)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Histogram (Density=True allows us to overlay the KDE curve properly)
    sns.histplot(data, bins=50, stat='density', color='#34495E', alpha=0.3,
                 edgecolor=None, label='Error Distribution', ax=ax)

    # 2. KDE Curve (Smooth line representation)
    sns.kdeplot(data, color='#2C3E50', linewidth=2.5, ax=ax, label='KDE Density')

    # 3. Vertical Reference Lines
    # Mean (Solid Red)
    ax.axvline(mean_val, color='#C0392B', linestyle='-', linewidth=2,
               label=f'Mean ({mean_val:.2f} mm)')

    # Median (Dashed Green)
    ax.axvline(median_val, color='#27AE60', linestyle='--', linewidth=2,
               label=f'Median ({median_val:.2f} mm)')

    # 95th Percentile (Dotted Orange - often useful in medical physics papers)
    ax.axvline(p95_val, color='#D35400', linestyle=':', linewidth=2,
               label=f'95% ({p95_val:.2f} mm)')

    # --- ANNOTATION BOX ---
    # Instead of a messy legend, a stats box is often preferred in journals
    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"N = {count}\n"
        f"Mean = {mean_val:.2f} mm\n"
        f"Median = {median_val:.2f} mm\n"
        f"$\sigma$ = {std_val:.2f} mm\n"
        f"95% < {p95_val:.2f} mm"
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
    ax.text(0.95, 0.75, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # --- LABELS & LIMITS ---
    ax.set_xlabel("Euclidean Error (mm)", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')
    ax.set_title("Distribution of Prediction Errors", fontweight='bold', pad=15)

    # Set x-limit to focus on the bulk of data (ignore extreme outliers for the plot view)
    # We clip the view to slightly past the 99th percentile for clarity
    limit_x = np.percentile(data, 99.5)
    ax.set_xlim(0, max(limit_x, 10)) # Ensure at least 0-10mm range

    # Add a simple legend for the lines (located where it doesn't overlap)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(f"{output_directory}/error_distribution.pdf", dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_directory}/error_distribution.pdf")
    plt.show()