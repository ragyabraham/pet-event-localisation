import numpy as np
from matplotlib import pyplot as plt


def plot_image(
        image_data: np.array,
        num_bins_z: float,
        num_bins_theta: int,
        frame_index: int,
        detector_face: str = "Inner",
        log_scale: bool = False
):
    # Reshape the image
    image_2d = image_data.reshape((num_bins_z, num_bins_theta))

    # Plot the image
    plt.figure(figsize=(8, 6))
    plt.xlabel('Theta Bins')
    plt.ylabel('Z Bins')
    plt.title(f'Image for Frame Index {frame_index}. On the {detector_face} detector surface')

    # Display image with or without a log scale
    if log_scale:
        plt.imshow(np.log1p(image_2d), aspect='equal', cmap='gray', origin='lower')
    else:
        plt.imshow(image_2d, aspect='equal', cmap='gray', origin='lower', vmin=0, vmax=1)

    plt.show()
