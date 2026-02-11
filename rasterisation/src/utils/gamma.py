import concurrent.futures
import os
import threading
import numpy as np
import csv
from helpers.cylinder_intersection import line_intersecting_with_cylinder
from utils.plotting import plot_image
import logging
from typing import List


def convert_coordinates_to_images(x1, y1, z1, x2, y2, z2, z_edges, theta_edges):
    # Wrap x1, y1, z1 in numpy arrays
    x = np.array([x1, x2])
    y = np.array([y1, y2])
    z = np.array([z1, z2])

    # Compute theta and adjust to [0, 2π]
    theta_1 = np.arctan2(y, x)
    c_1 = (theta_1 + 2 * np.pi) % (2 * np.pi)

    # Create the histogram
    gamma_image, _, _ = np.histogram2d(
        z, c_1,
        bins=[z_edges, theta_edges]
    )
    gamma_image[gamma_image > 0] = 1

    return gamma_image


def gamma_class_temporal(
        hits: np.ndarray,
        frame_length: float,
        energy_cutoff: float,
        base_directory: str,
        inner_circumference: int,
        axial_length: int,
        as_image: bool = False,
        debug: bool = True
):
    logging.info(f"Using energy cutoff: {energy_cutoff}")
    # Discard events with energy below the threshold. Filter for only gamma events
    gamma: np.ndarray = hits[hits[:, 1] >= energy_cutoff, :]

    # Number of rows after filtering by energy
    N = gamma.shape[0]
    logging.debug(f"Number of rows after filtering by energy: {N}")

    # Initialize a list to store the indices of gamma events
    indices = []
    g = 0
    while g < N:
        # Get the idx for that row
        t0 = gamma[g, 0]
        groundtruth_idx = int(np.floor(t0 / frame_length))
        # further the pointer by 1
        p = g + 1
        while p < N and (gamma[p, 0] - t0) < frame_length:
            p += 1
        indices.append((g, p, groundtruth_idx))
        g = p

    # Define the ranges for the histogram bins based on the detector's dimensions
    epsilon: float = 1e-6  # Small value to ensure inclusion
    theta_range: List[float] = [0 - epsilon, 2 * np.pi + epsilon]
    z_range: List[float] = [-148 - epsilon, 148 + epsilon]  # Adjust based on the detector's axial length

    # Define the number of bins (should match those used in frame_seperator)

    # Generate bin edges once
    z_edges: np.ndarray = np.linspace(z_range[0], z_range[1], inner_circumference + 1)
    theta_edges: np.ndarray = np.linspace(theta_range[0], theta_range[1], axial_length + 1)

    # Use a lock for thread-safe file writing
    file_lock: threading.Lock = threading.Lock()
    groundtruth_file_path: str = f'{base_directory}/groundtruth.csv'

    # Ensure the output file is empty before starting
    open(groundtruth_file_path, 'w').close()

    # Function to process a single gamma burst
    def process_burst(gamma, g, p, groundtruth_idx):
        try:
            # Initialise interaction points, event type, and event ID
            p1: np.ndarray = np.zeros(3)
            p2: np.ndarray = np.zeros(3)
            delta: int = p - g
            event_type: int = 0
            event_id: int = gamma[g, 3]  # Default event ID
            track_id: int = gamma[g, 5]
            burst_energies = gamma[g:p, 1]
            max_energies = np.max(burst_energies)
            is_photoelectric = 1 if max_energies > 0.5 else 0

            if delta == 1:  # Single event, either PE or Compton-escape
                event_type: int = 0
                p1: np.ndarray = gamma[g, 7:10]
                r1 = np.sqrt((p1[0] ** 2) + (p1[1] ** 2))
                if 278.296 >= r1 >= 235.422:
                    event_type: int = 0
                    p2: np.ndarray = p1
                else:
                    pass
            else:
                burst: np.ndarray = gamma[g:p, :]
                # Unique event IDs in the burst
                event_ids: np.ndarray = np.unique(burst[:, 3])
                if event_ids.shape[0] == 1:
                    # One event ID only in this burst
                    event_id: int = event_ids[0]
                    unique_track_ids = np.unique(burst[:, 5])
                    if len(unique_track_ids) >= 2:
                        # Take the first two tracks found (ignoring any secondaries like electrons for now)
                        t1_id = unique_track_ids[0]
                        t2_id = unique_track_ids[1]

                        # Get the *first* interaction of track 1
                        track1_hits = burst[burst[:, 5] == t1_id]
                        # Sort by time or position in file to ensure first hit
                        p1 = track1_hits[0, 7:10]

                        # Get the *first* interaction of track 2
                        track2_hits = burst[burst[:, 5] == t2_id]
                        p2 = track2_hits[0, 7:10]

                        # Check geometry
                        r1 = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
                        r2 = np.sqrt(p2[0] ** 2 + p2[1] ** 2)

                        if (235.422 <= r1 <= 278.296) and (235.422 <= r2 <= 278.296):
                            event_type = 1  # TRUE COINCIDENCE
                    else:
                        p1: np.ndarray = burst[0, 7:10]
                        r1 = np.sqrt((p1[0] ** 2) + (p1[1] ** 2))
                        if 278.296 >= r1 >= 235.422:
                            event_type: int = 0
                            p2: np.ndarray = p1
                        else:
                            pass
                else:
                    # Multiple event IDs, currently not handling these
                    pass

            # Get the coordinates of the line that intersects the cylinder drawn across the 2 points
            coordinates: List[List[np.ndarray]] = [[*p1, *p2]]
            # Extract x, y, z coordinates of the intersection point
            x1, y1, z1 = coordinates[0][0], coordinates[0][1], coordinates[0][2]
            x2, y2, z2 = coordinates[0][3], coordinates[0][4], coordinates[0][5]
            if as_image:
                # The dimensions of the cylinder relative to the origin
                cylinder_coordinates: np.ndarray = np.array([0, 0, -5, 0, 0, 5, 5])
                intersection_with_cylinder = line_intersecting_with_cylinder(np.array(coordinates), cylinder_coordinates)
                if len(intersection_with_cylinder) > 0:
                    points = intersection_with_cylinder[0]

                    # Extract x, y, z coordinates of the intersection point
                    x1, y1, z1 = points[0], points[1], points[2]
                    x2, y2, z2 = points[3], points[4], points[5]
                gamma_image = convert_coordinates_to_images(x1, y1, z1, x2, y2, z2, z_edges, theta_edges)
                # Flatten the image
                gamma_image_flat = gamma_image.flatten()
                # Combine metadata and image data into one list
                event_data = [groundtruth_idx, int(event_id), int(track_id), event_type] + gamma_image_flat.astype(int).tolist()

                # Debugging: Print the number of non-zero elements
                if debug:
                    non_zero_elements = np.count_nonzero(gamma_image)
                    logging.debug(f"Number of non-zero elements in gamma_image: {non_zero_elements}")
                    logging.debug(f"Image size before plotting: {gamma_image.shape}")
                    plot_image(gamma_image_flat, inner_circumference, axial_length, g, log_scale=False)
            else:
                event_data: List[np.ndarray] = [groundtruth_idx, int(event_id), int(track_id), event_type, is_photoelectric, x1, y1, z1, x2, y2, z2]

            if len(event_data) > 0:
                # Write the result to the file
                with file_lock:
                    with open(groundtruth_file_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(event_data)

        except Exception as e:
            logging.debug(f"Exception in processing burst starting at index {g}: {e}")

    # Use ThreadPoolExecutor to process bursts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for g, p, groundtruth_idx in indices:
            futures.append(executor.submit(process_burst, gamma, g, p, groundtruth_idx))
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    logging.info("All gamma bursts have been processed and written to the groundtruth.csv file.")
