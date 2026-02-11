import logging

import numpy as np


def line_intersecting_with_cylinder(line_points, cylinder):
    """
    Computes the intersection points between multiple lines and a single cylinder in 3D space.
    For each line, it determines if it intersects the cylinder and,
    if so, computes the exact points of intersection.

    Lines (line parameter):

        - An array of shape (N, 6), where N is the number of lines you're considering.
        - Each line is represented by six values:
            - Starting Point: [x0, y0, z0] — the coordinates where the line begins.
            - Direction Vector: [dx, dy, dz] — the components that define the line's direction.

    Cylinder (cylinder parameter):

        - An array of shape (7,), representing a single cylinder.
        - The cylinder is defined by seven values:
            - Start of Cylinder Axis: [x1, y1, z1] — one end of the cylinder's central axis.
            - End of Cylinder Axis: [x2, y2, z2] — the other end of the cylinder's central axis.
            - Radius: R — the radius of the cylinder.

    Notes:
        - Infinite Cylinder Assumption:
            The function treats the cylinder as infinitely long along its axis when calculating intersections.
            It does not check if the intersection points lie within the finite length between the cylinder's start and end points.
            If you need to consider a finite cylinder, you'll have to add additional checks to ensure the intersection points are within the cylinder's bounds.
        - No Intersection Handling:
            Lines that do not intersect the cylinder will have their corresponding output row set to zeros.

    Parameters:
    - line: array-like of shape (N, 6), where each row is [x0, y0, z0, dx, dy, dz]
    - cylinder: array-like of shape (7,), [x1, y1, z1, x2, y2, z2, R]

    Returns:
    - points: array of shape (N, 6), where each row is [x1, y1, z1, x2, y2, z2]

    """

    try:
        # Starting point of the line (N x 3)
        l0 = line_points[:, 0:3]

        # Ending point of the line (N x 3)
        l1 = line_points[:, 3:6]

        # Direction vector of the line (N x 3)
        dl = l1 - l0

        # Starting position of the cylinder axis (1 x 3)
        c0 = cylinder[0:3]

        # Direction vector of the cylinder axis (1 x 3)
        dc = cylinder[3:6] - c0

        # Radius of the cylinder (scalar)
        r = cylinder[6]

        # Compute projection of dl onto dc
        dl_dot_dc = np.dot(dl, dc)  # (N,)
        dc_dot_dc = np.dot(dc, dc)  # scalar
        proj_dl_onto_dc = (dl_dot_dc[:, np.newaxis] / dc_dot_dc) * dc  # (N x 3)

        # Compute e vector
        e = dl - proj_dl_onto_dc  # (N x 3)

        # Compute projection of l0 - c0 onto dc
        tmp = l0 - c0  # (N x 3)
        tmp_dot_dc = np.dot(tmp, dc)  # (N,)
        proj_tmp_onto_dc = (tmp_dot_dc[:, np.newaxis] / dc_dot_dc) * dc  # (N x 3)

        # Compute f vector
        f = tmp - proj_tmp_onto_dc  # (N x 3)

        # Coefficients of the quadratic equation
        A = np.einsum('ij,ij->i', e, e)  # (N,)
        B = 2 * np.einsum('ij,ij->i', e, f)  # (N,)
        C = np.einsum('ij,ij->i', f, f) - r ** 2  # (N,)

        # Compute discriminant
        delta = B ** 2 - 4 * A * C  # (N,)

        # Indices where there is no real solution
        no_real_solution = delta < 0
        delta[no_real_solution] = 0  # Avoid NaNs in sqrt

        # Compute roots
        sqrt_delta = np.sqrt(delta)
        denominator = 2 * A
        x1 = (-B - sqrt_delta) / denominator  # (N,)
        x2 = (-B + sqrt_delta) / denominator  # (N,)

        # Compute intersection points
        point1 = l0 + x1[:, np.newaxis] * dl  # (N x 3)
        point2 = l0 + x2[:, np.newaxis] * dl  # (N x 3)

        # Combine points into a single array
        points = np.hstack((point1, point2))  # (N x 6)

        # Zero out points for lines that do not intersect the cylinder
        points[no_real_solution, :] = 0

        return points
    except Exception as e:
        logging.exception(f"An error occurred while computing line intersection with virtual cylinder. Error: {e}")
        return None


if __name__ == "__main__":
    # Define lines (e.g., 2 lines)
    lines = np.array([
        [0, 0, 0, 1, 1, 1],  # Line starting at (0,0,0) in the direction (1,1,1)
        [0, 0, 0, -1, -1, 2]  # Line starting at (1,2,3) in the direction (-1,0,2)
    ])

    # Define a cylinder
    c = np.array([0, 0, -148, 0, 0, 148, 235.422])  # Cylinder along Z-axis from (0,0,0) to (0,0,10) with radius 5

    # Compute intersection points
    intersection_points = line_intersecting_with_cylinder(lines, c)

    logging.debug(intersection_points)
