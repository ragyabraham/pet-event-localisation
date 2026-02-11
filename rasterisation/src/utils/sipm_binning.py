from typing import Any

import numpy as np
from numpy import bool_, dtype, floating, ndarray


def sipm_binning(num_sipm: int, num_pix: int, dim_pix: int, length: float, spacing: float) -> ndarray[Any, dtype[bool_]]:

    """

    Calculates the edges of bins for a series of Silicon Photo-multipliers (SiPMs) arranged linearly.

    :param num_sipm: The total number of SiPMs.
    :param num_pix: The number of pixels within each SiPM.
    :param dim_pix: The physical dimension (e.g., width) of each pixel.
    :param length: The total length of the arrangement of all SiPMs.
    :param spacing: The spacing between adjacent SiPMs.
    :return: The function returns the edges array, which contains the calculated bin edges for all SiPM pixels in the arrangement.
    """

    pix_start = spacing
    pix_end = spacing + (num_pix * dim_pix)

    sipm_vec = np.arange(pix_start, pix_end, dim_pix)
    sipm_vec = np.append(sipm_vec, pix_end + spacing)  # Edges for one SiPM
    sipm_vec_rep = np.tile(sipm_vec, num_sipm)  # Repeat for all SiPMs

    sipm_width = 2 * spacing + num_pix * dim_pix
    start_edge = -(length / 2)

    offset = start_edge + np.arange(num_sipm) * sipm_width
    # Create an offset for each element in each SiPM
    offset_vec = np.repeat(offset, len(sipm_vec))

    edges = sipm_vec_rep + offset_vec
    edges[np.abs(edges) < 1E-5] = 0

    return edges



# def sipm_binning(num_sipm: int, num_pix: int, dim_pix: int, length: float, spacing: float) -> ndarray[Any, dtype[unsignedinteger[Any]]]:
#     """
#     This function calculates the positions of pixel edges (histogram binning) in [C,Z] space.
#     It creates a 1D array of edge positions for pixels arranged within multiple SiPMs.
#     Think of it like setting up the “borders” for pixels in a series of detector modules,
#     where each module (SiPM) has several pixels. It calculates these positions in
#     a global coordinate system that’s centered around zero (using the length parameter).
#     These pixel borders become the image bins where values can be allocated
#
#     Parameters:
#     -----------
#     num_sipm : int
#         Number of SiPMs.
#     num_pix : int
#         Number of pixels in each SiPM.
#     dim_pix : float
#         Dimension (width/height) of each pixel.
#     length_ : float
#         Length used to offset the start position (equivalent to the 'length' parameter in MATLAB).
#     spacing : float
#         Spacing (border) around each SiPM.
#
#     Returns:
#     --------
#     edges : numpy.ndarray
#         1D array containing the positions of pixel edges.
#     """
#     # The pixel region starts at a distance equal to the given spacing. This acts as a left border inside a SiPM.
#     pix_start = spacing
#     # add the total width of all pixels (number of pixels multiplied by the width of one pixel) to the spacing.
#     # This gives the rightmost edge of the pixel area (before the right border).
#     pix_end = spacing + (num_pix * dim_pix)
#
#     # Create the Edge Positions for One SiPM (sipm_vec):
#     # 1. Generates a sequence starting at pix_start and increasing in steps of dim_pix until it reaches just past pix_end.
#     sipm_vec = np.arange(pix_start, pix_end + (dim_pix * 0.5), dim_pix)
#     sipm_vec = np.concatenate([sipm_vec, [pix_end + spacing]])
#
#     # This repeats the vector for a single SiPM (sipm_vec) for all SiPMs.
#     sipm_vec_rep = np.concatenate([[0], np.tile(sipm_vec, num_sipm)])
#
#     # sipm_el = num_pix + 2; % number of elements - pixels and 2 borders
#     sipm_el = num_pix + 2
#
#     # sipm_width = 2 * spacing + num_pix * dim_pix;
#     sipm_width = 2 * spacing + (num_pix * dim_pix)
#
#     # start_edge = - (length / 2);
#     start_edge = -(length / 2.0)
#
#     # offset = start_edge + (0:1:num_sipm - 1) * sipm_width;
#     offset = start_edge + np.arange(num_sipm) * sipm_width
#
#     # offset_vec = [start_edge repelem(offset, sipm_el)];
#     # The first element is start_edge, followed by each value in offset repeated sipm_el times.
#     offset_vec = np.concatenate([[start_edge], np.repeat(offset, sipm_el)])
#
#     # edges = sipm_vec_rep + offset_vec;
#     edges = sipm_vec_rep + offset_vec
#
#     # Removing the rounding error from the centerpoint (should equal 0).
#     # edges (edges < 1E-5 & edges > -1E-5) = 0;
#     is_center = (edges < 1e-5) & (edges > -1e-5)
#     edges[is_center] = 0.0
#
#     return edges


if __name__ == "__main__":
    # Example usage (test):
    # The values here are just placeholders.
    # You can adjust them to match your scenario or verify with your MATLAB outputs.
    result = sipm_binning(num_sipm=2, num_pix=2, dim_pix=1, length=10.0, spacing=0.5)
    print("Edges:", result)
