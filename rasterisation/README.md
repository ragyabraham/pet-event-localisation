# Data Pre-processing

After we have completed the GATE simulation a `.root` file is generated. That file is then passed through a script, which we call `read_root` that converts the root file into a `.dat` file type.

This repository is now responsible for reading that `.dat` file and converting it into a format that is suitable for machine learning.

Steps:
1. based on the PET arch used we will bin the data into individual histograms that correspond to SiPM pixels
2. based on the frame length set out in the function variable, we will calculate the frame that each photon occurred in
3. using the `gamma.py` script, we will generate the `ground truth` data. this will serve as our true north
4. Iterate through all frames. For every frame:
    - find all rows that are associated with this frame
    - extract all photons (both gamma and optical) from this frame
    - Split the data into inner-detector and outer-detector (we do this by drawing a line exactly halfway through the face of the ring and all events that occur above that line we denote as belonging to the outer-detector and all events that occur below the line belong to the inner detector)
    - convert everything to cartesian coordinates
    - A circumferential coordinate is calculated for both outer and inner detectors by multiplying `rho` (radial distance) with `theta` (angle in radians). This provides a rough estimate of the position along the circumference of the detection ring.
    - Finally we create a 2D histogram for both the inner and outer detectors:
        - `im1, _, _ = np.histogram2d(z_in, c_in, bins=[edges_Z, edges_in])`
            - The function `np.histogram2d` creates a 2D histogram.
            - `c_in` (circumferential coordinate) and `z_in` (height) are used as the two axes of the histogram.
            - `edges_in` and `edges_Z` are arrays defining the bin edges for the circumferential and z-axes, respectively.
            - `im1` will contain the 2D histogram values, providing a count of events within each circumferential and z-coordinate bin combination.
              
              
                