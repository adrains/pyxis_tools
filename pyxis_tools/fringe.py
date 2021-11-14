"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pyxis_tools.utils as utils
import matplotlib.ticker as plticker
from matplotlib.colors import LogNorm, PowerNorm

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
# Camera
PIXEL_SCALE = 3.45  # um/px

# Laser sources
LAMBDA_1 = 780      # nm
LAMBDA_2 = 830      # nm

# Grating
GRATING_LINE_SEPARATION = 12.7  # um
GRATING_BETA_785nm = -56.04     # degrees
GRATING_BETA_830nm = -55.68     # degrees

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def extract_fringe(
    fringe_2d,
    n_fringes_to_extract=6,
    spatial_x0=408,
    lambda_0=501,
    gradient=-13.75, #13.875
    width=4,
    image_plot_x_bounds=(460,590),
    image_plot_y_bounds=(220,500),):
    """Function to extract a series of fringes from a detector image.

    Currently relies on manually defining a reference point in the spatial and
    wavelength dimensions to start the extraction from, as well as whatever
    angle the fringes have if not incident vertically on the detector.
    """
    # Calculate vertical intercept (in spatial direction, x)
    spatial_x_int = spatial_x0 - gradient * lambda_0
    
    lambdas, spatial_xx = np.meshgrid(
        np.arange(fringe_2d.shape[1]),
        np.arange(fringe_2d.shape[0]),)

    # Initialise output
    fringes = []

    # Intialise plots
    plt.close("all")
    fig, (ax_image, ax_fringe) = plt.subplots(1,2)

    img = ax_image.imshow(
        fringe_2d,
        aspect="equal",
        origin="lower",
        norm=PowerNorm(gamma=0.5))
    cb = fig.colorbar(img, ax=ax_image)
    cb.set_label("Counts")

    # Dark/bias correct before moving on
    fringe_2d = fringe_2d.copy() - np.median(fringe_2d)

    # Extract each fringe
    for fringe_i in range(n_fringes_to_extract):
        # Calculate the boundaries of the fringe
        bound_low = (lambdas-fringe_i*width)*gradient + spatial_x_int
        bound_high = (lambdas-(fringe_i+1)*width)*gradient + spatial_x_int
        
        # Plot the boundaries of the fringe
        
        ax_image.plot(lambdas[0], bound_low[0], c="red", linewidth=0.2,)
        ax_image.plot(lambdas[0], bound_high[0], c="red", linewidth=0.2,)

        # Create mask
        fringe_start = spatial_xx > bound_low
        fringe_end = spatial_xx < bound_high

        fringe_mask = np.logical_and(fringe_start, fringe_end)

        # Now mask fringe
        masked_fringe_2d = fringe_2d.copy()
        masked_fringe_2d[~fringe_mask] = 0

        # Collapse
        fringe = np.sum(masked_fringe_2d, axis=1)

        # Plot fringe, subtracting our spatial x0 value
        scaled_x = (spatial_xx[:,1] - spatial_x0) * PIXEL_SCALE

        ax_fringe.plot(
            scaled_x,
            fringe,
            linewidth=0.4,
            label="Fringe {:0.0f}".format(fringe_i))
        
        fringes.append(fringe)

    # Finalise plots
    ax_image.set_xlabel(r"$\lambda$")
    ax_image.set_ylabel("X (px)")
    ax_image.set_xlim(image_plot_x_bounds)
    ax_image.set_ylim(image_plot_y_bounds)
    
    #ax_fringe.set_yscale("log")
    ax_fringe.set_xlabel(r"X ($\mu$m)")
    ax_fringe.legend(loc="best")

    ax_fringe.xaxis.set_major_locator(plticker.MultipleLocator(base=300))
    ax_fringe.xaxis.set_minor_locator(plticker.MultipleLocator(base=150))
    #x_ticks = ax_fringe.get_xticks().tolist()
    #x_ticks_new = [str(int(float(tick)*PIXEL_SCALE)) for tick in x_ticks]
    #ax_fringe.set_xticklabels(x_ticks_new)

    fringes = np.array(fringes)

    return fringes