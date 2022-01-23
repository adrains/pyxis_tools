"""
Functions for extracting fringes and phases from multi-wavelength 
interferometer data with two laser diodes.

lambda_1, lambda_2  --> wavelengths for laser 1 and 2
Lambda_1_2          --> synthetic wavelength from wavelengths 1 and 2
n_1, n_2            --> fringe orders for wavelengths 1 and 2
phi_1, phi_2        --> phase for wavelength 1 and 2 respectively
delta_r_1_2         --> path length difference between laser 1 and 2


delta_r_1_2 = Lambda_1_2 * (phi_1 - phi_2) / 2pi        (*)

(*) Under the assumption that delta_n_1_2 is approximately 0, which is true
where lambda_1 ~= lambda_2.

"""
import numpy as np
import matplotlib.pyplot as plt
import pyxis_tools.utils as utils
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from scipy.ndimage import rotate
from matplotlib.colors import LogNorm, PowerNorm

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
# Camera
# ------
PIXEL_SCALE = 3.45  # um/px
DETECTOR_SHAPE = (1080, 1440)

# Laser sources
# -------------
# 1) LPS-PM785-FC (pigtail) with diode: DL4140-001S (superceded by: L785P25)
# https://www.thorlabs.com/thorproduct.cfm?partnumber=LPS-PM785-FC
LAMBDA_1 = 785          # nm
MODE_SPACING_1 = 0.3    # nm

# 2) LPS-PM830-FC
# https://www.thorlabs.com/thorproduct.cfm?partnumber=LPS-PM830-FC
LAMBDA_2 = 830          # nm
MODE_SPACING_2 = 0.25   # nm

#LAMBDA_1_2 = LAMBDA_1 * LAMBDA_2 / (LAMBDA_2 - LAMBDA_1)

# Grating (Thorlabs GE2550-0863 Echelle Grating, 79.0 Grooves/mm, 63° Blaze)
GRATING_LINE_SEPARATION = 12.7  # um
GRATING_BETA_785nm = -56.04     # degrees
GRATING_BETA_830nm = -55.68     # degrees

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------
def extract_wavelength_dimension(
    fringe_2d,
    n_fringes_to_extract=6,
    spatial_x0=408,
    lambda_0=501,
    gradient=-13.75, #13.875
    width=4,
    image_plot_x_bounds=(460,590),
    image_plot_y_bounds=(220,500),):
    """Extracts the _wavelength_ dimension of the camera image - e.g. this will
    display the discrete modes of the laser diodes. 
    
    NOTE: Not useful for fringe or phase extraction.

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


def extract_fringes_from_image(
    fringe_image,
    dark_image,
    n_fringes_to_extract=18,
    spatial_x0=570,
    spatial_lims=(540,590),
    lambda_0=638,
    lambda_lims=(625, 825),
    lambda_width=6,
    lambda_sep=10,
    rotation_angle=0,
    do_plot_laser_modes=False,):
    """Function to extract a series of fringes from a detector image. 

    Currently relies on manually defining a reference point in the spatial and
    wavelength dimensions to start the extraction from, as well as whatever
    angle the fringes have if not incident vertically on the detector.

    Steps:
     1) Dark correct image,
     2) Rotate (optional),
     3) Extract n_fringes_to_extract 1D fringes, binning in the wavelength 
        dimension,
     4) Plot a) fringe image with overplotted diagnostics, and b) extracted
        fringes.

    Parameters
    ----------
    fringe_image: float array
        Camera image, by default with shape (1080, 1440). The short axis is the
        wavelength dimension, and the long axis is the fringe dimension. This 
        is transposed upon import if not already to give wavelength on the 
        horizontal axis and fringes on the vertical axis.
    
    dark_image: float array
        Dark frame of same shape as fringe_image.
    
    n_fringes_to_extract: int, default:18
        The number of fringes to extract.

    spatial_x0: int, default:565
        Reference plane for spatial (fringe) axis.

    spatial_lims: int tuple, default:(540,590)
        Limits of the spatial (fringe) axis for plotting.

    lambda_0: int, default:629
        Reference plane for wavelength axis, corresponding to the start of the
        first fringe.

    lambda_lims: int tuple, default:(625,825)
        Limits of the wavelength axis for plotting.

    lambda_width: int, default:6
        Adopted width of each fringe (in px) in wavelength space.

    lambda_sep: float, default:4.3
        Adopted separation between subsequent fringes (in px) in wavelength 
        space.

    rotation_angle: float, default:0
        Angle in degrees to rotate fringe_image by in the case that the optics
        are not perfectly aligned. Note that this may introduce artefacts by
        rotating the pixels, but likely fine for testing.

    Returns
    -------
    fringes: float array
        Array of extracted fringes, of shape (n_fringes_to_extract, n_px_lambda)
    """
    # Rotate image by 90 deg if we haven't already. This makes the spatial
    # dimension our vertical one, and the wavelength dimension horizontal.
    if fringe_image.shape == DETECTOR_SHAPE:
        fringe_image = fringe_image.T
        dark_image = dark_image.T
    
    lambdas, spatial_xx = np.meshgrid(
        np.arange(fringe_image.shape[1]),
        np.arange(fringe_image.shape[0]),)

    # Initialise output
    fringes = []

    # Intialise plots
    plt.close("all")

    if do_plot_laser_modes:
        fig, (ax_image, ax_laser_modes, ax_fringe) = plt.subplots(3,1)
    else:
        fig, (ax_image, ax_fringe) = plt.subplots(2,1)

    # Dark/bias correct before moving on
    fringe_image_dc = fringe_image - dark_image

    # Now rotate the image if required
    if rotation_angle != 0:
        fringe_image_dc = rotate(
            fringe_image_dc,
            angle=rotation_angle,
            reshape=False)

    # Plot the image
    img = ax_image.imshow(
        fringe_image_dc,
        aspect="equal",
        origin="lower",
        norm=PowerNorm(gamma=0.5))
    cb = fig.colorbar(img, ax=ax_image)
    cb.set_label("Counts")

    if do_plot_laser_modes:
        ax_laser_modes.plot(
            np.sum(fringe_image_dc, axis=0),
            linewidth=0.5,)

    # Hatch in the first section
    rect = patches.Rectangle(
            xy=(lambda_lims[0], spatial_lims[0]),
            width=lambda_0-lambda_lims[0],
            height=spatial_lims[1]-spatial_lims[0],
            fill=False,
            hatch="///",)
    ax_image.add_patch(rect)

    # And plot our reference line
    ax_image.hlines(
        spatial_x0,
        lambda_lims[0],
        lambda_lims[1],
        color="red",
        linewidth=0.5,)

    colours = plt.cm.plasma(np.linspace(0,1,n_fringes_to_extract))

    # Extract each fringe
    for fringe_i in range(n_fringes_to_extract):
        # Calculate the boundaries of the fringe
        bound_low = lambda_0 + (lambda_width + lambda_sep) * fringe_i
        bound_high = lambda_0 + (lambda_width + lambda_sep) * fringe_i + lambda_width
        
        # Plot the boundaries of the fringe
        ax_image.vlines(
            bound_low,
            spatial_lims[0],
            spatial_lims[1],
            color="red",
            linewidth=0.5,)

        ax_image.vlines(
            bound_high,
            spatial_lims[0],
            spatial_lims[1],
            color="red",
            linewidth=0.5,)
        
        # Hatch space between
        if fringe_i == n_fringes_to_extract-1:
            hatch_width = lambda_lims[1] - bound_high
        else:
            hatch_width = lambda_sep

        rect = patches.Rectangle(
            xy=(bound_high, spatial_lims[0]),
            width=hatch_width,
            height=spatial_lims[1]-spatial_lims[0],
            fill=False,
            hatch="///",)
        ax_image.add_patch(rect)

        # Plot fringe number
        ax_image.text(
            x=bound_low+lambda_width/2,
            y=spatial_lims[1]-5,
            s=fringe_i,
            color="red",
            horizontalalignment="center",)

        # Create mask for fringe extraction
        fringe_start = lambdas > bound_low
        fringe_end = lambdas < bound_high

        fringe_mask = np.logical_and(fringe_start, fringe_end)

        # Now mask fringe
        masked_fringe_2d = fringe_image_dc.copy()
        masked_fringe_2d[~fringe_mask] = 0

        # Collapse into 1D along wavelength dimension (keeping fringe dimension)
        fringe = np.sum(masked_fringe_2d, axis=1)

        # Plot fringe, subtracting our spatial x0 value
        scaled_x = (spatial_xx[:,1] - spatial_x0) * PIXEL_SCALE

        ax_fringe.plot(
            scaled_x,
            fringe,
            linewidth=1,
            color=colours[fringe_i],
            label="Fringe {:0.0f}".format(fringe_i))
        
        fringes.append(fringe)

    # Finalise plots
    ax_image.set_xlabel(r"$\lambda$ (px)")
    ax_image.set_ylabel("X (px)")
    ax_image.set_xlim(lambda_lims)
    ax_image.set_ylim(spatial_lims)
    
    ax_fringe.set_xlim((
        (spatial_lims[0]-spatial_x0)*PIXEL_SCALE, 
        (spatial_lims[1]-spatial_x0)*PIXEL_SCALE))
    ax_fringe.set_xlabel(r"X ($\mu$m)")
    ax_fringe.set_ylabel(r"Y (counts)")
    ax_fringe.legend(loc="center right", ncol=2)

    ax_fringe.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    ax_fringe.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))

    fringes = np.array(fringes)

    return fringes


def extract_fringe_phase(
    fringes,
    skip_missing_fringes=False,
    skip_threshold=10,
    spatial_x0=570,
    x_width=50,):
    """Fourier transform each fringe to extract fringe phase. Plots fringe
    magnitude and phase.

    NOTE: This this function is not yet complete. Please see pseudocode in 
    comments.

    Parameters
    ----------
    fringes: float array
        Array of extracted fringes, of shape (n_fringes_to_extract, n_px_lambda)

    skip_missing_fringes: boolean, default:False
        Whether to skip fringes that have no signal.

    skip_threshold: int, default:10
        Threshold signal level associated with skip_missing_fringes.

    spatial_x0: int, default:570
        Reference plane for spatial (fringe) axis.

    x_width: int, default:50
        Width of fringe to extract in pixels.
    """
    # Initialise plot
    plt.close("all")
    fig, (ax_mag, ax_phase) = plt.subplots(1,2)

    # Initialise colours
    colours = plt.cm.plasma(np.linspace(0,1,len(fringes)))

    # Plot magnitude and phase for every fringe
    for fringe_i in range(len(fringes)):
        # Only plot if our fringe actually has data
        if skip_missing_fringes and np.sum(fringes[fringe_i]) < skip_threshold:
            continue
        
        # Do Fourier Transform and extract fringe magnitude and phase
        # fringe_truc = fringe_i[ref_pix-16:ref_pix+16]
        # out = np.fft.rfft(fringe_trunc)

        # Only analyse the fringe itself
        fringe_trunc = fringes[fringe_i][spatial_x0-x_width:spatial_x0+x_width]

        # Do Fourier Transform (1D FFT on real data)
        fft_out = np.fft.rfft(fringe_trunc)

        # What is this constant??? This is the location we want to measure the 
        # phase at - it will be a float, so we have to find the two adjacent 
        # pixels and interpolate between them
        # 17/12/21: Mike said constant comes from peak spacing in Fourier domain
        # ft_pix = CONST/wavelenvth

        # Interpolate between pixels to get the phase
        # ft_pix_int = int(ft_pix)
        # ft_pix_frac = ft_pix = ft_pix_int
        # interp_ft_pix = out[ft_pix_int]*(1-ft_pix_frac) + out[ft_pix_int+1]*ft_pix_frac
        # phase_we_care_about = np.angle(interp_ft_pix)

        #out = np.fft.fft(fringes[fringe_i])
        mag = np.sqrt(fft_out.real**2 + fft_out.imag**2)
        phase = np.arctan2(fft_out.imag, fft_out.real)
        
        # Plot magnitude and phase
        ax_mag.plot(
            mag, #np.fft.fftshift(mag),
            label="Fringe {}".format(fringe_i),
            linewidth=0.5,
            color=colours[fringe_i],)
        ax_phase.plot(
            phase, #np.fft.fftshift(phase),
            label="Fringe {}".format(fringe_i),
            linewidth=0.5,
            color=colours[fringe_i],)
    
    # Finish setting up plots
    fig.suptitle("FFT of Fringes")
    ax_mag.set_title("Magnitude")
    ax_mag.set_ylabel("|counts|")
    ax_mag.legend(loc="center right")
    ax_phase.set_title("Phase")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.legend(loc="center right")