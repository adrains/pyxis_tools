import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_raw8(filename, shape=(1080,1440)):

    """ Read in the raw data and return as a numpy array. 

    

    It seems that the data was saved in 8 bit format - this has to be changed

    to dtype=np.uint16 if there is a 16 bit data option.

    """

    raw = open(filename, 'rb')

    im = np.frombuffer(raw.read(), dtype=np.uint8).reshape(shape)

    return im


def create_master_dark(path, dark_fn_with_wildcard):
    """Create master dark file from median of N darks.
    """
    filepath_base = os.path.join(path, dark_fn_with_wildcard)
    dark_filepaths = glob.glob(filepath_base)

    darks = np.array([read_raw8(fp).astype(float) for fp in dark_filepaths])

    median_dark = np.median(darks, axis=0)

    return median_dark


#Some test code to show the data.

if __name__=="__main__":

    im_uint8 = read_raw8('785_fringe_1.raw')

    im_float = im_uint8.astype(np.float)

    #Show an image with square root scaling...

    plt.clf()

    plt.imshow(im_float[200:500,400:600]**0.5, aspect='auto')