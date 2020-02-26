import numpy as np
import warnings

# default mask bits from the stack
BAD_COLUMN = np.int64(2**0)
SAT = np.int64(2**1)
COSMIC_RAY = np.int64(2**3)
EDGE = np.int64(2**4)

# these are not official values, as far as I know
# the stack doesn't mark such things (ES 2020-02-18)
STAR = np.int64(2**10)
BLEED = np.int64(2**11)

# saturation value for images
# TODO get a good value
SAT_VAL = 1.e18


# double check they match the stack
def _check_bits_against_stack():
    try:
        import lsst.afw.image as afw_image

        sat_val = afw_image.Mask.getPlaneBitMask('SAT')
        cr_val = afw_image.Mask.getPlaneBitMask('CR')
        bad_val = afw_image.Mask.getPlaneBitMask('BAD')
        edge_val = afw_image.Mask.getPlaneBitMask('EDGE')

        if (cr_val != COSMIC_RAY or
                bad_val != BAD_COLUMN or
                edge_val != EDGE or
                sat_val != SAT):
            warnings.warn(
                "simulation bit mask flags do not match those of the DM stack")

    except ImportError:
        warnings.warn(
            "the DM stack could not be imported to check the simulation "
            "bit mask flags")


# do this here
_check_bits_against_stack()
