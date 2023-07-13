import warnings
import numpy as np
import os
import copy
import galsim
from galsim import DeVaucouleurs
from galsim import Exponential
import descwl
import pandas as pd
import math

from .shifts import get_shifts, get_pair_shifts
from .constants import SCALE
from .cache_tools import cached_catalog_read


DEFAULT_FIXED_GAL_CONFIG = {
    "mag": 17.0,
    "hlr": 0.5,
    "morph": "exp",
}


def make_galaxy_catalog(
    *,
    rng,
    gal_type,
    coadd_dim=None,
    buff=0,
    layout=None,
    gal_config=None,
    sep=None,
):
    """
    rng: numpy.random.RandomState
        Numpy random state
    gal_type: string
        'fixed', 'varying' or 'wldeblend'
    coadd_dim: int
        Dimensions of coadd
    buff: int, optional
        Buffer around the edge where no objects are drawn.  Ignored for
        layout 'grid'.  Default 0.
    layout: string, optional
        'grid' or 'random'.  Ignored for gal_type "wldeblend", otherwise
        required.
    gal_config: dict or None
        Can be sent for fixed galaxy catalog.  See DEFAULT_FIXED_GAL_CONFIG
        for defaults mag, hlr and morph
    sep: float, optional
        Separation of pair in arcsec for layout='pair'
    """
    if layout == 'pair':
        if sep is None:
            raise ValueError(
                f'send sep= for gal_type {gal_type} and layout {layout}'
            )
        gal_config = get_fixed_gal_config(config=gal_config)

        if gal_type in ['fixed', 'exp']:  # TODO remove exp
            cls = FixedPairGalaxyCatalog
        else:
            cls = PairGalaxyCatalog

        galaxy_catalog = cls(
            rng=rng,
            mag=gal_config['mag'],
            hlr=gal_config['hlr'],
            morph=gal_config['morph'],
            sep=sep,
        )

    else:
        if coadd_dim is None:
            raise ValueError(
                f'send coadd_dim= for gal_type {gal_type} and layout {layout}'
            )

        if gal_type == 'wldeblend':
            if layout is None:
                layout = "random"

            galaxy_catalog = WLDeblendGalaxyCatalog(
                rng=rng,
                coadd_dim=coadd_dim,
                buff=buff,
                layout=layout,
            )
        elif gal_type in ['fixed', 'varying', 'exp']:  # TODO remove exp
            if layout is None:
                raise ValueError("send layout= for gal_type '%s'" % gal_type)

            gal_config = get_fixed_gal_config(config=gal_config)

            if gal_type == 'fixed':
                cls = FixedGalaxyCatalog
            else:
                cls = GalaxyCatalog

            galaxy_catalog = cls(
                rng=rng,
                coadd_dim=coadd_dim,
                buff=buff,
                layout=layout,
                mag=gal_config['mag'],
                hlr=gal_config['hlr'],
                morph=gal_config['morph'],
            )

        else:
            raise ValueError(f'bad gal_type "{gal_type}"')

    return galaxy_catalog


def get_fixed_gal_config(config=None):
    """
    get the configuration for fixed galaxies, with defaults in place

    Parameters
    ----------
    config: dict, optional
        The input config. Over-rides defaults

    Returns
    -------
    the config dict
    """
    out_config = copy.deepcopy(DEFAULT_FIXED_GAL_CONFIG)

    if config is not None:
        for key in config:
            if key not in out_config:
                raise ValueError("bad key for fixed gals: '%s'" % key)
        out_config.update(config)
    return out_config


class FixedGalaxyCatalog(object):
    """
    Galaxies of fixed galsim type, flux, and size and shape.

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        dimensions of the coadd
    layout: string
        The layout of objects, either 'grid' or 'random'
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, coadd_dim, layout, mag, hlr, buff=0, morph='exp'):
        self.gal_type = 'fixed'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr

        self.shifts_array = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
        )

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts]
        """

        flux = survey.get_flux(self.mag)

        sarray = self.shifts_array
        objlist = []
        shifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(flux))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

        return objlist, shifts

    def _get_galaxy(self, flux):
        """
        get a galaxy object

        Parameters
        ----------
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """

        if self.morph == 'exp':
            gal = _generate_exp(hlr=self.hlr, flux=flux)
        elif self.morph == 'dev':
            gal = _generate_dev(hlr=self.hlr, flux=flux)
        elif self.morph == 'bd':
            gal = _generate_bd(hlr=self.hlr, flux=flux)
        elif self.morph == 'bdk':
            gal = _generate_bdk(hlr=self.hlr, flux=flux)
        else:
            raise ValueError(f"bad gal type '{self.morph}'")

        return gal


class GalaxyCatalog(FixedGalaxyCatalog):
    """
    Galaxies of fixed galsim type, but varying properties.

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        dimensions of the coadd
    layout: string
        The layout of objects, either 'grid' or 'random'
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, coadd_dim, layout, mag, hlr, buff=0, morph='exp'):
        super().__init__(
            rng=rng, coadd_dim=coadd_dim, buff=buff, layout=layout,
            mag=mag, hlr=hlr, morph=morph,
        )
        self.gal_type = 'varying'

        # we use this to ensure the same galaxies are generated in different
        # bands
        self.morph_seed = rng.randint(0, 2**31)
        self.gs_morph_seed = rng.randint(0, 2**31)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        band: string
            Get objects for this band.  For the fixed
            catalog, the objects are the same for every band

        Returns
        -------
        [galsim objects], [shifts]
        """

        self._morph_rng = np.random.RandomState(self.morph_seed)
        self._gs_morph_rng = galsim.BaseDeviate(seed=self.gs_morph_seed)
        return super().get_objlist(survey=survey)

    def _get_galaxy(self, flux):
        """
        get a galaxy object

        Parameters
        ----------
        flux: float
            Flux of object

        Returns
        --------
        galsim.GSObject
        """

        if self.morph == 'exp':
            gal = _generate_exp(
                hlr=self.hlr, flux=flux, vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'dev':
            gal = _generate_dev(
                hlr=self.hlr, flux=flux, vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'bd':
            gal = _generate_bd(
                hlr=self.hlr, flux=flux,
                vary=True, rng=self._morph_rng,
            )
        elif self.morph == 'bdk':
            gal = _generate_bdk(
                hlr=self.hlr, flux=flux,
                vary=True,
                rng=self._morph_rng, gsrng=self._gs_morph_rng,
            )
        else:
            raise ValueError(f"bad morph '{self.morph}'")

        return gal


def _generate_exp(hlr, flux, vary=False, rng=None):
    gal = Exponential(half_light_radius=hlr, flux=flux)

    if vary:
        g1, g2 = _generate_g1g2(rng)
        gal = gal.shear(g1=g1, g2=g2)

    return gal


def _generate_dev(hlr, flux, vary=False, rng=None):
    gal = DeVaucouleurs(half_light_radius=hlr, flux=flux)
    if vary:
        g1, g2 = _generate_g1g2(rng)
        gal = gal.shear(g1=g1, g2=g2)

    return gal


def _generate_bd(
    hlr, flux,
    vary=False,
    rng=None,
    max_bulge_shift_frac=0.1,  # fraction of hlr
    max_bulge_rot=np.pi/4,
):

    if vary:
        bulge_frac = _generate_bulge_frac(rng)
    else:
        bulge_frac = 0.5

    disk_frac = (1.0 - bulge_frac)

    bulge = DeVaucouleurs(half_light_radius=hlr, flux=flux * bulge_frac)
    disk = Exponential(half_light_radius=hlr, flux=flux * disk_frac)

    if vary:
        bulge = _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac)

    if vary:
        g1disk, g2disk = _generate_g1g2(rng)

        g1bulge, g2bulge = g1disk, g2disk
        if vary:
            g1bulge, g2bulge = _rotate_bulge(rng, max_bulge_rot, g1bulge, g2bulge)

        bulge = bulge.shear(g1=g1bulge, g2=g2bulge)
        disk = disk.shear(g1=g1disk, g2=g2disk)

    return galsim.Add(bulge, disk)


def _generate_bdk(
    hlr, flux,
    vary=False,
    rng=None,
    gsrng=None,
    knots_hlr_frac=0.25,
    max_knots_disk_frac=0.1,  # fraction of disk light
    max_bulge_shift_frac=0.1,  # fraction of hlr
    max_bulge_rot=np.pi/4,
):

    if vary:
        bulge_frac = _generate_bulge_frac(rng)
    else:
        bulge_frac = 0.5

    all_disk_frac = (1.0 - bulge_frac)

    knots_hlr = knots_hlr_frac * hlr
    if vary:
        knots_sub_frac = _generate_knots_sub_frac(rng, max_knots_disk_frac)
    else:
        knots_sub_frac = max_knots_disk_frac

    disk_frac = (1 - knots_sub_frac) * all_disk_frac
    knots_frac = knots_sub_frac * all_disk_frac

    bulge = DeVaucouleurs(half_light_radius=hlr, flux=flux * bulge_frac)
    disk = Exponential(half_light_radius=hlr, flux=flux * disk_frac)

    if gsrng is None:
        # fixed galaxy, so fix the rng
        gsrng = galsim.BaseDeviate(123)

    knots = galsim.RandomKnots(
        npoints=10,
        half_light_radius=knots_hlr,
        flux=flux * knots_frac,
        rng=gsrng,
    )

    if vary:
        bulge = _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac)

    if vary:
        g1disk, g2disk = _generate_g1g2(rng)

        g1bulge, g2bulge = g1disk, g2disk
        if vary:
            g1bulge, g2bulge = _rotate_bulge(rng, max_bulge_rot, g1bulge, g2bulge)

        bulge = bulge.shear(g1=g1bulge, g2=g2bulge)
        disk = disk.shear(g1=g1disk, g2=g2disk)
        knots = knots.shear(g1=g1disk, g2=g2disk)

    return galsim.Add(bulge, disk, knots)


def _generate_bulge_frac(rng):
    assert rng is not None, 'send rng to generate bulge fraction'
    return rng.uniform(low=0.0, high=1.0)


def _generate_g1g2(rng, std=0.2):
    assert rng is not None, 'send rng to vary shape'
    while True:
        g1, g2 = rng.normal(scale=std, size=2)
        g = np.sqrt(g1**2 + g2**2)
        if abs(g) < 0.9999:
            break

    return g1, g2


def _generate_bulge_shift(rng, hlr, max_bulge_shift_frac):
    bulge_shift = rng.uniform(low=0.0, high=max_bulge_shift_frac*hlr)
    bulge_shift_angle = rng.uniform(low=0, high=2*np.pi)
    bulge_shiftx = bulge_shift * np.cos(bulge_shift_angle)
    bulge_shifty = bulge_shift * np.sin(bulge_shift_angle)

    return bulge_shiftx, bulge_shifty


def _shift_bulge(rng, bulge, hlr, max_bulge_shift_frac):
    bulge_shiftx, bulge_shifty = _generate_bulge_shift(
        rng, hlr, max_bulge_shift_frac,
    )
    return bulge.shift(bulge_shiftx, bulge_shifty)


def _rotate_bulge(rng, max_bulge_rot, g1, g2):
    assert rng is not None, 'send rng to rotate bulge'
    bulge_rot = rng.uniform(low=-max_bulge_rot, high=max_bulge_rot/4)
    return _rotate_shape(g1, g2, bulge_rot)


def _rotate_shape(g1, g2, theta_radians):
    twotheta = 2.0 * theta_radians

    cos2angle = np.cos(twotheta)
    sin2angle = np.sin(twotheta)
    g1rot = g1 * cos2angle + g2 * sin2angle
    g2rot = -g1 * sin2angle + g2 * cos2angle

    return g1rot, g2rot


def _generate_knots_sub_frac(rng, max_knots_disk_frac):
    assert rng is not None, 'send rng to generate knots sub frac'
    return rng.uniform(low=0.0, high=max_knots_disk_frac)


class FixedPairGalaxyCatalog(FixedGalaxyCatalog):
    """
    A pair of galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    sep: float
        Separation of pair in arcsec
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, mag, hlr, sep, morph='exp'):
        self.gal_type = 'fixed'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.shifts_array = get_pair_shifts(
            rng=rng,
            sep=sep,
        )


class PairGalaxyCatalog(GalaxyCatalog):
    """
    A pair of galaxies of fixed galsim type, flux, and size

    Same for all bands

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    mag: float
        Magnitude of all objects. Objects brighter than magntiude 17 (e.g., 14
        since mags are opposite) tend to cause the Rubin Observatory science
        pipeline detection algorithm to misdetect isolted objects in unphysical
        ways. This effect causes the shear response to be non-linear and so
        metadetect will fail. For this reason, you should use the default
        magnitude of 17 or fainter for this kind of galaxy.
    hlr: float
        Half light radius of all objects
    sep: float
        Separation of pair in arcsec
    morph: str
        Galaxy morphology, 'exp', 'dev' or 'bd', 'bdk'.  Default 'exp'
    """
    def __init__(self, *, rng, mag, hlr, sep, morph='exp'):
        self.gal_type = 'varying'
        self.morph = morph
        self.mag = mag
        self.hlr = hlr
        self.rng = rng

        self.morph_seed = rng.randint(0, 2**31)
        self.gs_morph_seed = rng.randint(0, 2**31)

        self.shifts_array = get_pair_shifts(
            rng=rng,
            sep=sep,
        )


class WLDeblendGalaxyCatalog(object):
    """
    Catalog of galaxies from wldeblend

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    layout: str, optional

    """
    def __init__(self, *, rng, coadd_dim, buff=0, layout='random'):
        self.gal_type = 'wldeblend'
        self.rng = rng

        self._wldeblend_cat = read_wldeblend_cat(rng)

        # one square degree catalog, convert to arcmin
        gal_dens = self._wldeblend_cat.size / (60 * 60)
        if layout == 'random':
            # this layout is random in a square
            if (coadd_dim - 2*buff) < 2:
                warnings.warn("dim - 2*buff <= 2, force it to 2.")
                area = (2**SCALE/60)**2.
            else:
                area = ((coadd_dim - 2*buff)*SCALE/60)**2

        elif layout == 'random_disk':
            # this layout is random in a circle
            if (coadd_dim - 2*buff) < 2:
                warnings.warn("dim - 2*buff <= 2, force it to 2.")
                radius = 2.*SCALE/60
                area = np.pi*radius**2
            else:
                radius = (coadd_dim/2. - buff)*SCALE/60
                area = np.pi*radius**2
            del radius
        else:
            raise ValueError("layout can only be 'random' or 'random_disk' \
                    for wldeblend")

        # a least 1 expected galaxy (used for simple tests)
        nobj_mean = max(area * gal_dens, 1)

        nobj = rng.poisson(nobj_mean)

        self.shifts_array = get_shifts(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout=layout,
            nobj=nobj,
        )

        num = len(self)
        self.indices = self.rng.randint(
            0,
            self._wldeblend_cat.size,
            size=num,
        )

        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts]
        """

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        band = survey.filter_band

        sarray = self.shifts_array
        objlist = []
        shifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

        return objlist, shifts

    def _get_galaxy(self, builder, band, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self._wldeblend_cat[index],
            0,
            0,
            band,
        ).model.rotate(
            angle * galsim.degrees,
        )

        return galaxy


def read_wldeblend_cat(rng):
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator

    Returns
    -------
    array with fields
    """
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )

    # not thread safe
    cat = cached_catalog_read(fname)
    return cat

#----------------------------------------------------------------------------------------------------------------
def sersic_second_moments(n,hlr,q,beta):
    """Calculate the second-moment tensor of a sheared Sersic radial profile.

    Args:
        n(int): Sersic index of radial profile. Only n = 1 and n = 4 are supported.
        hlr(float): Radius of 50% isophote before shearing, in arcseconds.
        q(float): Ratio b/a of Sersic isophotes after shearing.
        beta(float): Position angle of sheared isophotes in radians, measured anti-clockwise
            from the positive x-axis.

    Returns:
        numpy.ndarray: Array of shape (2,2) with values of the second-moments tensor
            matrix, in units of square arcseconds.

    Raises:
        RuntimeError: Invalid Sersic index n.
    """
    # Lookup the value of cn = 0.5*(r0/hlr)**2 Gamma(4*n)/Gamma(2*n)
    if n == 1:
        cn = 1.06502
    elif n == 4:
        cn = 10.8396
    else:
        raise RuntimeError('Invalid Sersic index n.')
    e_mag = (1.-q)/(1.+q)
    e_mag_sq = e_mag**2
    e1 = e_mag*math.cos(2*beta)
    e2 = e_mag*math.sin(2*beta)
    Q11 = 1 + e_mag_sq + 2*e1
    Q22 = 1 + e_mag_sq - 2*e1
    Q12 = 2*e2
    return np.array(((Q11,Q12),(Q12,Q22)))*cn*hlr**2/(1-e_mag_sq)**2

class IAGalaxyCatalog(object):
    """
    Catalog of galaxies from IA sims

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    coadd_dim: int
        Dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ignored
        for layout 'grid'.  Default 0.
    """
    def __init__(self, *, rng, coadd_dim, buff=0, ia_angles = True, wcs):
        self.gal_type = 'ia'
        self.rng = rng

        self._ia_cat = read_ia_cat()
        #finding shifts based on coadd wcs
        ras = self._ia_cat['ra_gal'].values
        decs = self._ia_cat['dec_gal'].values
        x,y = wcs.radecToxy(ras, decs, units=galsim.degrees)
        cen = (coadd_dim-1)/2
        lims = ((coadd_dim - 2*buff)*SCALE/60)/2
        self.im_mask = np.where((x>(cen-cen))&(x<(cen+cen))&(y>(cen-cen))&(y<(cen+cen)))
        self.gal_ids = self._ia_cat['unique_gal_id'][self.im_mask[0]]
        size = len(self.im_mask[0])
        
        shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])

        shifts['dx'] = x[self.im_mask[0]]
        shifts['dy'] = y[self.im_mask[0]]

        self.shifts_array = shifts
        self.ia_angles = ia_angles
        num = len(self)
        self.indices = np.arange(0,len(shifts))
        self.angles = np.zeros(num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts]
        """
        
        builder = IAGalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
            ia_angles = False,
        )

        band = survey.filter_band

        sarray = self.shifts_array
        objlist = []
        shifts = []
        for i in range(len(self.im_mask[0])):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray['dx'][i], sarray['dy'][i]))

        return objlist, shifts

    def _get_galaxy(self, builder, band, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self.im_mask[0][i]

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self._ia_cat.iloc[index],
            0,
            0,
            band,
        ).model.rotate(
            angle * galsim.degrees,
        )

        return galaxy
    
def read_ia_cat():
    """
    Read the catalog from the cache, but update the position angles each time

    Parameters
    ----------

    Returns
    -------
    array with fields
    """
    fname = os.path.join(
        os.environ.get('MICE_DIR', '.'),
        '14322ia.parquet',
    )

 
    cat = pd.read_parquet('14322ia.parquet')
    return cat

class IAGalaxyBuilder(object):
    """Build galaxy source models.

    Args:
        survey(descwl.survey.Survey): Survey to use for flux normalization and cosmic shear.
        no_disk(bool): Ignore any Sersic n=1 component in the model if it is present in the catalog.
        no_bulge(bool): Ignore any Sersic n=4 component in the model if it is present in the catalog.
        no_agn(bool): Ignore any PSF-like component in the model if it is present in the catalog.
        verbose_model(bool): Provide verbose output from model building process.
    """
    def __init__(self,survey,no_disk,no_bulge,no_agn,verbose_model,ia_angles):
        if no_disk and no_bulge and no_agn:
            raise RuntimeError('Must build at least one galaxy component.')
        self.survey = survey
        self.no_disk = no_disk
        self.no_bulge = no_bulge
        self.no_agn = no_agn
        self.verbose_model = verbose_model
        self.ia_angles = ia_angles

    def from_catalog(self,entry,dx_arcsecs,dy_arcsecs,filter_band):
        """Build a :class:Galaxy object from a catalog entry.

        Fluxes are distributed between the three possible components (disk,bulge,AGN) assuming
        that each component has the same spectral energy distribution, so that the resulting
        proportions are independent of the filter band.

        Args:
            entry(pandas dataframe): A single row from a galaxy.
            dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
                in arcseconds.
            dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
                in arcseconds.
            filter_band(str): The LSST filter band to use for calculating flux, which must
                be one of 'u','g','r','i','z','y'.

        Returns:
            :class:`Galaxy`: A newly created galaxy source model.

        Raises:
            SourceNotVisible: All of the galaxy's components are being ignored.
            RuntimeError: Catalog entry is missing AB flux value in requested filter band.
        """
        # Calculate the object's total flux in detected electrons.
        try:
            ab_magnitude = entry['des_asahi_full_' + filter_band + '_true']
            ri_color = entry['des_asahi_full_r_true'] - entry['des_asahi_full_i_true']
        except KeyError:
            raise RuntimeError('Catalog entry is missing required AB magnitudes.')
        total_flux = self.survey.get_flux(ab_magnitude)
        # Calculate the flux of each component in detected electrons.
        total_fluxnorm = entry['bulge_fraction'] + (1-entry['bulge_fraction'])
        disk_flux = 0. if self.no_disk else (1-entry['bulge_fraction'])/total_fluxnorm*total_flux
        bulge_flux = 0. if self.no_bulge else entry['bulge_fraction']/total_fluxnorm*total_flux
        agn_flux = 0. if self.no_agn else 0.
        # Is there any flux to simulate?
        if disk_flux + bulge_flux + agn_flux == 0:
            raise SourceNotVisible
        # Calculate the position of angle of the Sersic components, which are assumed to be the same.
        if disk_flux > 0:
            if self.ia_angles == True:
                beta_radians = (np.pi/2) - 0.5*np.arctan2(entry['eps2_gal'],entry['eps1_gal'])
            else:
                beta_radians = math.radians(entry['disk_angle']) #I believe beta is 0 for the mice catalog need to confirm
#             if bulge_flux > 0:
#                 assert entry['pa_disk'] == entry['pa_bulge'],'Sersic components have different beta.'
        elif bulge_flux > 0:
            if self.ia_angles == True:
                beta_radians = (np.pi/2) - 0.5*np.arctan2(entry['eps2_gal'],entry['eps1_gal'])
            else:
                beta_radians = math.radians(entry['bulge_angle'])
        else:
            # This might happen if we only have an AGN component.
            beta_radians = None
        # Calculate shapes hlr = sqrt(a*b) and q = b/a of Sersic components.
        if disk_flux > 0:
            if self.ia_angles == True:
                disk_q = (1 - np.sqrt(entry['eps1_gal'].values**2 + entry['eps2_gal'].values**2))/(1+np.sqrt(entry['eps1_gal'].values**2 + entry['eps2_gal'].values**2))
            else:
                disk_q = entry['disk_axis_ratio']
            a_d,b_d = entry['disk_length']/2,disk_q*(entry['disk_length']/2)
            disk_hlr_arcsecs = math.sqrt(a_d*b_d)
            
        else:
            disk_hlr_arcsecs,disk_q = None,None
        if bulge_flux > 0:
            if self.ia_angles == True:
                bulge_q = (1 - np.sqrt(entry['eps1_gal'].values**2 + entry['eps2_gal'].values**2))/(1+np.sqrt(entry['eps1_gal'].values**2 + entry['eps2_gal'].values**2))
            else:
                bulge_q = entry['bulge_axis_ratio']
            a_b,b_b = entry['bulge_length']/2,bulge_q*(entry['bulge_length']/2)
            bulge_hlr_arcsecs = math.sqrt(a_b*b_b)
            bulge_beta = math.radians(0)#0 I believe for IA sims
        else:
            bulge_hlr_arcsecs,bulge_q = None,None
        # Look up extra catalog metadata.
        identifier = entry['unique_gal_id']
        redshift = entry['z_cgal']
        if self.verbose_model:
            print('Building galaxy model for id=%d with z=%.3f' % (identifier,redshift))
            print('flux = %.3g detected electrons (%s-band AB = %.1f)' % (
                total_flux,filter_band,ab_magnitude))
            print('centroid at (%.6f,%.6f) arcsec relative to image center, beta = %.6f rad' % (
                dx_arcsecs,dy_arcsecs,beta_radians))
            if disk_flux > 0:
                print(' disk: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    disk_flux/total_flux,disk_hlr_arcsecs,disk_q))
            if bulge_flux > 0:
                print('bulge: frac = %.6f, hlr = %.6f arcsec, q = %.6f' % (
                    bulge_flux/total_flux,bulge_hlr_arcsecs,bulge_q))
            if agn_flux > 0:
                print('  AGN: frac = %.6f' % (agn_flux/total_flux))
        return Galaxy(identifier,redshift,ab_magnitude,ri_color,
            entry['gamma1'],entry['gamma2'],
            dx_arcsecs,dy_arcsecs,beta_radians,disk_flux,disk_hlr_arcsecs,disk_q,
            bulge_flux,bulge_hlr_arcsecs,bulge_q,agn_flux)

    @staticmethod
    def add_args(parser):
        """Add command-line arguments for constructing a new :class:`GalaxyBuilder`.

        The added arguments are our constructor parameters with '_' replaced by '-' in the names.

        Args:
            parser(argparse.ArgumentParser): Arguments will be added to this parser object using its
                add_argument method.
        """
        parser.add_argument('--no-disk', action = 'store_true',
            help = 'Ignore any Sersic n=1 component in the model if it is present in the catalog.')
        parser.add_argument('--no-bulge', action = 'store_true',
            help = 'Ignore any Sersic n=4 component in the model if it is present in the catalog.')
        parser.add_argument('--no-agn', action = 'store_true',
            help = 'Ignore any PSF-like component in the model if it is present in the catalog.')
        parser.add_argument('--verbose-model', action = 'store_true',
            help = 'Provide verbose output from model building process.')

    @classmethod
    def from_args(cls,survey,args):
        """Create a new :class:`GalaxyBuilder` object from a set of arguments.

        Args:
            survey(descwl.survey.Survey): Survey to build source models for.
            args(object): A set of arguments accessed as a :py:class:`dict` using the
                built-in :py:func:`vars` function. Any extra arguments beyond those defined
                in :func:`add_args` will be silently ignored.

        Returns:
            :class:`GalaxyBuilder`: A newly constructed Reader object.
        """
        # Look up the named constructor parameters.
        pnames = (inspect.getargspec(cls.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Filter the dictionary to only include constructor parameters.
        filtered_dict = { key:args_dict[key] for key in (set(pnames) & set(args_dict)) }
        return cls(survey,**filtered_dict)

class Galaxy(object):
    """Source model for a galaxy.

    Galaxies are modeled using up to three components: a disk (Sersic n=1), a bulge
    (Sersic n=4), and an AGN (PSF-like). Not all components are required.  All components
    are assumed to have the same centroid and the extended (disk,bulge) components are
    assumed to have the same position angle.

    Args:
        identifier(int): Unique integer identifier for this galaxy in the source catalog.
        redshift(float): Catalog redshift of this galaxy.
        ab_magnitude(float): Catalog AB magnitude of this galaxy in the filter band being
            simulated.
        ri_color(float): Catalog source color calculated as (r-i) AB magnitude difference.
        cosmic_shear_g1(float): Cosmic shear ellipticity component g1 (+) with \|g\| = (a-b)/(a+b).
        cosmic_shear_g2(float): Cosmic shear ellipticity component g2 (x) with \|g\| = (a-b)/(a+b).
        dx_arcsecs(float): Horizontal offset of catalog entry's centroid from image center
            in arcseconds.
        dy_arcsecs(float): Vertical offset of catalog entry's centroid from image center
            in arcseconds.
        beta_radians(float): Position angle beta of Sersic components in radians, measured
            anti-clockwise from the positive x-axis. Ignored if disk_flux and bulge_flux are
            both zero.
        disk_flux(float): Total flux in detected electrons of Sersic n=1 component.
        disk_hlr_arcsecs(float): Half-light radius sqrt(a*b) of circularized 50% isophote
            for Sersic n=1 component, in arcseconds. Ignored if disk_flux is zero.
        disk_q(float): Ratio b/a of 50% isophote semi-minor (b) to semi-major (a) axis
            lengths for Sersic n=1 component. Ignored if disk_flux is zero.
        bulge_flux(float): Total flux in detected electrons of Sersic n=4 component.
        bulge_hlr_arcsecs(float): Half-light radius sqrt(a*b) of circularized 50% isophote
            for Sersic n=4 component, in arcseconds. Ignored if bulge_flux is zero.
        bulge_q(float): Ratio b/a of 50% isophote semi-minor (b) to semi-major (a) axis
            lengths for Sersic n=4 component. Ignored if bulge_flux is zero.
        agn_flux(float): Total flux in detected electrons of PSF-like component.
    """
    def __init__(self,identifier,redshift,ab_magnitude,ri_color,
        cosmic_shear_g1,cosmic_shear_g2,
        dx_arcsecs,dy_arcsecs,beta_radians,disk_flux,disk_hlr_arcsecs,disk_q,
        bulge_flux,bulge_hlr_arcsecs,bulge_q,agn_flux):
        self.identifier = identifier
        self.redshift = redshift
        self.ab_magnitude = ab_magnitude
        self.ri_color = ri_color
        self.dx_arcsecs = dx_arcsecs
        self.dy_arcsecs = dy_arcsecs
        self.cosmic_shear_g1 = cosmic_shear_g1
        self.cosmic_shear_g2 = cosmic_shear_g2
        components = [ ]
        # Initialize second-moments tensor. Note that we can only add the tensors for the
        # n = 1,4 components, as we do below, since they have the same centroid.
        self.second_moments = np.zeros((2,2))
        total_flux = disk_flux + bulge_flux + agn_flux
        self.disk_fraction = disk_flux/total_flux
        self.bulge_fraction = bulge_flux/total_flux
        if disk_flux > 0:
            disk = galsim.Exponential(flux = disk_flux, half_light_radius = disk_hlr_arcsecs).shear(q = disk_q, beta = beta_radians*galsim.radians)
            components.append(disk)
            self.second_moments += self.disk_fraction*sersic_second_moments(
                n=1,hlr=disk_hlr_arcsecs,q=disk_q,beta=beta_radians)

        if bulge_flux > 0:
            bulge = galsim.DeVaucouleurs(
                flux = bulge_flux, half_light_radius = bulge_hlr_arcsecs).shear(
                q = bulge_q, beta = beta_radians*galsim.radians)
            components.append(bulge)
            self.second_moments += self.bulge_fraction*sersic_second_moments(
                n=1,hlr=bulge_hlr_arcsecs,q=bulge_q,beta=beta_radians)
        # GalSim does not currently provide a "delta-function" component to model the AGN
        # so we use a very narrow Gaussian. See this GalSim issue for details:
        # https://github.com/GalSim-developers/GalSim/issues/533
        if agn_flux > 0:
            agn = galsim.Gaussian(flux = agn_flux, sigma = 1e-8)
            components.append(agn)

        # Combine the components into our final profile.
        self.profile = galsim.Add(components)
        # Apply transforms to build the final model.

        self.model = self.get_transformed_model()

        # Shear the second moments, if necessary.
#         if self.cosmic_shear_g1 != 0 or self.cosmic_shear_g2 != 0:
#             self.second_moments = sheared_second_moments(
#                 self.second_moments,self.cosmic_shear_g1,self.cosmic_shear_g2)

    def get_transformed_model(self,dx=0.,dy=0.,ds=0.,dg1=0.,dg2=0.):
        """Apply transforms to our model.

        The nominal model returned by `get_transformed_model()` is available via
        the `model` attribute.

        Args:
            dx(float): Amount to shift centroid in x, in arcseconds.
            dy(float): Amount to shift centroid in y, in arcseconds.
            ds(float): Relative amount to scale the galaxy profile in the
                radial direction while conserving flux, before applying shear
                or convolving with the PSF.
            dg1(float): Amount to adjust the + shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.
            dg2(float): Amount to adjust the x shear applied to the galaxy profile,
                with \|g\| = (a-b)/(a+b), before convolving with the PSF.

        Returns:
            galsim.GSObject: New model constructed using our source profile with
                the requested transforms applied.
        """

        return (self.profile
            .dilate(1 + ds)
            .shear(g1 = self.cosmic_shear_g1 + dg1,g2 = self.cosmic_shear_g2 + dg2)
            .shift(dx = self.dx_arcsecs + dx,dy = self.dy_arcsecs + dy))


