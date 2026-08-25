"""
Module: shmlib.solar

Where the sun is, seen from the site, by the NOAA formulation.

Everything here is computed from the date and the site coordinates alone, so it
is independent of every measured channel and is defined at timestamps where a
radiation sensor is missing, broken or absent altogether. That independence is
the reason it is preferred to any data-derived daylight indicator: the same mask
can be applied to the on-structure sensor, to the ground station and to the
reanalysis without any of them helping to define it.

The calculation takes no account of cloud, of the local horizon formed by the
surrounding terrain, or of atmospheric refraction near the horizon. It answers
where the sun is, not how much radiation reached the wall.

**Timestamps are interpreted as UTC.** The logger records Italian civil time and
observes daylight saving (``shmlib.site.SITE_TZ``), so an index taken from the
archive must be localised and converted before it is passed to anything here.
"""

import numpy as np
import pandas as pd


# The site coordinates originate in ``auxiliary/oiko.py``, lines 84 and 85,
# where they are the ``LAT`` and ``LON`` arguments of the Oikolab API request
# that produced ``data/raw/proxies/oikolab_weather.csv``. They therefore fix two
# things at once: the point at which the ERA5 grid cell was extracted, and the
# solar geometry used by the clock tests and by the daylight mask. Changing one
# without the other would silently compare a reanalysis series against the sun
# seen from somewhere else, so the two must always be edited together.

#: Site longitude, degrees east. See the provenance note above.
SITE_LONGITUDE = 12.582047

#: Site latitude, degrees north. See the provenance note above.
SITE_LATITUDE = 43.35343

# The two elevation thresholds below were measured by study 1 directly on the
# clean days of the on-structure radiation record and moved here from that
# study's ``de_lib.py`` because study 2 needs the identical thresholds — the
# two studies must not disagree about where the sun was — and ``shmlib`` must
# never be reached into from one study by way of another. ``de_lib.py`` keeps a
# thin alias at each original name, so every call site there is unaffected.

#: Solar elevation below which no natural light reaches the sensor, in degrees.
#: Civil twilight. Measured on clean days, the radiation decays from a median
#: of 4.75 W/m² between -2° and 0° to 0.31 between -6° and -4°, and is flat
#: below that; what remains there is the instrument's own floor, not sky
#: brightness. Cutting at the geometric horizon instead would erase real
#: twilight.
NIGHT_ELEVATION = -6.0

#: Solar elevation above which a working sensor must report substantial
#: radiation. Chosen so that even a December day at this latitude, where the
#: sun barely exceeds 23°, still contributes samples.
SUN_HIGH_ELEVATION = 15.0


def _equation_of_time(gamma):
    """
    The equation of time, in minutes, for a fractional year angle.

    Shared by :func:`solar_noon_utc` and :func:`solar_elevation` so that the two
    cannot drift apart. It moves solar noon by up to about sixteen minutes over
    the year, which would otherwise be mistaken for a clock drift.

    Parameters
    ----------
    gamma : np.ndarray
        Fractional year angle in radians.

    Returns
    -------
    np.ndarray
        Equation of time, in minutes.
    """
    return 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )


def solar_noon_utc(dates, longitude=SITE_LONGITUDE):
    """
    Solar noon in UTC hours, for each date, by the NOAA formulation.

    Accounts for the equation of time. Used as the absolute reference in the
    clock tests: a channel whose radiation curve peaks far from this hour is
    keeping a clock of its own.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates to evaluate.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.

    Returns
    -------
    pd.Series
        Solar noon, in hours after UTC midnight.
    """
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0)
    minutes = 720.0 - 4.0 * longitude - _equation_of_time(gamma)
    return pd.Series(minutes / 60.0, index=pd.DatetimeIndex(dates))


def solar_elevation(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE):
    """
    Solar elevation angle at each timestamp, by the NOAA formulation.

    The elevation is the angle of the sun above the horizon, negative at night.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate. Interpreted as UTC, consistent with
        :func:`solar_noon_utc`.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.

    Returns
    -------
    pd.Series
        Elevation in degrees, indexed by ``index``.
    """
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy(dtype=float)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0)
    eqtime = _equation_of_time(gamma)

    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    minutes = (index.hour.to_numpy(dtype=float) * 60.0
               + index.minute.to_numpy(dtype=float)
               + index.second.to_numpy(dtype=float) / 60.0)
    true_solar_time = minutes + eqtime + 4.0 * longitude
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)

    lat = np.radians(latitude)
    cos_zenith = (np.sin(lat) * np.sin(declination)
                  + np.cos(lat) * np.cos(declination) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return pd.Series(90.0 - np.degrees(np.arccos(cos_zenith)), index=index)


def daylight_mask(index, latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE,
                  min_elevation=0.0):
    """
    True where the sun stands above a given elevation.

    Roughly half of every solar-radiation record is a structural night-time
    zero. Those zeros are real measurements and are never removed from the
    record, but they distort every agreement statistic computed over the full
    day: correlations are inflated by a large block of samples on which all
    sources trivially agree, limits of agreement are computed over a bimodal
    distribution, and the trough hour of a mean diurnal cycle is an arbitrary
    point inside the night. This mask allows those statistics to be reported a
    second time over the daylight hours alone, beside the all-hours figures
    rather than in place of them.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate. Interpreted as UTC.
    latitude : float, optional
        Degrees north. Default :data:`SITE_LATITUDE`.
    longitude : float, optional
        Degrees east. Default :data:`SITE_LONGITUDE`.
    min_elevation : float, optional
        Elevation threshold in degrees. Default ``0.0``, the geometric horizon.

    Returns
    -------
    pd.Series
        Boolean, indexed by ``index``.
    """
    return solar_elevation(index, latitude=latitude,
                           longitude=longitude) > min_elevation
