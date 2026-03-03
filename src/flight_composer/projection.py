import functools

from pyproj import CRS, Transformer

from flight_composer.flight_data import ENUCoordinate, WGS84Coordinate

_WGS84 = CRS.from_epsg(4326)
_EPBC = WGS84Coordinate(lat=52.2689025181874, lon=20.91072684036565, alt=106.1)

DEFAULT_ORIGIN = _EPBC


@functools.lru_cache(maxsize=4)
def _make_enu_transformer(lat_0: float, lon_0: float) -> Transformer:
    """Return a cached WGS 84 → AEqD ``Transformer`` centred at *lat_0*, *lon_0*."""
    aeqd_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat_0} +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs(_WGS84, aeqd_crs, always_xy=True)


def wgs84_to_enu(
    coord: WGS84Coordinate,
    origin: WGS84Coordinate = _EPBC,
) -> ENUCoordinate:
    """Project *coord* into the local ENU frame centred at *origin*.

    Horizontal mapping uses the Azimuthal Equidistant projection on the
    WGS 84 ellipsoid.  The vertical component is the simple altitude
    difference ``coord.alt − origin.alt``.
    """
    transformer = _make_enu_transformer(origin.lat, origin.lon)
    # always_xy=True  →  input order is (lon, lat)
    east, north = transformer.transform(coord.lon, coord.lat)
    up = coord.alt - origin.alt
    return ENUCoordinate(east=east, north=north, up=up)
