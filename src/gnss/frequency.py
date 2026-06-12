"""
GNSS observation carrier frequencies from RINEX v4.

Frequencies are in MHz. Wavelength is derived from the speed of light in vacuum.
"""

from __future__ import annotations

from dataclasses import dataclass

SPEED_OF_LIGHT_M_S = 299_792_458.0

# (constellation, obs_type) -> nominal carrier frequency in MHz
_OBS_FREQUENCY_MHZ: dict[tuple[str, str], float] = {
    # GPS (G)
    ("G", "S1C"): 1575.42,
    ("G", "S1S"): 1575.42,
    ("G", "S1L"): 1575.42,
    ("G", "S1X"): 1575.42,
    ("G", "S1P"): 1575.42,
    ("G", "S1W"): 1575.42,
    ("G", "S1Y"): 1575.42,
    ("G", "S1M"): 1575.42,
    ("G", "S1N"): 1575.42,
    ("G", "S2C"): 1227.60,
    ("G", "S2D"): 1227.60,
    ("G", "S2S"): 1227.60,
    ("G", "S2L"): 1227.60,
    ("G", "S2X"): 1227.60,
    ("G", "S2P"): 1227.60,
    ("G", "S2W"): 1227.60,
    ("G", "S2Y"): 1227.60,
    ("G", "S2M"): 1227.60,
    ("G", "S2N"): 1227.60,
    ("G", "S5I"): 1176.45,
    ("G", "S5Q"): 1176.45,
    ("G", "S5X"): 1176.45,
    # GLONASS (R) - channel-independent entries
    ("R", "S4A"): 1600.995,
    ("R", "S4B"): 1600.995,
    ("R", "S4X"): 1600.995,
    ("R", "S6A"): 1248.06,
    ("R", "S6B"): 1248.06,
    ("R", "S6X"): 1248.06,
    ("R", "S3I"): 1202.025,
    ("R", "S3Q"): 1202.025,
    ("R", "S3X"): 1202.025,
    # Galileo (E)
    ("E", "S1A"): 1575.42,
    ("E", "S1B"): 1575.42,
    ("E", "S1C"): 1575.42,
    ("E", "S1X"): 1575.42,
    ("E", "S1Z"): 1575.42,
    ("E", "S5I"): 1176.45,
    ("E", "S5Q"): 1176.45,
    ("E", "S5X"): 1176.45,
    ("E", "S7I"): 1207.140,
    ("E", "S7Q"): 1207.140,
    ("E", "S7X"): 1207.140,
    ("E", "S8I"): 1191.795,
    ("E", "S8Q"): 1191.795,
    ("E", "S8X"): 1191.795,
    ("E", "S6A"): 1278.75,
    ("E", "S6B"): 1278.75,
    ("E", "S6C"): 1278.75,
    ("E", "S6X"): 1278.75,
    ("E", "S6Z"): 1278.75,
    # QZSS (J)
    ("J", "S1C"): 1575.42,
    ("J", "S1E"): 1575.42,
    ("J", "S1S"): 1575.42,
    ("J", "S1L"): 1575.42,
    ("J", "S1X"): 1575.42,
    ("J", "S1Z"): 1575.42,
    ("J", "S1B"): 1575.42,
    ("J", "S2S"): 1227.60,
    ("J", "S2L"): 1227.60,
    ("J", "S2X"): 1227.60,
    ("J", "S5I"): 1176.45,
    ("J", "S5Q"): 1176.45,
    ("J", "S5X"): 1176.45,
    ("J", "S5D"): 1176.45,
    ("J", "S5P"): 1176.45,
    ("J", "S5Z"): 1176.45,
    ("J", "S6S"): 1278.75,
    ("J", "S6L"): 1278.75,
    ("J", "S6X"): 1278.75,
    ("J", "S6E"): 1278.75,
    ("J", "S6Z"): 1278.75,
    # BeiDou (C)
    ("C", "S2I"): 1561.098,
    ("C", "S2Q"): 1561.098,
    ("C", "S2X"): 1561.098,
    ("C", "S1I"): 1561.098,  # legacy alias of S2I (RINEX v4)
    ("C", "S1Q"): 1561.098,  # legacy alias of S2Q (RINEX v4)
    ("C", "S1D"): 1575.42,
    ("C", "S1P"): 1575.42,
    ("C", "S1X"): 1575.42,
    ("C", "S1S"): 1575.42,
    ("C", "S1L"): 1575.42,
    ("C", "S1Z"): 1575.42,
    ("C", "S5D"): 1176.45,
    ("C", "S5P"): 1176.45,
    ("C", "S5X"): 1176.45,
    ("C", "S7I"): 1207.140,
    ("C", "S7Q"): 1207.140,
    ("C", "S7X"): 1207.140,
    ("C", "S7D"): 1207.140,
    ("C", "S7P"): 1207.140,
    ("C", "S7Z"): 1207.140,
    ("C", "S8D"): 1191.795,
    ("C", "S8P"): 1191.795,
    ("C", "S8X"): 1191.795,
    ("C", "S6I"): 1268.52,
    ("C", "S6Q"): 1268.52,
    ("C", "S6X"): 1268.52,
    ("C", "S6D"): 1268.52,
    ("C", "S6P"): 1268.52,
    ("C", "S6Z"): 1268.52,
}

# GLONASS channel-dependent obs types: obs_type -> (base MHz, MHz per channel step)
_GLONASS_CHANNEL_OBS: dict[str, tuple[float, float]] = {
    "S1C": (1602.0, 9.0 / 16.0),  # G1: 1602 + k * 9/16
    "S1P": (1602.0, 9.0 / 16.0),
    "S2C": (1246.0, 7.0 / 16.0),  # G2: 1246 + k * 7/16
    "S2P": (1246.0, 7.0 / 16.0),
}

_GLONASS_CHANNEL_MIN = -7
_GLONASS_CHANNEL_MAX = 12

# GLONASS slot (RINEX satID) -> frequency channel k
_GLONASS_SLOT_CHANNEL: dict[str, int] = {
    "R01": 1,
    "R02": -4,
    "R03": 5,
    "R04": 6,
    "R05": 1,
    "R06": -4,
    "R07": 5,
    "R08": 6,
    "R09": -2,
    "R10": -7,
    "R11": 0,
    "R12": -1,
    "R13": -2,
    "R14": -7,
    "R15": 0,
    "R16": -1,
    "R17": 4,
    "R18": -3,
    "R19": 3,
    "R20": 2,
    "R21": 4,
    "R22": -3,
    "R23": 3,
    "R24": 2,
    "R27": -5,
    "R28": 7,
}


@dataclass(frozen=True)
class ObservationRfInfo:
    """Carrier frequency and wavelength for a constellation/obsType pair."""

    constellation: str
    obs_type: str
    frequency_mhz: float
    wavelength_m: float
    sat_id: str | None = None
    channel_dependent: bool = False
    channel: int | None = None


def _normalize_constellation(constellation: str) -> str:
    constellation = constellation.strip().upper()
    if not constellation:
        raise ValueError("constellation must be a non-empty RINEX system code")
    return constellation[0]


def _normalize_obs_type(obs_type: str) -> str:
    obs_type = obs_type.strip().upper()
    if not obs_type:
        raise ValueError("obs_type must be a non-empty RINEX observation code")
    return obs_type


def _mhz_to_wavelength_m(frequency_mhz: float) -> float:
    return SPEED_OF_LIGHT_M_S / (frequency_mhz * 1_000_000.0)


def _normalize_glonass_sat_id(sat_id: str) -> str:
    sat_id = sat_id.strip().upper()
    if not sat_id.startswith("R") or len(sat_id) < 2:
        raise ValueError(f"GLONASS sat_id must start with 'R', got {sat_id!r}")

    slot = sat_id[1:]
    if not slot.isdigit():
        raise ValueError(f"Invalid GLONASS sat_id format: {sat_id!r}")

    return f"R{int(slot):02d}"


def get_glonass_channel(sat_id: str) -> int:
    """Return the GLONASS frequency channel k for a RINEX satID such as R10."""
    normalized_sat_id = _normalize_glonass_sat_id(sat_id)

    if normalized_sat_id not in _GLONASS_SLOT_CHANNEL:
        known_slots = ", ".join(sorted(_GLONASS_SLOT_CHANNEL))
        raise KeyError(
            f"No GLONASS channel mapping for sat_id={normalized_sat_id!r}. Known slots: {known_slots}"
        )

    return _GLONASS_SLOT_CHANNEL[normalized_sat_id]


def _resolve_glonass_channel_frequency(
    obs_type: str,
    channel: int,
) -> tuple[float, int]:
    base_mhz, step_mhz = _GLONASS_CHANNEL_OBS[obs_type]

    if not _GLONASS_CHANNEL_MIN <= channel <= _GLONASS_CHANNEL_MAX:
        raise ValueError(
            f"GLONASS channel must be in [{_GLONASS_CHANNEL_MIN}, {_GLONASS_CHANNEL_MAX}], got {channel}"
        )

    return base_mhz + channel * step_mhz, channel


def get_obs_frequency_mhz(
    constellation: str,
    obs_type: str,
    sat_id: str | None = None,
) -> float:
    """
    Return the carrier frequency in MHz for a constellation and obsType.

    For GLONASS channel-dependent signals (S1C/S1P, S2C/S2P), pass the RINEX
    satID (e.g. R10) so the frequency channel is resolved automatically.
    """
    return get_obs_rf_info(constellation, obs_type, sat_id=sat_id).frequency_mhz


def get_obs_wavelength_m(
    constellation: str,
    obs_type: str,
    sat_id: str | None = None,
) -> float:
    """Return the carrier wavelength in meters for a constellation and obsType."""
    return get_obs_rf_info(constellation, obs_type, sat_id=sat_id).wavelength_m


def get_obs_rf_info(
    constellation: str,
    obs_type: str,
    sat_id: str | None = None,
) -> ObservationRfInfo:
    """
    Return carrier frequency and wavelength for any supported constellation/obsType.

    Parameters
    ----------
    constellation:
        RINEX system code (G, R, E, J, C).
    obs_type:
        RINEX observation type code (e.g. S2I, S1C).
    sat_id:
        RINEX satellite ID (e.g. R10). Required for GLONASS channel-dependent
        obs types (S1C/S1P, S2C/S2P). Ignored for other constellations and obs types.
    """
    constellation = _normalize_constellation(constellation)
    obs_type = _normalize_obs_type(obs_type)
    normalized_sat_id = sat_id.strip().upper() if sat_id else None

    channel_dependent = False
    resolved_channel: int | None = None

    if constellation == "R" and obs_type in _GLONASS_CHANNEL_OBS:
        if not normalized_sat_id:
            raise ValueError(
                f"sat_id is required for GLONASS channel-dependent obs_type={obs_type!r}"
            )

        resolved_channel = get_glonass_channel(normalized_sat_id)
        frequency_mhz, resolved_channel = _resolve_glonass_channel_frequency(
            obs_type,
            resolved_channel,
        )
        channel_dependent = True
    else:
        key = (constellation, obs_type)
        if key not in _OBS_FREQUENCY_MHZ:
            raise KeyError(
                f"No RINEX v4 frequency mapping for constellation={constellation!r}, obs_type={obs_type!r}"
            )
        frequency_mhz = _OBS_FREQUENCY_MHZ[key]

    return ObservationRfInfo(
        constellation=constellation,
        obs_type=obs_type,
        frequency_mhz=frequency_mhz,
        wavelength_m=_mhz_to_wavelength_m(frequency_mhz),
        sat_id=normalized_sat_id,
        channel_dependent=channel_dependent,
        channel=resolved_channel,
    )
