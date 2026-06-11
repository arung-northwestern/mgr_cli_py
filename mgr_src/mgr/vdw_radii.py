"""Static VDW radii tables for distance-grid calculations (picometers).

Tables are vendored from the mendeleev package data (no runtime mendeleev import):

* CRC_MENDELEEV_VDW_RADII_PM — vdw_radius (CRC Handbook via mendeleev; default)
* BONDI_VDW_RADII_PM — vdw_radius_bondi (Bondi 1964; ASE default where defined)
* ALVAREZ_VDW_RADII_PM — vdw_radius_alvarez (Alvarez 2013 + noble-gas update)

See https://mendeleev.readthedocs.io/en/stable/data.html
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VDW_RADII_SETS = ('crc', 'bondi', 'alvarez')


# ---------------------------------------------------------------------------
CRC_MENDELEEV_VDW_RADII_PM = pd.Series(
    {
        "H": 110.00000000000001,
        "He": 140.0,
        "Li": 182.0,
        "Be": 153.0,
        "B": 192.0,
        "C": 170.0,
        "N": 155.0,
        "O": 152.0,
        "F": 147.0,
        "Ne": 154.0,
        "Na": 227.0,
        "Mg": 173.0,
        "Al": 184.0,
        "Si": 210.0,
        "P": 180.0,
        "S": 180.0,
        "Cl": 175.0,
        "Ar": 188.0,
        "K": 275.0,
        "Ca": 231.0,
        "Sc": 215.0,
        "Ti": 211.0,
        "V": 206.99999999999997,
        "Cr": 206.0,
        "Mn": 204.99999999999997,
        "Fe": 204.0,
        "Co": 200.0,
        "Ni": 197.0,
        "Cu": 196.0,
        "Zn": 200.99999999999997,
        "Ga": 187.0,
        "Ge": 211.0,
        "As": 185.0,
        "Se": 190.0,
        "Br": 185.0,
        "Kr": 202.0,
        "Rb": 303.0,
        "Sr": 249.00000000000003,
        "Y": 231.99999999999997,
        "Zr": 223.0,
        "Nb": 218.00000000000003,
        "Mo": 217.0,
        "Tc": 216.0,
        "Ru": 213.0,
        "Rh": 210.0,
        "Pd": 210.0,
        "Ag": 211.0,
        "Cd": 218.00000000000003,
        "In": 193.0,
        "Sn": 217.0,
        "Sb": 206.0,
        "Te": 206.0,
        "I": 198.0,
        "Xe": 216.0,
        "Cs": 343.0,
        "Ba": 268.0,
        "La": 243.00000000000003,
        "Ce": 242.0,
        "Pr": 240.0,
        "Nd": 239.0,
        "Pm": 238.0,
        "Sm": 236.0,
        "Eu": 235.0,
        "Gd": 234.0,
        "Tb": 233.0,
        "Dy": 231.0,
        "Ho": 229.99999999999997,
        "Er": 229.0,
        "Tm": 227.0,
        "Yb": 225.99999999999997,
        "Lu": 224.00000000000003,
        "Hf": 223.0,
        "Ta": 222.00000000000003,
        "W": 218.00000000000003,
        "Re": 216.0,
        "Os": 216.0,
        "Ir": 213.0,
        "Pt": 213.0,
        "Au": 214.0,
        "Hg": 223.0,
        "Tl": 196.0,
        "Pb": 202.0,
        "Bi": 206.99999999999997,
        "Po": 197.0,
        "At": 202.0,
        "Rn": 220.00000000000003,
        "Fr": 348.0,
        "Ra": 283.0,
        "Ac": 247.00000000000003,
        "Th": 245.00000000000003,
        "Pa": 243.00000000000003,
        "U": 241.0,
        "Np": 239.0,
        "Pu": 243.00000000000003,
        "Am": 244.0,
        "Cm": 245.00000000000003,
        "Bk": 244.0,
        "Cf": 245.00000000000003,
        "Es": 245.00000000000003,
        "Fm": 245.00000000000003,
        "Md": 246.0,
        "No": 246.0,
        "Lr": 246.0,
        "Rf": np.nan,
        "Db": np.nan,
        "Sg": np.nan,
        "Bh": np.nan,
        "Hs": np.nan,
        "Mt": np.nan,
        "Ds": np.nan,
        "Rg": np.nan,
        "Cn": np.nan,
        "Nh": np.nan,
        "Fl": np.nan,
        "Mc": np.nan,
        "Lv": np.nan,
        "Ts": np.nan,
        "Og": np.nan,
    }
)

# ---------------------------------------------------------------------------
BONDI_VDW_RADII_PM = pd.Series(
    {
        "H": 120.0,
        "He": 140.0,
        "Li": 181.0,
        "Be": np.nan,
        "B": np.nan,
        "C": 170.0,
        "N": 155.0,
        "O": 152.0,
        "F": 147.0,
        "Ne": 154.0,
        "Na": 227.0,
        "Mg": 173.0,
        "Al": np.nan,
        "Si": 210.0,
        "P": 180.0,
        "S": 180.0,
        "Cl": 175.0,
        "Ar": 188.0,
        "K": 275.0,
        "Ca": np.nan,
        "Sc": np.nan,
        "Ti": np.nan,
        "V": np.nan,
        "Cr": np.nan,
        "Mn": np.nan,
        "Fe": np.nan,
        "Co": np.nan,
        "Ni": np.nan,
        "Cu": np.nan,
        "Zn": np.nan,
        "Ga": 187.0,
        "Ge": np.nan,
        "As": 185.0,
        "Se": 190.0,
        "Br": 183.0,
        "Kr": 202.0,
        "Rb": np.nan,
        "Sr": np.nan,
        "Y": np.nan,
        "Zr": np.nan,
        "Nb": np.nan,
        "Mo": np.nan,
        "Tc": np.nan,
        "Ru": np.nan,
        "Rh": np.nan,
        "Pd": np.nan,
        "Ag": np.nan,
        "Cd": np.nan,
        "In": 193.0,
        "Sn": 217.0,
        "Sb": np.nan,
        "Te": 206.0,
        "I": 198.0,
        "Xe": 216.0,
        "Cs": np.nan,
        "Ba": np.nan,
        "La": np.nan,
        "Ce": np.nan,
        "Pr": np.nan,
        "Nd": np.nan,
        "Pm": np.nan,
        "Sm": np.nan,
        "Eu": np.nan,
        "Gd": np.nan,
        "Tb": np.nan,
        "Dy": np.nan,
        "Ho": np.nan,
        "Er": np.nan,
        "Tm": np.nan,
        "Yb": np.nan,
        "Lu": np.nan,
        "Hf": np.nan,
        "Ta": np.nan,
        "W": np.nan,
        "Re": np.nan,
        "Os": np.nan,
        "Ir": np.nan,
        "Pt": np.nan,
        "Au": np.nan,
        "Hg": np.nan,
        "Tl": 196.0,
        "Pb": 202.0,
        "Bi": np.nan,
        "Po": np.nan,
        "At": np.nan,
        "Rn": np.nan,
        "Fr": np.nan,
        "Ra": np.nan,
        "Ac": np.nan,
        "Th": np.nan,
        "Pa": np.nan,
        "U": np.nan,
        "Np": np.nan,
        "Pu": np.nan,
        "Am": np.nan,
        "Cm": np.nan,
        "Bk": np.nan,
        "Cf": np.nan,
        "Es": np.nan,
        "Fm": np.nan,
        "Md": np.nan,
        "No": np.nan,
        "Lr": np.nan,
        "Rf": np.nan,
        "Db": np.nan,
        "Sg": np.nan,
        "Bh": np.nan,
        "Hs": np.nan,
        "Mt": np.nan,
        "Ds": np.nan,
        "Rg": np.nan,
        "Cn": np.nan,
        "Nh": np.nan,
        "Fl": np.nan,
        "Mc": np.nan,
        "Lv": np.nan,
        "Ts": np.nan,
        "Og": np.nan,
    }
)

# ---------------------------------------------------------------------------
ALVAREZ_VDW_RADII_PM = pd.Series(
    {
        "H": 120.0,
        "He": 143.0,
        "Li": 212.0,
        "Be": 198.0,
        "B": 191.0,
        "C": 177.0,
        "N": 166.0,
        "O": 150.0,
        "F": 146.0,
        "Ne": 158.0,
        "Na": 250.0,
        "Mg": 251.0,
        "Al": 225.0,
        "Si": 219.0,
        "P": 190.0,
        "S": 189.0,
        "Cl": 182.0,
        "Ar": 194.0,
        "K": 273.0,
        "Ca": 262.0,
        "Sc": 258.0,
        "Ti": 246.0,
        "V": 242.0,
        "Cr": 245.0,
        "Mn": 245.0,
        "Fe": 244.0,
        "Co": 240.0,
        "Ni": 240.0,
        "Cu": 238.0,
        "Zn": 239.0,
        "Ga": 232.0,
        "Ge": 229.0,
        "As": 188.0,
        "Se": 182.0,
        "Br": 186.0,
        "Kr": 207.0,
        "Rb": 321.0,
        "Sr": 284.0,
        "Y": 275.0,
        "Zr": 252.0,
        "Nb": 256.0,
        "Mo": 245.0,
        "Tc": 244.0,
        "Ru": 246.0,
        "Rh": 244.0,
        "Pd": 215.0,
        "Ag": 253.0,
        "Cd": 249.0,
        "In": 243.0,
        "Sn": 242.0,
        "Sb": 247.0,
        "Te": 199.0,
        "I": 204.0,
        "Xe": 228.0,
        "Cs": 348.0,
        "Ba": 303.0,
        "La": 298.0,
        "Ce": 288.0,
        "Pr": 292.0,
        "Nd": 295.0,
        "Pm": np.nan,
        "Sm": 290.0,
        "Eu": 287.0,
        "Gd": 283.0,
        "Tb": 279.0,
        "Dy": 287.0,
        "Ho": 281.0,
        "Er": 283.0,
        "Tm": 279.0,
        "Yb": 280.0,
        "Lu": 274.0,
        "Hf": 263.0,
        "Ta": 253.0,
        "W": 257.0,
        "Re": 249.0,
        "Os": 248.0,
        "Ir": 241.0,
        "Pt": 229.0,
        "Au": 232.0,
        "Hg": 245.0,
        "Tl": 247.0,
        "Pb": 260.0,
        "Bi": 254.0,
        "Po": np.nan,
        "At": np.nan,
        "Rn": 240.0,
        "Fr": np.nan,
        "Ra": np.nan,
        "Ac": 280.0,
        "Th": 293.0,
        "Pa": 288.0,
        "U": 271.0,
        "Np": 282.0,
        "Pu": 281.0,
        "Am": 283.0,
        "Cm": 305.0,
        "Bk": 340.0,
        "Cf": 305.0,
        "Es": 270.0,
        "Fm": np.nan,
        "Md": np.nan,
        "No": np.nan,
        "Lr": np.nan,
        "Rf": np.nan,
        "Db": np.nan,
        "Sg": np.nan,
        "Bh": np.nan,
        "Hs": np.nan,
        "Mt": np.nan,
        "Ds": np.nan,
        "Rg": np.nan,
        "Cn": np.nan,
        "Nh": np.nan,
        "Fl": np.nan,
        "Mc": np.nan,
        "Lv": np.nan,
        "Ts": np.nan,
        "Og": np.nan,
    }
)

# Backward-compatible aliases (older code used these names).
MENDELEEV_VDW_RADII_PM = CRC_MENDELEEV_VDW_RADII_PM
CRC_VDW_RADII_PM = CRC_MENDELEEV_VDW_RADII_PM


def _symbols_to_pm(symbols, table_pm: pd.Series, fallback_pm: pd.Series | None = None) -> np.ndarray:
    vals = table_pm.reindex(symbols).to_numpy(dtype=float).copy()
    if fallback_pm is not None:
        fb = fallback_pm.reindex(symbols).to_numpy(dtype=float)
        missing = np.isnan(vals)
        vals[missing] = fb[missing]
    still = np.isnan(vals)
    if still.any():
        from ase.data import vdw_radii as ase_vdw
        from ase.data import atomic_numbers
        for i, sym in enumerate(symbols):
            if np.isnan(vals[i]):
                z = atomic_numbers[sym]
                vals[i] = float(ase_vdw[z]) * 100.0
    return vals


def vdw_radii_angstrom(ase_atoms, radii_set: str = 'crc') -> np.ndarray:
    """Per-atom VDW radii in Å for the requested table."""
    symbols = ase_atoms.get_chemical_symbols()
    key = str(radii_set).lower()
    if key == 'mendeleev':
        key = 'crc'
    if key == 'bondi':
        pm = _symbols_to_pm(symbols, BONDI_VDW_RADII_PM, CRC_MENDELEEV_VDW_RADII_PM)
    elif key == 'alvarez':
        pm = _symbols_to_pm(symbols, ALVAREZ_VDW_RADII_PM, CRC_MENDELEEV_VDW_RADII_PM)
    else:
        pm = _symbols_to_pm(symbols, CRC_MENDELEEV_VDW_RADII_PM)
    return pm / 100.0


def default_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    """Default per-atom VDW radii (CRC Handbook table via mendeleev)."""
    return vdw_radii_angstrom(ase_atoms, radii_set='crc')


def mendeleev_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    """Alias for :func:`default_vdw_radii_angstrom`."""
    return default_vdw_radii_angstrom(ase_atoms)


def bondi_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    return vdw_radii_angstrom(ase_atoms, radii_set='bondi')


def alvarez_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    return vdw_radii_angstrom(ase_atoms, radii_set='alvarez')


def vdw_radii_table_pm(radii_set: str) -> pd.Series:
    key = str(radii_set).lower()
    if key == 'mendeleev':
        key = 'crc'
    if key == 'bondi':
        return BONDI_VDW_RADII_PM
    if key == 'alvarez':
        return ALVAREZ_VDW_RADII_PM
    return CRC_MENDELEEV_VDW_RADII_PM
