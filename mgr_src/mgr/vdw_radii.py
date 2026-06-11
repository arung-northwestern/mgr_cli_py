"""Default VDW radii for distance-grid calculations.

Two tables are provided:

* ``CRC_VDW_RADII_PM`` — Bondi (1964) / CRC Handbook values.  These are
  the default for the Rust distance-grid backends and are the values used
  by tools such as zeo++ and PoreBlazer for pore-size analysis.
* ``MENDELEEV_VDW_RADII_PM`` — Mendeleev database values (larger for many
  transition metals).  Retained for backwards compatibility.

References
----------
Bondi, A. "van der Waals Volumes and Radii."
J. Phys. Chem. 1964, 68 (3), 441–451.

Mantina, M.; Chamberlin, A. C.; Valero, R.; Cramer, C. J.; Truhlar, D. G.
"Consistent van der Waals Radii for the Whole Main Group."
J. Phys. Chem. A 2009, 113 (19), 5806–5812.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CRC / Bondi (1964) VDW radii — picometers
# ---------------------------------------------------------------------------
CRC_VDW_RADII_PM = pd.Series(
    {
        "H": 120.0,
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
        "Sc": 211.0,
        "Ti": 200.0,
        "V": 190.0,
        "Cr": 188.0,
        "Mn": 187.0,
        "Fe": 186.0,
        "Co": 185.0,
        "Ni": 184.0,
        "Cu": 170.0,
        "Zn": 173.0,
        "Ga": 184.0,
        "Ge": 211.0,
        "As": 185.0,
        "Se": 190.0,
        "Br": 185.0,
        "Kr": 202.0,
        "Rb": 303.0,
        "Sr": 249.0,
        "Y": 212.0,
        "Zr": 212.0,
        "Nb": 198.0,
        "Mo": 190.0,
        "Tc": 183.0,
        "Ru": 178.0,
        "Rh": 173.0,
        "Pd": 169.0,
        "Ag": 172.0,
        "Cd": 158.0,
        "In": 193.0,
        "Sn": 210.0,
        "Sb": 206.0,
        "Te": 207.0,
        "I": 198.0,
        "Xe": 216.0,
        "Cs": 343.0,
        "Ba": 268.0,
        "La": 250.0,
        "Ce": 255.0,
        "Pr": 255.0,
        "Nd": 255.0,
        "Pm": 255.0,
        "Sm": 255.0,
        "Eu": 255.0,
        "Gd": 255.0,
        "Tb": 255.0,
        "Dy": 255.0,
        "Ho": 255.0,
        "Er": 255.0,
        "Tm": 255.0,
        "Yb": 255.0,
        "Lu": 255.0,
        "Hf": 212.0,
        "Ta": 200.0,
        "W": 190.0,
        "Re": 188.0,
        "Os": 186.0,
        "Ir": 185.0,
        "Pt": 180.0,
        "Au": 166.0,
        "Hg": 155.0,
        "Tl": 196.0,
        "Pb": 202.0,
        "Bi": 207.0,
        "Po": 197.0,
        "At": 202.0,
        "Rn": 220.0,
        "Fr": 348.0,
        "Ra": 283.0,
        "Ac": 247.0,
        "Th": 245.0,
        "Pa": 243.0,
        "U": 241.0,
        "Np": 239.0,
        "Pu": 243.0,
        "Am": 244.0,
        "Cm": 245.0,
        "Bk": 244.0,
        "Cf": 245.0,
        "Es": 245.0,
        "Fm": 245.0,
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
# Mendeleev VDW radii — picometers (legacy, larger for transition metals)
# ---------------------------------------------------------------------------
MENDELEEV_VDW_RADII_PM = pd.Series(
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


def default_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    """Per-atom VDW radii in Å for an ASE Atoms object.

    Uses the CRC / Bondi (1964) radii by default.  For the legacy
    Mendeleev table, call :func:`mendeleev_vdw_radii_angstrom` instead.
    """
    return (
        CRC_VDW_RADII_PM[ase_atoms.get_chemical_symbols()].values / 100.0
    )


def mendeleev_vdw_radii_angstrom(ase_atoms) -> np.ndarray:
    """Per-atom Mendeleev VDW radii in Å (legacy table, larger for metals)."""
    return (
        MENDELEEV_VDW_RADII_PM[ase_atoms.get_chemical_symbols()].values / 100.0
    )
