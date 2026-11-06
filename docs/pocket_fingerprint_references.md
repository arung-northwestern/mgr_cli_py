# Pocket environment fingerprint — references and design

## What we compute

For each pocket: framework atoms in an **adaptive shell** (scaled from dominant PSD diameter d₁), positions relative to the **void centroid** (MIC), then:

1. **Normalized radial histogram** \(g(r / R_{\mathrm{shell}})\) per element  
2. **Spherical-harmonic power spectrum** \( \sum_m |a_{\ell m}|^2 \) per \(\ell\) (rotation-invariant angular content)

Similarity = mean cosine similarity across channels. **Small pockets** (d₁ ≤ 8 Å): angular (SH) channels only, merge if cosine ≥ 0.96.

**Bonding:** covalent bonds are used only for **one-hop bond closure** (expand the atom shell). Bond order / aromatic vs single is **not** part of the fingerprint vector (unlike the WL graph hash).

## Relation to published ideas (citable)

This is a **lightweight custom descriptor**, not a full reimplementation of any single paper. Closest lines of work:

| Idea | Reference (example) | Our use |
|------|---------------------|--------|
| Rotation-invariant atomic environments | A. P. Bartók, R. E. Jones, et al., *Phys. Rev. B* **88**, 054104 (2013) — SOAP | Same goal (compare local geometry); we use SH power on a unit-sphere atom density instead of Gaussian SOAP basis |
| 3D shape / ligand comparison via spherical harmonics | e.g. R. B. Morris et al., *J. Chem. Inf. Comput. Sci.* **36** (1996); similar SH shape signatures in structural biology | Project atom directions onto a sphere around a center; compare via harmonic magnitudes |
| Bond-orientational / angular order parameters | P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* **28**, 784 (1983) | Conceptually related (angular structure around a point); we use full SH spectrum per element, not only \(\ell=6\) BOO |
| MOF pore / cavity descriptors | Various zeolite/MOF pore-size and window analyses (PSD, spherical probes) | We already use probe PSD for d₁; fingerprint adds **local framework angular layout** around the void |

**Plain summary for a methods section:**  
*Local pocket environments were compared using rotation-invariant descriptors: atom positions relative to the pocket centroid were expressed in normalized radial histograms and in the power spectrum of a spherical-harmonic expansion of the angular density on the unit sphere (cf. SOAP-style local environments and SH-based molecular shape comparison). Similarity was quantified by cosine similarity between descriptor vectors.*

## What is “common sense” vs literature

- **Adaptive shell from pore size** — engineering choice (probe radius scales with cavity).  
- **SH power spectrum for rotation invariance** — standard in 3D shape analysis; not novel.  
- **Stricter merge in the small d₁ band** — tuned for ZIF-20–like materials in this repo, not from a paper.
