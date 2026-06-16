"""
====================================
Nighttime NEE Partitioning (RECO/GPP)
====================================

Partition net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO) with the nighttime method of Reichstein et al.
(2005), a faithful Python port of the ONEFlux reference implementation.

The nighttime method exploits the fact that there is no photosynthesis at
night, so nighttime NEE is pure respiration. A temperature-response model
(Lloyd & Taylor 1994) is fitted to nighttime NEE and extrapolated to daytime
temperatures to recover daytime respiration; GPP is then the difference
``GPP = RECO - NEE``.

Best for: deriving GPP and RECO from gap-filled, USTAR-filtered NEE.
"""

# %%
# Why partition NEE?
# ^^^^^^^^^^^^^^^^^^
# Eddy covariance measures the *net* CO2 flux (NEE), which bundles together two
# opposing gross fluxes: uptake by photosynthesis (GPP) and release by
# respiration (RECO). Ecologists usually want the two components separately.
# The nighttime method estimates RECO from the night and reconstructs GPP.

# %%
# Load data
# ^^^^^^^^^
# The bundled CH-DAV dataset already contains FLUXNET-produced RECO/GPP
# (``Reco_CUT_REF`` / ``GPP_CUT_REF_f``), which lets us sanity-check the result.

import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2018].copy()  # single year keeps the example quick

print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run the partitioning
# ^^^^^^^^^^^^^^^^^^^^^
# Inputs:
#   - measured NEE and the gap-filled NEE (for the GPP residual)
#   - measured air temperature and gap-filled air temperature (for RECO)
#   - incoming shortwave radiation (for the day/night split)
#   - site latitude

part = dv.flux.NighttimePartitioning(
    nee=df['NEE_CUT_REF_orig'],   # measured NEE (NaN where not measured)
    ta=df['Tair_orig'],           # measured air temperature
    sw_in=df['Rg_orig'],          # incoming shortwave radiation
    nee_f=df['NEE_CUT_REF_f'],    # gap-filled NEE
    ta_f=df['Tair_f'],            # gap-filled air temperature
    lat=46.815,                   # CH-DAV latitude
)
part.run()
results = part.results

print("\nResult columns:", list(results.columns))
print(results[['RECO_NT', 'GPP_NT', 'E0_NT']].describe())

# %%
# Compare to the FLUXNET reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RECO and GPP should track the FLUXNET-produced columns closely.

reco_ref = df['Reco_CUT_REF']
gpp_ref = df['GPP_CUT_REF_f']

m_reco = results['RECO_NT'].notna() & reco_ref.notna()
m_gpp = results['GPP_NT'].notna() & gpp_ref.notna()

print(f"RECO correlation vs FLUXNET: "
      f"{np.corrcoef(results['RECO_NT'][m_reco], reco_ref[m_reco])[0, 1]:.4f}")
print(f"GPP  correlation vs FLUXNET: "
      f"{np.corrcoef(results['GPP_NT'][m_gpp], gpp_ref[m_gpp])[0, 1]:.4f}")

print(f"\nAnnual sums (umol m-2 s-1, mean):")
print(f"  RECO_NT mean: {results['RECO_NT'].mean():.3f}   "
      f"FLUXNET Reco mean: {reco_ref.mean():.3f}")
print(f"  GPP_NT  mean: {results['GPP_NT'].mean():.3f}   "
      f"FLUXNET GPP  mean: {gpp_ref.mean():.3f}")

# %%
# Outlier-robust variants
# ^^^^^^^^^^^^^^^^^^^^^^^^
# ``RECO_NT_ROB`` / ``GPP_NT_ROB`` use a trimmed Rref estimate that downweights
# the largest nighttime deviations, useful when nighttime NEE is noisy.

print(results[['RECO_NT', 'RECO_NT_ROB', 'GPP_NT', 'GPP_NT_ROB']].mean())
