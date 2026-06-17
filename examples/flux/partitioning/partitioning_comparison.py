"""
======================================================
NEE Partitioning: comparison of all diive methods
======================================================

diive ships faithful Python ports of four reference NEE partitioning routines:

- **Nighttime ONEFlux** (Reichstein et al. 2005),
  :class:`~diive.flux.NighttimePartitioningOneFlux` (``_NT_OF`` columns)
- **Nighttime REddyProc** (Reichstein et al. 2005),
  :class:`~diive.flux.NighttimePartitioningReddyProc` (``_NT_RP`` columns)
- **Daytime REddyProc** (Lasslop et al. 2010),
  :class:`~diive.flux.DaytimePartitioningReddyProc` (``_DT_RP`` columns)
- **Daytime ONEFlux** (Lasslop et al. 2010),
  :class:`~diive.flux.DaytimePartitioningOneFlux` (``_DT_OF`` columns)

The two nighttime methods fit the temperature response of nighttime NEE; the two
daytime methods fit a light-response curve to daytime NEE. They estimate the
same quantities (GPP and RECO) from different parts of the diel cycle and via
different reference implementations, so they do not produce identical numbers.
This example runs all four on the same input and compares them against each
other and against the bundled REddyProc reference columns (``Reco_CUT_REF`` /
``GPP_CUT_REF_f`` for the nighttime methods, ``Reco_DT_CUT_REF`` /
``GPP_DT_CUT_REF`` for the daytime methods).

Best for: choosing a partitioning method, or seeing how much the choice matters.
"""

# %%
# Load data
# ^^^^^^^^^
import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2017].copy()  # single year keeps the example quick

print("Comparing 2 NIGHTTIME methods (*_NT_OF, *_NT_RP) and 2 DAYTIME methods (*_DT_RP, *_DT_OF)")
print(f"Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} records)")

# %%
# Run all three ports on identical inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# All three share the measured/gap-filled NEE and air temperature plus incoming
# shortwave radiation. The REddyProc ports additionally need longitude and the
# UTC offset (solar-time day/night split); the daytime port additionally needs
# VPD (the light-response curve's VPD term).

of = dv.flux.NighttimePartitioningOneFlux(
    nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
    nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=46.815,
).run().results

nt = dv.flux.NighttimePartitioningReddyProc(
    nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
    nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'],
    lat=46.815, lon=9.855, utc_offset=1,
).run().results

dt = dv.flux.DaytimePartitioningReddyProc(
    nee=df['NEE_CUT_REF_orig'], ta=df['Tair_f'], vpd=df['VPD_f'], sw_in=df['Rg_f'],
    lat=46.815, lon=9.855, utc_offset=1,
).run().results

# The ONEFlux daytime method uses ONEFlux's measured-radiation day/night split
# (no latitude) and needs both the measured and gap-filled drivers.
dt_of = dv.flux.DaytimePartitioningOneFlux(
    nee=df['NEE_CUT_REF_orig'], ta=df['Tair_orig'], sw_in=df['Rg_orig'],
    ta_f=df['Tair_f'], sw_in_f=df['Rg_f'], vpd=df['VPD_f'],
).run().results

# Reference columns: nighttime methods share the NT reference, the daytime methods
# share the DT reference.
reco_ref_nt, gpp_ref_nt = df['Reco_CUT_REF'], df['GPP_CUT_REF_f']
reco_ref_dt, gpp_ref_dt = df['Reco_DT_CUT_REF'], df['GPP_DT_CUT_REF']

# %%
# Numerical comparison
# ^^^^^^^^^^^^^^^^^^^^^
# Each port vs its own bundled reference, and the three ports against each other.


def _r(a, b):
    m = a.notna() & b.notna()
    return np.corrcoef(a[m], b[m])[0, 1]


print("\nRECO vs reference:")
print(f"  diive (ONEFLUX nighttime)   vs reference (NT): {_r(of['RECO_NT_OF'], reco_ref_nt):.4f}")
print(f"  diive (REddyProc nighttime) vs reference (NT): {_r(nt['RECO_NT_RP'], reco_ref_nt):.4f}")
print(f"  diive (REddyProc daytime)   vs reference (DT): {_r(dt['RECO_DT_RP'], reco_ref_dt):.4f}")
print(f"  diive (ONEFLUX daytime)     vs reference (DT): {_r(dt_of['RECO_DT_OF'], reco_ref_dt):.4f}")

print("\nGPP vs reference:")
print(f"  diive (ONEFLUX nighttime)   vs reference (NT): {_r(of['GPP_NT_OF'], gpp_ref_nt):.4f}")
print(f"  diive (REddyProc nighttime) vs reference (NT): {_r(nt['GPP_NT_RP'], gpp_ref_nt):.4f}")
print(f"  diive (REddyProc daytime)   vs reference (DT): {_r(dt['GPP_DT_RP'], gpp_ref_dt):.4f}")
print(f"  diive (ONEFLUX daytime)     vs reference (DT): {_r(dt_of['GPP_DT_OF'], gpp_ref_dt):.4f}")

print("\nMethods against each other (RECO / GPP):")
print(f"  ONEFLUX-nt  vs REddyProc-nt: {_r(of['RECO_NT_OF'], nt['RECO_NT_RP']):.4f} / "
      f"{_r(of['GPP_NT_OF'], nt['GPP_NT_RP']):.4f}")
print(f"  REddyProc-nt vs REddyProc-dt: {_r(nt['RECO_NT_RP'], dt['RECO_DT_RP']):.4f} / "
      f"{_r(nt['GPP_NT_RP'], dt['GPP_DT_RP']):.4f}")
print(f"  REddyProc-dt vs ONEFLUX-dt:  {_r(dt['RECO_DT_RP'], dt_of['RECO_DT_OF']):.4f} / "
      f"{_r(dt['GPP_DT_RP'], dt_of['GPP_DT_OF']):.4f}")

print("\nMeans (umol m-2 s-1):")
print(f"  RECO  ONEFLUX-nt {of['RECO_NT_OF'].mean():.3f}   REddyProc-nt {nt['RECO_NT_RP'].mean():.3f}   "
      f"REddyProc-dt {dt['RECO_DT_RP'].mean():.3f}   ONEFLUX-dt {dt_of['RECO_DT_OF'].mean():.3f}")
print(f"  GPP   ONEFLUX-nt {of['GPP_NT_OF'].mean():.3f}   REddyProc-nt {nt['GPP_NT_RP'].mean():.3f}   "
      f"REddyProc-dt {dt['GPP_DT_RP'].mean():.3f}   ONEFLUX-dt {dt_of['GPP_DT_OF'].mean():.3f}")

# %%
# Time-series comparison (daily means + cumulative)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Three panels on a shared x-axis: RECO daily means (top), GPP daily means
# (middle), and the running cumulative of both RECO and GPP (bottom). Each method
# has its own distinct colour (Okabe-Ito colour-blind-safe palette); the diive
# ports are solid, the bundled references dotted. In the cumulative panel colour
# still encodes the method, while RECO is solid and GPP dashed.

# Okabe-Ito colour-blind-safe palette, one colour per method.
COLORS = {
    'of': '#0072B2',      # diive ONEFLUX nighttime - blue
    'nt': '#E69F00',      # diive REddyProc nighttime - orange
    'dt': '#009E73',      # diive REddyProc daytime - bluish green
    'dt_of': '#56B4E9',   # diive ONEFLUX daytime - sky blue
    'ref_nt': '#CC79A7',  # REddyProc reference (NT) - reddish purple
    'ref_dt': '#D55E00',  # REddyProc reference (DT) - vermillion
}


def _daily(s):
    return s.resample('D').mean()


def _cumulative(s):
    return s.fillna(0.0).cumsum()


fig, (ax_reco, ax_gpp, ax_cum) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.06, hspace=0.13)

ax_reco.plot(_daily(of['RECO_NT_OF']), color=COLORS['of'], lw=1.4, label='diive (ONEFLUX nighttime)')
ax_reco.plot(_daily(nt['RECO_NT_RP']), color=COLORS['nt'], lw=1.4, label='diive (REddyProc nighttime)')
ax_reco.plot(_daily(dt['RECO_DT_RP']), color=COLORS['dt'], lw=1.4, label='diive (REddyProc daytime)')
ax_reco.plot(_daily(dt_of['RECO_DT_OF']), color=COLORS['dt_of'], lw=1.4, label='diive (ONEFLUX daytime)')
ax_reco.plot(_daily(reco_ref_nt), color=COLORS['ref_nt'], lw=1.0, ls=':', label='REddyProc reference (NT)')
ax_reco.plot(_daily(reco_ref_dt), color=COLORS['ref_dt'], lw=1.0, ls=':', label='REddyProc reference (DT)')
ax_reco.set_ylabel('RECO (umol m-2 s-1)')
ax_reco.legend(ncol=3, frameon=False, fontsize=8)

ax_gpp.plot(_daily(of['GPP_NT_OF']), color=COLORS['of'], lw=1.4, label='diive (ONEFLUX nighttime)')
ax_gpp.plot(_daily(nt['GPP_NT_RP']), color=COLORS['nt'], lw=1.4, label='diive (REddyProc nighttime)')
ax_gpp.plot(_daily(dt['GPP_DT_RP']), color=COLORS['dt'], lw=1.4, label='diive (REddyProc daytime)')
ax_gpp.plot(_daily(dt_of['GPP_DT_OF']), color=COLORS['dt_of'], lw=1.4, label='diive (ONEFLUX daytime)')
ax_gpp.plot(_daily(gpp_ref_nt), color=COLORS['ref_nt'], lw=1.0, ls=':', label='REddyProc reference (NT)')
ax_gpp.plot(_daily(gpp_ref_dt), color=COLORS['ref_dt'], lw=1.0, ls=':', label='REddyProc reference (DT)')
ax_gpp.set_ylabel('GPP (umol m-2 s-1)')
ax_gpp.legend(ncol=3, frameon=False, fontsize=8)

# Cumulative: colour = method, RECO solid, GPP dashed.
for s, key, ls, label in [
    (of['RECO_NT_OF'], 'of', '-', 'RECO ONEFLUX-nt'),
    (nt['RECO_NT_RP'], 'nt', '-', 'RECO REddyProc-nt'),
    (dt['RECO_DT_RP'], 'dt', '-', 'RECO REddyProc-dt'),
    (dt_of['RECO_DT_OF'], 'dt_of', '-', 'RECO ONEFLUX-dt'),
    (of['GPP_NT_OF'], 'of', '--', 'GPP ONEFLUX-nt'),
    (nt['GPP_NT_RP'], 'nt', '--', 'GPP REddyProc-nt'),
    (dt['GPP_DT_RP'], 'dt', '--', 'GPP REddyProc-dt'),
    (dt_of['GPP_DT_OF'], 'dt_of', '--', 'GPP ONEFLUX-dt'),
]:
    ax_cum.plot(_cumulative(s), color=COLORS[key], ls=ls, lw=1.3, label=label)
ax_cum.set_ylabel('cumulative flux (umol m-2 s-1)')
ax_cum.set_xlabel('2017')
ax_cum.legend(ncol=2, frameon=False, fontsize=8)

for ax in (ax_reco, ax_gpp):
    ax.axhline(0, color='#455A64', lw=0.8)

fig.suptitle('diive NEE partitioning: ONEFLUX nighttime vs REddyProc nighttime vs '
             'REddyProc daytime vs ONEFLUX daytime - daily means and cumulative (CH-DAV 2017)')
plt.show()
