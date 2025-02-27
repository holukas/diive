"""
https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
"""

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from diive.core.plotting.styles.LightTheme import colorwheel_36
from diive.configs.exampledata import load_exampledata_parquet

df = load_exampledata_parquet()
print(df)
col = 'NEE_CUT_REF_f'
df = df[[col]].copy()
df['YEAR'] = df.index.year
df['MONTH'] = df.index.month
# locs = (df['YEAR'] >= 2015) & (df['YEAR'] <= 2020)
# df = df[locs].copy()
years = [x for x in np.unique(df['YEAR'])]
months = [x for x in np.unique(df['MONTH'])]
# colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000',
#           '#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000']

colors = colorwheel_36()
gs = (grid_spec.GridSpec(len(months), 1))
fig = plt.figure(figsize=(8, 8))

i = 0

# Creating empty list
ax_objs = []

for month in months:
    month = months[i]
    x = np.array(df[df['MONTH'] == month][col])
    x_d = np.linspace(-20, 30, 1000)

    kde = KernelDensity(bandwidth=0.99, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # creating new axes object
    ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))

    # plotting the distribution
    ax_objs[-1].plot(x_d, np.exp(logprob), color="black", lw=1)
    ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color=colors[i])

    # setting uniform x and y lims
    ax_objs[-1].set_xlim(-20, 10)
    ax_objs[-1].set_ylim(0, 0.3)

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])
    ax_objs[-1].set_yticks([])

    if i == len(months) - 1:
        ax_objs[-1].set_xlabel("X", fontsize=16, fontweight="bold")
    else:
        ax_objs[-1].set_xticklabels([])
        ax_objs[-1].set_xticks([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    adj_country = str(month).replace(" ", "\n")
    ax_objs[-1].text(-21, 0.01, adj_country, fontweight="bold", fontsize=14, ha="right")

    i += 1

gs.update(hspace=-0.7)

fig.text(0.07, 0.85, "TITLE", fontsize=20)

plt.tight_layout()
plt.show()
