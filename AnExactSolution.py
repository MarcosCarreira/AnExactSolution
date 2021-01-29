# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulation

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Generate random uniform numbers

# %%
n = 1000000

# %%
runs = 100000

# %%
from numpy.random import default_rng
rng = default_rng()

# %% [markdown]
# ## Generate GM indifference numbers

# %%
from sympy.solvers import solve
from sympy import Symbol, binomial, summation, N

# %%
x = Symbol('x', positive=True)
j = Symbol('j')


# %%
def eqin(i):
    return summation((binomial(i, j) * (x ** (i - j)) * ((1 - x) ** j)) / j, (j, 1, i))


# %%
def solvi(i):
    sol = solve([x**(i - 1) -  eqin(i - 1), x < 1], x)
    return N(sol.rhs)


# %%
# %%time

gmd100 = pd.DataFrame([solvi(i) for i in range(2, 100 + 1)], columns=['ki'], index=range(2, 100 + 1))

# %%
gmd100.index.name = 'i'
gmd100.loc[1] = 0.
gmd100.sort_index(inplace=True)

# %%
gmd100


# %%
def solv2ndordi(i):
    return 1 / (1 + 0.80435226286 / (i - 1) + 0.183199 / (i - 1) ** 2)


# %%
# %%time

gmdn = pd.DataFrame([solv2ndordi(i) for i in range(2, n + 1)], columns=['ki'], index=range(2, n + 1))

# %%
gmdn.index.name = 'i'
gmdn.loc[1] = 0.
gmdn.sort_index(inplace=True)

# %%
gmdn.loc[:100]

# %%
ksgm = np.flip(gmdn.values.T[0])

# %% [markdown]
# ## Generate MC cutoff numbers

# %%
ksmc = np.array([((1 - 1 / n) + np.log((n - r) / n) / n) for r in range(1, n)] + [0])


# %% [markdown]
# ## Run both choices

# %%
def compare(xsr, ks1, ks2, ctr):
    rmax = np.argmax(xsr)
    diff1 = np.sign(xsr - ks1)
    rpick1 = np.argmax(diff1)
    tp1, fp1, fn1 = [0, 0, 0]
    diff2 = np.sign(xsr - ks2)
    rpick2 = np.argmax(diff2)
    tp2, fp2, fn2 = [0, 0, 0]
    if rpick1 == rmax:
        tp1 = 1   
    else:
        if rpick1 < rmax:
            fp1 = 1
        else:
            fn1 = 1
    if rpick2 == rmax:
        tp2 = 1   
    else:
        if rpick2 < rmax:
            fp2 = 1
        else:
            fn2 = 1
    if ctr % 1000 == 0:
        print(ctr)
    return [rmax + 1, rpick1 + 1, rpick2 + 1, tp1, fp1, fn1, tp2, fp2, fn2]


# %%
# %%time

results = pd.DataFrame([compare(rng.random(size=n), ksgm, ksmc, run) for run in range(runs)],
                      columns=['R(Max)', 'R(Choice)-GM', 'R(Choice)-MC',
                               'TP-GM', 'FP-GM', 'FN-GM', 'TP-MC', 'FP-MC', 'FN-MC'])

# %%
results.mean()

# %%
