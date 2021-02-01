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
runs3 = 10000000

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

gmd3 = pd.DataFrame([solvi(i) for i in range(2, 3 + 1)], columns=['ki'], index=range(2, 3 + 1))

# %%
gmd3.index.name = 'i'
gmd3.loc[1] = 0.
gmd3.sort_index(inplace=True)

# %%
gmd3

# %%
ksgm3 = np.flip(gmd3.values.T[0])

# %%
ksgm3

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

# %%
ksmc3 = np.array([0.672608, 0.545532, 0])


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
    if ctr % 10000 == 0:
        print(ctr)
    return [rmax + 1, rpick1 + 1, rpick2 + 1, tp1, fp1, fn1, tp2, fp2, fn2]


# %%
def compare3(xsr, ks1, ks2, ctr):
    rmax = np.argmax(xsr)
    xmax = np.max(xsr)
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
    bucket1 = np.argmax(np.sign(xmax - ks1)) + 1
    bucket2 = np.argmax(np.sign(xmax - ks2)) + 1
    if ctr % 100000 == 0:
        print(ctr)
    return [rmax + 1, rpick1 + 1, rpick2 + 1, tp1, fp1, fn1, tp2, fp2, fn2,
           bucket1, bucket2]


# %%
# %%time

results3 = pd.DataFrame([compare3(rng.random(size=3), ksgm3, ksmc3, run) for run in range(runs3)],
                      columns=['R(Max)', 'R(Choice)-GM', 'R(Choice)-MC',
                               'TP-GM', 'FP-GM', 'FN-GM', 'TP-MC', 'FP-MC', 'FN-MC',
                               'Bucket- GM', 'Bucket- MC'])

# %%
results3.mean()

# %%
results3['R(Max)'].value_counts(normalize=True, sort=False)

# %%
results3['R(Choice)-GM'].value_counts(normalize=True, sort=False)

# %%
results3['R(Choice)-MC'].value_counts(normalize=True, sort=False)


# %%
def pwinsGM(k1, k2):
    p1 = 1 / 3 - (k1 ** 3) / 3
    p2 = k1 / 2 - (k1 ** 3) / 6 - (k2 ** 3) / 3
    p3 = (k1 ** 2) / 2 - (k1 ** 3) / 3 + (k2 ** 2) / 2 - (k2 ** 3) / 3
    return [p1, p2, p3]


# %%
def pwinsMC(k1, k2):
    p1 = 1 / 3 - (k1 ** 3) / 3
    p2 = k1 / 2 - (k1 ** 3) / 6 - (k2 ** 3) / 3
    p3 = k1 * k2 - ((k1 ** 2) * k2) / 2 - (k2 ** 3) / 6
    return [p1, p2, p3]


# %%
pwinsGM(ksgm3[0], ksgm3[1])

# %%
pwinsMC(ksgm3[0], ksgm3[1])

# %%
[
(results3['TP-GM'] * results3['R(Max)'] == 1).mean(),
(results3['TP-GM'] * results3['R(Max)'] == 2).mean(),
(results3['TP-GM'] * results3['R(Max)'] == 3).mean()]

# %%
np.sum(pwinsGM(ksgm3[0], ksgm3[1]))

# %%
np.sum(pwinsMC(ksgm3[0], ksgm3[1]))

# %%
results3['TP-GM'].mean()


# %%
def pwin3bucketsGM(k1, k2):
    b3 = (k2 ** 3) / 3
    b2 = (k1 ** 3) / 6 + (k1 * (k2 ** 2)) / 2 - (k2 ** 3) * 2 / 3
    b1 = (k1 ** 2) / 2 - (k1 ** 3) / 2 + (k2 ** 2) / 2 - (k1 * (k2 ** 2)) / 2
    return [b1, b2, b3]


# %%
pwin3bucketsGM(ksgm3[0], ksgm3[1])

# %%
np.sum(pwin3bucketsGM(ksgm3[0], ksgm3[1]))

# %%
[
(results3['TP-GM'] * (results3['R(Max)'] == 3) * (results3['Bucket- GM'] == 1)).mean(),
(results3['TP-GM'] * (results3['R(Max)'] == 3) * (results3['Bucket- GM'] == 2)).mean(),
(results3['TP-GM'] * (results3['R(Max)'] == 3) * (results3['Bucket- GM'] == 3)).mean()]


# %%
def round1MC(k1, k2):
    tp = 1 / 3 - (k1 ** 3) / 3
    fp = 2 / 3 - k1 + (k1 ** 3) / 3
    fn = (k1 ** 3) / 3
    return [tp, fp, fn]


# %%
def round2MC(k1, k2):
    tp = k1 / 2 - (k1 ** 3) / 6 - (k2 ** 3) / 3
    fp = k1 / 2 - (k1 ** 3) / 6 - (k1 * k2) + ((k1 ** 2) * k2) / 2 + (k2 ** 3) / 6
    fn = (k2 ** 3) / 3
    return [tp, fp, fn]


# %%
round1MC(ksgm3[0], ksgm3[1])

# %%
[
(results3['TP-GM'] * results3['R(Choice)-GM'] == 1).mean(),
(results3['FP-GM'] * results3['R(Choice)-GM'] == 1).mean(),
(results3['FN-GM'] * results3['R(Max)'] == 1).mean()]

# %%
round2MC(ksgm3[0], ksgm3[1])

# %%
[
(results3['TP-GM'] * results3['R(Choice)-GM'] == 2).mean(),
(results3['FP-GM'] * results3['R(Choice)-GM'] == 2).mean(),
(results3['FN-GM'] * results3['R(Max)'] == 2).mean()]

# %%
[
(results3['TP-GM'] * results3['R(Choice)-GM'] == 3).mean(),
(results3['FP-GM'] * results3['R(Choice)-GM'] == 3).mean(),
(results3['FN-GM'] * results3['R(Max)'] == 3).mean()]

# %%
# %%time

results = pd.DataFrame([compare(rng.random(size=n), ksgm, ksmc, run) for run in range(runs)],
                      columns=['R(Max)', 'R(Choice)-GM', 'R(Choice)-MC',
                               'TP-GM', 'FP-GM', 'FN-GM', 'TP-MC', 'FP-MC', 'FN-MC'])

# %%
results.mean()

# %%
