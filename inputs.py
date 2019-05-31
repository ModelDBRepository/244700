# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:28:09 2016

@author: tomasz, Bartosz, Romain
"""

from __future__ import print_function

import numpy as np
import numba

from scipy import stats
import math
import pylab as plt

#from PoissonPointProcess import SNCP_corr, corr_poisson, CCVF

def generate_poisson_old(freq, tmax):
     """Draw spike times from Poisson distribution"""
     t = np.cumsum(stats.expon.rvs(scale=1/freq, size=round(freq * tmax)))
     t = t[:t.searchsorted(tmax)]
     return t


# this function might be slower than the previous implementation - please check
def generate_poisson(freq, tmax):
    return np.sort(np.random.uniform(0,tmax,np.random.poisson(lam = freq*tmax)))            

def generate_correlated(mother, r):
    """Generate correlated spike train from a mother spike train
    process"""
    rn = np.random.rand(len(mother))
    return mother[rn<r]


def jitter_spikes(spt, tau):
    """add normaly distributed jitter to spike trains"""
    delta_t = stats.expon.rvs(scale=tau, size=len(spt))
    # to have symmetric distribution around 0 uncomment the following lines
    # sign = (np.random.rand(len(spt)) > 0.5) * 2 - 1
    # delta_t = sign * delta_t
    return np.sort(spt + delta_t)


def spike_trains_hierarch(nb_comp, nb_syn, rate, tmax, corr_coef_L, corr_coef_G, jtr):

    N = rate * tmax

    if corr_coef_L == 0. and corr_coef_G == 0.:

        spike_trains = [[np.sort(generate_poisson(rate, tmax)) for i in range(nb_syn)] for j in range(nb_comp)]

        return spike_trains


    if corr_coef_G == 0.:

        r_LC = corr_coef_L
        M = round(N / r_LC)
        freq_mother = float(M) / tmax
        mother_loc = [np.sort(generate_poisson(freq_mother, tmax)) for j in range(nb_comp)]
        spike_trains = [[jitter_spikes(generate_correlated(mother_loc[j], r_LC), jtr) for i in range(nb_syn)] for j
                        in range(nb_comp)]

        return spike_trains


    else:

        if corr_coef_L < 0. or corr_coef_L > 1.:

            raise ValueError('\n\n *** Local correlation outside [0,1]! ***')


        if corr_coef_G < 0. or corr_coef_G > 1.:

            raise ValueError('\n\n *** Global correlation outside [0,1]! ***')


        if corr_coef_L < corr_coef_G:

            raise ValueError('\n\n *** Global correlation cannot be lower than local correlation! ***')


        if corr_coef_G == 0.:

            "print generation"

            r_LC = corr_coef_L
            M = round(N / r_LC)
            freq_mother = float(M) / tmax
            mother_loc = [np.sort(generate_poisson(freq_mother, tmax)) for j in range(nb_comp)]
            spike_trains = [[jitter_spikes(generate_correlated(mother_loc[j], r_LC), jtr) for i in range(nb_syn)] for j
                            in range(nb_comp)]

            return spike_trains

        r_LC = corr_coef_L
        r_GL = corr_coef_G / corr_coef_L

        M = round(N / (r_GL * r_LC))
        freq_mother = float(M) / tmax
        mother_glob = generate_poisson(freq_mother, tmax)
        mother_loc = [generate_correlated(mother_glob, r_GL) for j in range(nb_comp)]
        spike_trains = [[jitter_spikes(generate_correlated(mother_loc[j], r_LC), jtr) for i in range(nb_syn)] for j in range(nb_comp)]

        return spike_trains




# def spike_trains_hierarch_ind_global_bad(nc, ns, rate, tmax, rL, rG, jtr):
#
#     N = rate * tmax
#
#     M = round(N * nc * ns / (1 + rG * (nc - 1) + rL * (ns - 1)))
#
#     fM = M / tmax
#
#     ts = generate_poisson(fM, tmax)
#
#     C = np.random.randint(nc, size=len(ts))
#
#     S = np.random.randint(ns, size=len(ts))
#
#     ts_loc = [[[ts[i]], [[C[i], S[i]]]] for i in range(len(ts))]
#
#
#     for i in range(len(ts_loc)):
#
#         for s in range(ns):
#
#             if s != ts_loc[i][1][0][1]:
#
#                 r = np.random.rand()
#
#                 if r < rL:
#
#                     ts_loc[i][1] = ts_loc[i][1] + [[ts_loc[i][1][0][0],s]]
#
#         for c in range(nc):
#
#             if c != ts_loc[i][1][0][0]:
#
#                 r = np.random.rand()
#
#                 if r < rG:
#
#                     ts_loc[i][1] = ts_loc[i][1] + [[c, np.random.randint(ns)]]
#
#
#     ts_loc_a = [[[] for i in range(ns)] for j in range(nc)]
#
#     total_number = 0
#
#     for i in range(len(ts_loc)):
#
#         for j in range(len(ts_loc[i][1])):
#             ts_loc_a[ts_loc[i][1][j][0]][ts_loc[i][1][j][1]] = ts_loc_a[ts_loc[i][1][j][0]][ts_loc[i][1][j][1]] + ts_loc[i][0]
#
#             total_number = total_number + len(ts_loc[i][0])
#
#     print('Number of spikes: ' + str(total_number))
#     print('Number of spikes per synapse: ' + str(total_number/(nc * ns)))
#     print(N)
#
#     return ts_loc_a




def connect_matrix(t_length, nc, ns, rL, rG):

    from random import shuffle

    C = np.random.randint(nc, size=t_length)

    mat0 = np.zeros((t_length, nc * ns))

    glob_array = [rG] + [0] * (ns - 1)

    loc_array = [1.] + [rL] * (ns - 1)


    for i in range(t_length):

        shuffle(glob_array)
        shuffle(loc_array)

        mat0[i, :] = glob_array * nc

        mat0[i, (C[i])*ns : (C[i]+1) * ns] = loc_array

    return mat0



def generate_correlated_new(mother, vec):

    rn = np.random.rand(len(mother))
    return mother[rn<vec]


def spike_trains_hierarch_ind_global(nc, ns, rate, tmax, rL, rG, jtr):

    N = rate * tmax

    M = round(N * nc * ns / (1 + rG * (nc - 1) + rL * (ns - 1)))

    fM = M / tmax

    ts = generate_poisson(fM, tmax)

    cm = connect_matrix(len(ts), nc, ns, rL, rG)

    spike_train = [[jitter_spikes(generate_correlated_new(ts, cm[:,j*ns + i]), jtr) for i in range(ns)] for j in range(nc)]

    return spike_train




# import time
#
# nc = 3
# ns = 2
#
# tmax = 1
#
# rate = 1
#
# rL = 1.
#
# rG = 0.5
#
# jtr = 0.1
#
#
# t0 = time.time()
#
# sp1 = spike_trains_hierarch_ind_global(nc, ns, rate, tmax, rL, rG, jtr)
#
# print('new new method: ' + str(time.time() - t0))
#
# print(len(sp1[0][0]))
#
#
#
#
# import matplotlib.pyplot as plt
#
# plt.figure()
#
# for i in range(len(sp1)):
#
#     for j in range(len(sp1[0])):
#
#         plt.plot([i, sp1[i][j][0]],'+')
#
# plt.show()
#
#
#
#
#
# t0 = time.time()
#
# sp2 = spike_trains_hierarch_ind_global_bad(nc, ns, rate, tmax, rL, rG, jtr)
#
# print('new method: ' + str(time.time() - t0))
#
# print(len(sp2[0][0]))
#
#
#
# t0 = time.time()
#
# sp3 = spike_trains_hierarch(nc, ns, rate, tmax, rL, rG, jtr)
#
# print('old method: ' + str(time.time() - t0))
#
# print(len(sp3[0][0]))

