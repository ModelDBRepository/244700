



from inputs import *

import matplotlib.pyplot as plt

import time

import os


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams["font.size"] = "13"




def bin_spikes(spikes, tmax, dt):

    bins = np.arange(0, tmax + dt, dt)
    idx = np.searchsorted(spikes, bins)
    return np.diff(idx).astype(float)


def xcorr(x,y,s_range, dt):

    x_mean = np.sum(x)/len(x)
    y_mean = np.sum(y)/len(x)

    print(x_mean)

    t0 = time.time()

    ccf = [0]*2*s_range

    for s in range(-s_range, s_range):

        ccf[s + s_range] = np.sum(x * np.roll(y, s)) / len(x)

    ccf = np.array(ccf)


    print('time ' + str(time.time() - t0))


    ccvf =  ccf - x_mean * y_mean

    lm = np.sum(ccvf) * dt / x_mean

    return ccvf, lm


def xcorr0(x,y,dt):

    x_mean = np.sum(x)/len(x)
    y_mean = np.sum(y)/len(x)

    ccf0 = np.sum(x * y) / len(x)

    ccvf0 = (ccf0 - x_mean * y_mean) * dt / x_mean


    return ccvf0







n_cmp = 2
ns_syn = 2

rate = 20. # Hz
t_sim = 500.

corr_glob = 1.
corr_loc = 1.
jtr = 0.

dt = 0.0005

t_ref = 0.002

nt = int(t_sim/dt)

s_range = int(0.0001 * nt)

s_range = int(((t_ref/dt)))

tot_corrA = []

tot_corr_B = []

jtr_list = np.arange(0., 0.006, 0.001)
corr_list = np.arange(0., 1.1, 0.1)


mode = ''


for jtr in jtr_list:

    tot_corr_C = []

    for corr in corr_list:

        t0 = time.time()

        spt = spike_trains_hierarch_ind_global(n_cmp, ns_syn, rate, t_sim, 1., corr, jtr)

        print('time generation: ' + str(time.time() - t0))

        margin_cut = 1

        # division by dt is for scaling

        binned_1 = bin_spikes(spt[1][0][margin_cut:-margin_cut], t_sim, dt)/dt
        binned_2 = bin_spikes(spt[0][1][margin_cut:-margin_cut], t_sim, dt)/dt

        t0 = time.time()

        xcorrT, lm = xcorr(binned_1, binned_2, s_range, dt)

        print('time cross correlation: ' + str(time.time() - t0))

        print('lambda ' + str(lm))

        tot_corrA.append([corr,jtr,lm])

        tot_corr_C.append(lm)

        if mode == 'detailed':

            plt.figure()
            plt.plot(binned_1)
            plt.plot(binned_2)
            plt.show()

            plt.figure()
            plt.plot(xcorrT)
            plt.show()

    tot_corr_B.append(tot_corr_C)







# for jtr in jtr_list:
#
#     tot_corr_C = []
#
#     for corr in corr_list:
#
#         t0 = time.time()
#
#         spt = spike_trains_hierarch(n_cmp, ns_syn, rate, t_sim, corr, corr, jtr)
#
#         print('time generation: ' + str(time.time() - t0))
#
#         margin_cut = 1
#
#         # division by dt is for scaling
#
#         binned_1 = bin_spikes(spt[0][0][margin_cut:-margin_cut], t_sim, dt)/dt
#         binned_2 = bin_spikes(spt[0][1][margin_cut:-margin_cut], t_sim, dt)/dt
#
#         t0 = time.time()
#
#         xcorr0T = xcorr0(binned_1, binned_2, dt)
#
#         tot_corrA.append([corr,jtr,xcorr0T])
#
#         tot_corr_C.append(xcorr0T)
#
#
#     tot_corr_B.append(tot_corr_C)

date_time = time.strftime('%y-%m-%d--%H-%M-%S', time.localtime())

if not os.path.exists('../DATA/' + str(date_time) + '_corr_measure'):
	os.makedirs('../DATA/' + str(date_time) + '_corr_measure')



main_path = os.getcwd()
os.chdir('../DATA/' + str(date_time) + '_corr_measure')


tot_corrA = np.array(tot_corrA)

print(tot_corrA)


plt.plot(tot_corrA[:,0],tot_corrA[:,2])
plt.show()


plt.figure()

contf = plt.contourf(corr_list, jtr_list, tot_corr_B, cmap='YlOrRd', alpha=1., levels=np.arange(0., 1.05, 0.05))

cbar = plt.colorbar(contf, orientation = 'vertical', pad = 0.05, shrink = 0.9)


cbar.ax.get_yaxis().labelpad = 25
cbar.ax.tick_params(labelsize='14' )
cbar.ax.set_ylabel('Correlation', fontsize='18', rotation=270)

plt.xlabel('Ratio of shared spikes')
plt.ylabel('Jitter [ms]')

plt.savefig('figure_ref_doubled.png', dpi = 300)
plt.savefig('figure_ref_doubledsv.svg', dpi = 300)

plt.show()

os.chdir(main_path)

