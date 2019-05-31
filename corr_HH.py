

from brian2 import *
from inputs import *
import os
import time
import numpy


# BrianLogger.suppress_hierarchy('brian2.codegen')
# BrianLogger.suppress_name('resolution_conflict')
# BrianLogger.suppress_hierarchy('brian2')


# passive
Cm = 1e-6 * farad * cm**-2
Ri = 100 * ohm * cm
g_l = 1e-4 * siemens * cm**-2 # Some models use
E_l = -70 * mV


# hh
E_Na = 58 * mV
E_K = -80 * mV
g_Na = 12 * msiemens / cm**2  # I doubled this value / 12 : This is the value from (Destexhe,Pare,1999) Some models use 50 ms/cm2 (pyramidal neurons, Wang 1998) or 35 ms/cm2 (interneurons)
							  # Experiments: 1994 - Stuart/Sakmann > 4 mS/cm2
g_K = 7 * msiemens / cm**2  # I doubled this value /  7 :  In (Destexhe,Pare, 1999) = 10 ms/cm2. Some modles use 10.5 ms/cm2 (pyramidal neuron, Wang 1998)
v_th = - 63 * mV


# synaptic
E_ampa = 0 * mV
E_gaba = -75 * mV

tau_ampa = 5 * ms
tau_gaba = 5 * ms

dg_ampa = 1 * nS
dg_gaba = 1 * nS



### HERE PARAMETERS TO CHANGE ###

morphology = 'point'		# point, ball+stick
dend_shape = 'cone' 		# cylinder, cone

mode = 'simple'	#  simple, detailed
speed = 'fast'			# fast, long

chosen_param = 'wgh'		# rate_per_weight, wgh, ch_dns

ns_ampa_tot = 200
gaba_ratio = 0.2
jtr = 10. * ms #  plots for madex 10 ms; plots for hh 30 ms

ch_dns = 1.

# if scanning over rate
wgh = 0.5 # nS # 0.5


if morphology == 'point':
	rate_range = 17.

if morphology == 'ball+stick':
	rate_range = 5. # program will scan over 10%, 25%, 50%, 75% and 100% of this value


# rate range for 0.5 nS synapse: 7
# rate range for 0.2 nS synapse: 40
# rate range for 1.5 ns synapse: 1.2 (3: too high!!!)

# if scanning over wgh
rate_per_weight = 0.75 # 0.75


### NUMERICAL
method = 'rk2' 			# 'linear', 'exponential_euler', 'rk2 or 'heun'
dt = 0.05 * ms # 0.05 !!!

### MORPHOLOGY

diam_soma = 40 * um    # 40

# length of dendrite
length = 1000 * um
len_comp = 5 * um
n_cmp = int(length / um / (len_comp / um))

# cylinder
diam_cln = 1 * um

# cone
diam_0 = 5
diam_f = 0.5
beta = 0.05 # smaller more steep decrease of diameter 0.1

lengthT = linspace(0, length / um, num=n_cmp + 1)


### SYNAPTIC

if morphology == 'point':
	ns_ampa = ns_ampa_tot
	ns_gaba_soma = int(ceil(ns_ampa * gaba_ratio))
	wgh = 0.105  # nS # 0.11 is good for 0.5 synapse and 10.3 Hz rate

if morphology == 'ball+stick':
	ns_ampa = int(ns_ampa_tot / n_cmp)
	ns_gaba_soma = int(ceil(ns_ampa_tot * gaba_ratio))


### SCANNED PARAMETERS

param_to_scan = {}




# rate
param_to_scan['rate_per_weight'] = np.array([0.1, 0.25, 0.5, 0.75, 1.]) * rate_range
param_to_scan['rate_per_weight'] = param_to_scan['rate_per_weight'].tolist()

# weight

if chosen_param == 'wgh':

	if morphology == 'ball+stick':

		wgh_list = [0.25, 0.5, 0.75, 1.]  # nS
		param_to_scan['wgh'] = wgh_list
		rate_wgh = np.array([4., 2., 1.5, 1.])  # it was 2 then 1.85


	if morphology == 'point':

		wgh_list = [0.05, 0.1, 0.15, 0.2]  # nS
		param_to_scan['wgh'] = wgh_list
		rate_wgh = np.array([23., 11.5, 7.7, 5.75])  # it was 2 then 1.85



if chosen_param == 'ch_dns':

	if morphology == 'ball+stick':
		param_to_scan['ch_dns'] = [0., 0.25, 0.5, 0.75, 1.]
		# parameters used for wgh: 0.5 nS and jtr: 15 ms > 10 Hz for noncorrelated
		# rate_ch_dns = np.array([10.3, 8., 4., 2.3, 1.7]) * 1 # [20, 2] (2.5 for 0.75 was too high) (1.6 for 1 was too low)
		#rate_ch_dns = np.array([7, 2., 1, 0.5, 0.25]) # [20, 2] (2.5 for 0.75 was too high) (1.6 for 1 was too low)
		rate_ch_dns =  np.array([ 11.5,   9. ,   4.8 ,   2.5,   2.]) # it was 2 then 1.85

	### Scaling of synpses

	if morphology == 'point':
		param_to_scan['ch_dns'] = [0.]
		rate_ch_dns = np.array([11.5])  # [20, 2] (2.5 for 0.75 was too high) (1.6 for 1 was too low)


# local correlation
param_to_scan['rL'] = [0.1, 0.5, 1.]

# correlation scan
corr_scan = np.arange(0.,1.1,0.1)


if speed == 'fast':
	t_sim = 2000 * ms
	n_rep = 1

if speed == 'long':
	t_sim = 20000 * ms
	n_rep = 10

if mode == 'detailed':
	t_sim = 1000 * ms

if chosen_param == 'rate_per_weight':
	chosen_param_short = 'rate'

else:
	chosen_param_short = chosen_param


if morphology == 'point':
	length = diam_soma
	n_cmp = 1
	morpho = Soma(diameter = diam_soma)


if morphology == 'ball+stick':
	morpho = Soma(diameter=diam_soma)

	if dend_shape == 'cylinder':
		morpho.dendrite = Cylinder(diameter=diam_cln, length=length, n=n_cmp)

	if dend_shape == 'cone':

		alpha = (diam_0 - diam_f) / ((length / um) ** beta)
		diameterT = - alpha * lengthT ** beta + diam_0
		morpho.dendrite = Section(diameter=diameterT * um, length=[len_comp / um] * n_cmp * um, n=n_cmp)


# there is an error (Magic Network) when there are no spikes


date_time = time.strftime('%y-%m-%d--%H-%M-%S', time.localtime())

if not os.path.exists('../DATA/HH/' + str(date_time) + '_[scan_' + chosen_param_short + ']_[' + morphology + ']'):
	os.makedirs('../DATA/HH/' + str(date_time) + '_[scan_' + chosen_param_short + ']_[' + morphology + ']')

main_path = os.getcwd()
os.chdir('../DATA/HH/' + str(date_time) + '_[scan_' + chosen_param_short + ']_[' + morphology + ']')


saved_param = {'date_time': str(date_time),
			   'morphology': morphology,
			   'diam_soma': diam_soma,
			   'diam_0': diam_0,
			   'diam_f': diam_f,
			   'beta': beta,
			   'length': length,
			   'number_of_comp': n_cmp,
			   'ns_ampa': ns_ampa,
			   'gaba_ratio': gaba_ratio,
			   'chosen_param': chosen_param,
			   'rate_per_weight': rate_per_weight,
			   'jitter': jtr,
			   'wgh': wgh,
			   'param_to_scan': param_to_scan[chosen_param],
			   'corr_scan': corr_scan,
			   }

if chosen_param == 'ch_dns':
	saved_param['rate_ch_dns'] = rate_ch_dns


np.save('parameters.npy', saved_param)


print('\n    *** Chosen parameter: ' + str(chosen_param) + ' ***\n\n')

t0 = time.time()

counter = 0

for param in param_to_scan[chosen_param]:

	print('\n\n    *** ' + str(chosen_param) + ' = ' + str(round(param,4)) + ' *** \n')

	if mode == 'detailed':
		os.makedirs(str(chosen_param) + '_' + str(param))

	globals()[chosen_param] = param

	rate_ampa = rate_per_weight * Hz
	rate_gaba = rate_per_weight * Hz


	if chosen_param == 'ch_dns':
		rate_per_weight = rate_ch_dns[counter]
		rate_ampa = rate_per_weight * Hz
		rate_gaba = rate_per_weight * Hz

	if chosen_param == 'wgh':

		rate_per_weight = rate_wgh[counter]
		rate_ampa = rate_per_weight * Hz
		rate_gaba = rate_per_weight * Hz



	# parameters from Pare 1998

	eqs = """

	Im = g_l * (E_l-v) + g_Na * m**3 * h * (E_Na-v) + g_K * n**4 * (E_K-v) : amp/meter**2
	
	dm/dt = alpham * (1-m) - betam * m : 1
	dh/dt = alphah * (1-h) - betah * h : 1
	dn/dt = alphan * (1-n) - betan * n : 1

	alpham = - 0.32/mV * (v - v_th - 13 * mV) / (exp(-(v - v_th - 13 * mV)/(4 * mV)) - 1) /ms : Hz
	betam = 0.28/mV * (v - v_th - 40 * mV) / (exp((v - v_th - 40 * mV)/(5 * mV)) -1) /ms : Hz
	
	alphah = 0.128 * exp(-(v - v_th - 17 * mV) / (18 * mV)) /ms : Hz
	betah = 4 / (1 + exp(-(v - v_th - 40 * mV) / (5 * mV))) /ms : Hz
	
	alphan = -0.032 / mV * (v - v_th - 15 * mV) / (exp(-(v - v_th - 15 * mV) / (5 * mV)) - 1) / ms : Hz
	betan = 0.5 * exp(-(v - v_th - 10 * mV) / (40 * mV)) /ms : Hz
	
	Is = g_ampa * (E_ampa - v) + g_gaba * (E_gaba - v) + I : amp (point current)

	dg_ampa/dt = -g_ampa/tau_ampa : siemens
	dg_gaba/dt = -g_gaba/tau_gaba : siemens
	
	I : amp
	g_Na : siemens / meter**2
	g_K : siemens / meter**2

	"""

	corr_frq = [[0, 0]]

	for rG in corr_scan.tolist() * n_rep:

		if chosen_param != 'rL':
			rL = 1

		print('\n\nCorrelation: ' + str(round(rG,2)) + '\n')

		neuron_with_dendrite = SpatialNeuron(morphology = morpho, model = eqs,
											 method = method, dt = dt, Cm = Cm, Ri = Ri)

		neuron_with_dendrite.v = -70 * mV
		neuron_with_dendrite.h = 0.
		neuron_with_dendrite.m = 0.
		neuron_with_dendrite.n = 0.
		neuron_with_dendrite.I = 0.

		neuron_with_dendrite.g_Na[0] = g_Na
		neuron_with_dendrite.g_Na[1:] = ch_dns * g_Na

		neuron_with_dendrite.g_K[0] = g_K
		neuron_with_dendrite.g_K[1:] = ch_dns * g_K


		trace = StateMonitor(neuron_with_dendrite, 'v', record=True)

		spt_exc = spike_trains_hierarch(n_cmp, ns_ampa, rate_ampa/Hz, t_sim/second, rG, rG * rL, jtr/second)

		range_of_compartments = range(n_cmp)

		if morphology == 'ball+stick':
			range_of_compartments = range(1,n_cmp)


		times_all_exc = []
		compartments_all_exc = []

		for k in range_of_compartments:
			for i_s in range(ns_ampa):

				times = spt_exc[k][i_s]
				condition = times > 0.

				times = np.extract(condition, times)
				times_all_exc = np.concatenate((times_all_exc, times))

				compartments_all_exc = np.concatenate((compartments_all_exc, [int(k)] * len(times)))


		if times_all_exc.tolist() != []:

			indices_all_exc = np.array(range(len(times_all_exc)))
			inp_exc = SpikeGeneratorGroup(len(times_all_exc), indices_all_exc, times_all_exc * second)
			syn_exc = Synapses(inp_exc, neuron_with_dendrite, on_pre = 'g_ampa += ' + str(wgh) + '* dg_gaba')
			syn_exc.connect(i = indices_all_exc.astype(int), j = compartments_all_exc.astype(int))


		# inhibitory synapses method 2
		spt_inh = spike_trains_hierarch(1, ns_gaba_soma, rate_gaba / Hz, t_sim / second, 0, 0, jtr / second)
		times_all_inh = []
		compartments_all_inh = []

		for i_s in range(ns_gaba_soma):

			times = spt_inh[0][i_s]
			condition = times > 0.

			times = np.extract(condition, times)
			times_all_inh = np.concatenate((times_all_inh, times))

			compartments_all_inh = np.concatenate((compartments_all_inh, [0] * len(times)))

		if times_all_inh.tolist() != []:

			indices_all_inh = np.array(range(len(times_all_inh)))
			inp_inh = SpikeGeneratorGroup(len(times_all_inh), indices_all_inh, times_all_inh * second)
			syn_inh = Synapses(inp_inh, neuron_with_dendrite, on_pre = 'g_gaba += ' + str(wgh) + '* dg_gaba')
			syn_inh.connect(i = indices_all_inh.astype(int), j = compartments_all_inh.astype(int))

		if times_all_exc.tolist() == [] or times_all_inh.tolist() == []:

			continue


		# spt_inh = spike_trains_hierarch(n_cmp, ns_gaba, rate_gaba/Hz, t_sim/second, rG, rG, jtr/second)
        #
		# times_all_inh = []
		# compartments_all_inh = []
        #
		# for k in range_of_compartments:
		# 	for i_s in range(ns_gaba):
        #
		# 		times = spt_inh[k][i_s]
		# 		condition = times > 0.
        #
		# 		times = np.extract(condition, times)
		# 		times_all_inh = np.concatenate((times_all_inh, times))
        #
		# 		compartments_all_inh = np.concatenate((compartments_all_inh, [int(k)] * len(times)))
        #
        #
		# if times_all_inh.tolist() != []:
        #
		# 	indices_all_inh = np.array(range(len(times_all_inh)))
		# 	inp_inh = SpikeGeneratorGroup(len(times_all_inh), indices_all_inh, times_all_inh * second)
		# 	syn_inh = Synapses(inp_inh, neuron_with_dendrite, on_pre = 'g_gaba += ' + str(weight_gaba) + '* dg_gaba')
		# 	syn_inh.connect(i = indices_all_inh.astype(int), j = compartments_all_inh.astype(int))
        #
        #
		# if times_all_exc.tolist() == [] or times_all_inh.tolist() == []:
        #
		# 	continue


		tr = time.time()
		run(t_sim, report = 'text')
		print('Time of run: ' + str(time.time() - tr) + 's')


		if mode == 'detailed':

			figure()

			plot(trace.t/second,trace.v[0].T/mV )
			xlabel('time (ms)')
			ylabel('membrane potential (mV)')

			savefig(str(chosen_param) + '_' + str(param) + '/figure_V_t_wgh_' + str(wgh) + '_corr_' + str(rG) +'.png', dpi = 300)


			figure()

			plot(trace.t / second, trace.v[100].T / mV)
			xlabel('time (ms)')
			ylabel('dendritic membrane potential (mV)')

			savefig(str(chosen_param) + '_' + str(param) + '/figure_V_t_dend_wgh_' + str(wgh) + '_corr_' + str(rG) + '.png',
					dpi=300)


			figure()

			contf = contourf(cumsum(neuron_with_dendrite.length) / cm, trace.t / ms, trace.v.T / mV, cmap='YlOrRd', alpha=1.,
							 levels=np.linspace(np.amin(trace.v / mV), np.amax(trace.v / mV), num = 100))

			colorbar(contf)
			xlabel('Position [cm]')
			ylabel('Time [ms]')

			savefig(str(chosen_param) + '_' + str(param) + '/figure_V_t_contour_wgh' + str(wgh) + '_corr_' + str(rG) +'.png', dpi=300)


		# Spike counting

		vT = trace.v[0] /mV
		th = 0.

		df_vT = vT - th
		df_vT_rl = np.roll(df_vT,1)

		th_cr_det = df_vT[1:] * df_vT_rl[1:]
		ind_cr = np.where(th_cr_det < 0.)

		n_sp = float(len(ind_cr[0])) / 2.
		frq = n_sp / t_sim

		print('Frequency of spikes: ' + str(frq))

		corr_frq = np.append(corr_frq, [[rG, frq]], axis=0)


		if morphology == 'ball+stick':

			vT_dend = trace.v[100] /mV

			th_dend = -20.

			df_vT_dend = vT_dend - th_dend
			df_vT_dend_rl = np.roll(df_vT_dend, 1)

			th_cr_det_dend = df_vT_dend[1:] * df_vT_dend_rl[1:]
			ind_cr_dend = np.where(th_cr_det_dend < 0.)

			n_sp_dend = float(len(ind_cr_dend[0])) / 2.
			frq_dend = n_sp_dend / t_sim

			print('Frequency of dendritic spikes: ' + str(frq_dend))

	corr_frq = corr_frq[1:]
	corr_mean_frq = [mean(corr_frq[i:len(corr_frq):len(corr_scan)], axis=0).tolist() for i in range(len(corr_scan))]

	fig_n_sp = figure()

	plot(corr_frq[:, 0], corr_frq[:, 1], 'b+')
	plot(array(corr_mean_frq)[:, 0], array(corr_mean_frq)[:, 1], 'r+')

	savefig('figure_frq_weight_' + str(wgh) + '_jtr_' + str(jtr) + '_rate_per_weight_' + str(
		round(rate_per_weight, 4)) + '_rL_' + str(rL) + '_ch_dns_' + str(ch_dns) + '.png', dpi=300)
	np.save('corr_frq_weight_'  + str(wgh) + '_jtr_' + str(jtr) + '_rate_per_weight_'+ str(
		round(rate_per_weight, 4)) + '_rL_' + str(rL) + '_ch_dns_' + str(ch_dns) + '.npy', corr_frq)

	counter = counter + 1


t_cmp = (time.time() - t0) / 3600.

print('Time of computation: ' + str(t_cmp) + ' hours')

n_runs = len(param_to_scan[chosen_param] * n_rep * len(corr_scan))

print('For ' + str(n_runs) + ' of ' + str(t_sim/second) + ' s runs \n')
print('Which gives ' + str(round((t_cmp * 3600) / (n_runs * t_sim/second),2)) + ' s of computation for 1 s' )


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = 'Museo Sans'
plt.rcParams["font.size"] = "22"


corr_frq = {}
corr_frq_mean = {}
corr_frq_std = {}


n_corr = len(corr_scan)

low_plot = 0
high_plot = len(param_to_scan[str(chosen_param)])


maxT = []

counter = 0

for param in param_to_scan[chosen_param]:

    if chosen_param == 'ch_dns':
        rate_per_weight = rate_ch_dns[counter]

    if chosen_param == 'wgh':
        rate_per_weight = rate_wgh[counter]

    globals()[chosen_param] = param

    corr_frq[str(param)] = np.load('corr_frq_weight_'  + str(wgh) + '_jtr_' + str(jtr) + '_rate_per_weight_' + str(
		round(rate_per_weight,3)) + '_rL_' + str(rL) + '_ch_dns_' + str(ch_dns) + '.npy')
    corr_list = corr_frq[str(param)][:,0]

    bool_corr = {}
    corr_frq_red = {}

    all_mean = [[0,0]]
    all_std = [[0,0]]

    for rG in corr_scan:

        bool_corr[str(rG)] = np.where(corr_list == np.ones(len(corr_list)) * rG)[0]

        corr_frq_red[str(rG)] = corr_frq[str(param)][bool_corr[str(rG)]]

        mean = np.mean(corr_frq_red[str(rG)], axis = 0)
        std = np.std(corr_frq_red[str(rG)], axis = 0)

        all_mean = np.append(all_mean, [mean],axis = 0)
        all_std = np.append(all_std, [std],axis = 0)

    corr_frq_mean[str(param)] = all_mean[1:]
    corr_frq_std[str(param)] = all_std[1:]

    meanT = np.array(corr_frq_mean[str(param)])[:,1]

    mean_without_nan = meanT[~np.isnan(meanT)]

    maxT = np.append(maxT, np.amax(mean_without_nan))

    counter = counter + 1


maxmaxT = np.amax(maxT)

color_array = np.linspace(0.3, 1, num=len(param_to_scan[chosen_param]))
color_array = color_array.tolist()
#color_array_reversed = color_array.reverse()


param_to_scan[chosen_param] = param_to_scan[chosen_param][low_plot:high_plot]



# labels

labels = array(param_to_scan[chosen_param])

if chosen_param == 'rate_per_weight':
    labels = array(param_to_scan[chosen_param]) * wgh


fig_corr = plt.figure(figsize=(12,7))

ax = fig_corr.add_subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.9, box.height])

ax.set_ylim(bottom = 0, top = 1.2 * maxmaxT )
ax.set_xlabel('ratio of shared spikes', fontsize = 24)
ax.set_ylabel('firing rate [Hz]', fontsize = 24)


clr = 0

for param in param_to_scan[chosen_param]:

    globals()[chosen_param] = param

    c = cm.YlOrRd(color_array[clr], 1)

    upper_limit = np.array(corr_frq_mean[str(param)])[:,1] + np.array(corr_frq_std[str(param)])[:,1]
    lower_limit = np.array(corr_frq_mean[str(param)])[:,1] - np.array(corr_frq_std[str(param)])[:,1]

    ax.plot(np.array(corr_frq_mean[str(param)])[:,0], np.array(corr_frq_mean[str(param)])[:,1], color = c, label = labels[clr])
    ax.fill_between(np.array(corr_frq_mean[str(param)])[:,0], lower_limit, upper_limit, color = c, alpha = 0.5)

    leg = plt.legend()
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
     	line.set_linewidth(4)

    clr = clr + 1


ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5), frameon = False, fontsize = 22)


plt.savefig('[scan_' + chosen_param_short + '].png', dpi = 300)
plt.savefig('[scan_' + chosen_param_short + '].svg', dpi = 300)

np.save('chosen_param.npy', chosen_param)
np.save('param_to_scan.npy', param_to_scan[chosen_param])
np.save('corr_frq_mean.npy', corr_frq_mean)
np.save('corr_frq_std.npy', corr_frq_std)


os.chdir(main_path)