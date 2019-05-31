

from brian2 import *
from inputs import *
import os
import time
import numpy


#BrianLogger.suppress_hierarchy('brian2.codegen')
#BrianLogger.suppress_name('resolution_conflict')
#BrianLogger.suppress_hierarchy('brian2')


# passive
Cm = 1e-6 * farad * cm**-2
Ri = 100 * ohm * cm
g_l = 1e-4 * siemens * cm**-2
E_l = -70 * mV


# adex
V_th = -50 * mV # -48 mV for local dendritic spikes and -56 mV for somatic spikes
D_th = 2 * mV
V_amp = 50 * mV # 48 70 (in original simulations) 67 mV (counting from -70 mV) (Magee: ... dendritic spikes in CA1 pyramidal neurons (2004))
V_cut = E_l + V_amp

t_rep = 1 * ms
tau_rep = - log(0.01 * mV / (V_cut - E_l)) / t_rep

tauw = 500 * ms
a = 0. * 1e-3 * siemens * cm ** -2
b = 0. * amp * cm ** -2



# synaptic

E_ampa = 0 * mV
E_gaba = -75 * mV

tau_ampa = 5 * ms
tau_gaba = 5 * ms

dg_ampa = 1 * nS
dg_gaba = 1 * nS



### HERE PARAMETERS TO CHANGE ###

morphology = 'point'		# point or ball+stick
dend_shape = 'cone' 		# cylinder or cone
dendritic_spikes = 'on'
adaptation = 'off'

mode = 'simple'  		# detailed or simple
speed = 'long'			# 'very fast' 'fast' 'long' 'test speed'

chosen_param = 't_ref'		# 't_ref' 'rate_per_weight' 'wgh' 'b' 'rL, a'




ns_ampa_tot = 200
gaba_ratio = 0.2
jtr = 10. * ms #  plots for madex 10 ms; plots for hh 30 ms


if adaptation == 'on':
	tauw = 500 * ms
	a = 0. * 1e-3 * siemens * cm ** -2
	b = 0. * amp * cm ** -2




### NUMERICAL
method = 'rk2' 			# 'linear', 'exponential_euler', 'rk2 or 'heun'
dt = 0.05 * ms


### MORPHOLOGY

diam_soma = 40 * um    # 40

# length of dendrite
length = 1000 * um
len_comp = 5 * um  # 5 !!!!
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
	ns_gaba_soma = int(round(ns_ampa * gaba_ratio))

if morphology == 'ball+stick':
	ns_ampa = int(ns_ampa_tot / n_cmp)
	ns_gaba_soma = int(round(ns_ampa_tot * gaba_ratio))

	if chosen_param == 'rL':
		ns_ampa = 10 # [[10]]

print('number of excitatory synapses')
print(ns_ampa)
print(ns_gaba_soma)

### FIXED PARAMETERS

wgh = 0.105 # [0.08] [[0.05]] 0.1
#wgh = 0.5 # [0.08] [[0.05]] 0.1
rate_per_weight = 2. # 0.7 # [[0.07]] 0.05 for 10 synapses on 5um compartment [PASSIVE: 30]] [ACTIVE: 0.5]] [SOMA: 1]   [wgh scan: ACTIVE: 0.4, PASSIVE: 30, SOMA: 5]
t_ref = 5 * ms # 5 * ms
jtr = 10. * ms # [20 * ms] [10 * ms : right one first scans]


### SCANNED PARAMETERS

rate_range = 30. #  [WITH DENDRITE [OFF: high 100 / low 40] [ON: high 4 / low 1] [CYLINDER: 4 [1um]]] [SOMA: 8]
t_ref_range = 20 * ms # 2-4 ms absolute to 15 ms relative
param_to_scan = {}

# rate
param_to_scan['rate_per_weight'] = np.array([0.1, 0.25, 0.5, 0.75, 1.]) * rate_range
param_to_scan['rate_per_weight'] = np.array([0.1, 0.5, 1.]) * rate_range

param_to_scan['rate_per_weight'] = param_to_scan['rate_per_weight'].tolist()

# refractory period
param_to_scan['t_ref'] = np.array([3., 6.,10.]) * 1 * ms
param_to_scan['t_ref'] = param_to_scan['t_ref'].tolist()

# for neuron with dendrite

rate_t_ref = np.array([1.7, 2, 2.2])  # it was 2 then 1.85
rate_t_ref = np.array([2., 2.25, 2.5])  # it was 2 then 1.85
rate_t_ref = np.array([2.15, 2.3, 2.7])  # it was 2 then 1.85
rate_t_ref = np.array([2.07, 2.4, 3])  # it was 2 then 1.85

# for point neuron

rate_t_ref = np.array([18, 20, 22])  # it was 2 then 1.85
rate_t_ref = np.array([19.5, 20, 20.5])  # it was 2 then 1.85


# weight
param_to_scan['wgh'] = [0.1, 0.25, 0.5, 0.75, 1., 1.5]




# adaptation

b_range = 1. * 10**(-7) *  amp * cm ** -2 # 1.33 * 10**(-7)
param_to_scan['b'] = [0, 1, 10] *  b_range
param_to_scan['b'] = param_to_scan['b'].tolist()


# a_range =





# local correlation
param_to_scan['rL'] = [0., 0.1, 0.25, 0.5, 0.75, 1.]

# correlation scan
corr_scan = np.arange(0.,1.1,0.1)


if speed == 'test speed':

    t_sim = 1000 * ms
    n_rep = 20
    corr_scan = np.array([0])

    chosen_param = 'wgh'
    param_to_scan['wgh'] = array([0.5])
    param_to_scan['wgh'] = param_to_scan['wgh'].tolist()


if speed == 'fast':
	t_sim = 3000 * ms
	n_rep = 1

if speed == 'long':
	t_sim = 20000 * ms
	n_rep = 10


if mode == 'detailed':
    t_sim = 500 * ms
    n_rep = 1




if morphology == 'ball+stick':

	morpho = Soma(diameter=diam_soma)

	if dend_shape == 'cylinder':
		morpho.dendrite = Cylinder(diameter=diam_cln, length=length, n=n_cmp)

	if dend_shape == 'cone':

		alpha = (diam_0 - diam_f) / ((length / um) ** beta)

		diameterT = - alpha * lengthT ** beta + diam_0

		morpho.dendrite = Section(diameter=diameterT * um, length=[len_comp / um] * n_cmp * um, n=n_cmp)


if morphology == 'point':

	length = diam_soma
	n_cmp = 1

	morpho = Soma(diameter = diam_soma)


# there is an error (Magic Network) when there are no spikes


date_time = time.strftime('%y-%m-%d--%H-%M-%S', time.localtime())

if not os.path.exists('../DATA/AE/' + str(date_time) + '_[' + morphology + ']_[dspikes_' + dendritic_spikes + ']_[dshape_' + dend_shape + ']_[scan_' + chosen_param + ']'):
	os.makedirs('../DATA/AE/' + str(date_time) + '_[' + morphology + ']_[dspikes_' + dendritic_spikes + ']_[dshape_' + dend_shape + ']_[scan_' + chosen_param + ']')

main_path = os.getcwd()
os.chdir('../DATA/AE/' + str(date_time) + '_[' + morphology + ']_[dspikes_' + dendritic_spikes + ']_[dshape_' + dend_shape + ']_[scan_' + chosen_param + ']')


saved_param = {'date_time': str(date_time),
			   'morphology': morphology,
			   'dendritic_spikes': dendritic_spikes,
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
			   't_ref': t_ref,
			   'jitter': jtr,
			   'wgh': wgh,
			   'param_to_scan': param_to_scan[chosen_param],
			   'corr_scan': corr_scan}


np.save('parameters.npy', saved_param)


t0 = time.time()

counter = 0

print('\n    *** Chosen parameter: ' + str(chosen_param) + ' ***\n\n')

for param in param_to_scan[chosen_param]:

	print('\n\n    *** Parameter value: ' + str(param) + ' *** \n')

	if mode == 'detailed':
		os.makedirs(str(chosen_param) + '_' + str(param))

	globals()[chosen_param] = param

	b_tag = np.asarray(b)

	if chosen_param == 't_ref':

		rate_per_weight = rate_t_ref[counter]

	print(b_tag)

	weight_ampa = wgh
	weight_gaba = wgh

	rate_ampa = rate_per_weight * Hz
	rate_gaba = rate_per_weight * Hz


	eqs = """
	Im = g_l * (E_l - v) + exp_c * int(not_refractory)
	- tau_rep * (v - E_l) * Cm * (1. - int(not_refractory)) : amp * meter**-2

	Is = g_ampa * (E_ampa - v) + g_gaba * (E_gaba - v) + I : amp (point current)

	exp_c : amp * meter**-2

	dg_ampa/dt = -g_ampa/tau_ampa : siemens
	dg_gaba/dt = -g_gaba/tau_gaba : siemens

	I : amp
	"""


	corr_frq = [[0,0]]

	for rG in corr_scan.tolist() * n_rep:

		if chosen_param != 'rL':
			rL = 1


		print('\n\nCorrelation: ' + str(round(rG,2)) + '\n')

		if dendritic_spikes == 'on':

			neuron_with_dendrite = SpatialNeuron(morphology = morpho, model = eqs,
							threshold = 'v > V_cut', reset = "v = V_cut",
							refractory = t_ref, method = method, dt = dt,
							Cm = Cm, Ri = Ri)

			neuron_with_dendrite.run_regularly('exp_c = g_l * D_th * exp((v - V_th) / D_th)')

			neuron_with_dendrite.run_regularly('v = clip(v, -inf * mV, V_cut)', when='end')


			# neuron_with_dendrite.w = 0 * amp * meter**-2

		neuron_with_dendrite.v = E_l


		if mode == 'detailed':

			trace = StateMonitor(neuron_with_dendrite, 'v', record=True)

		spikes = SpikeMonitor(neuron_with_dendrite, variables='v')

		#spt_exc = spike_trains_hierarch_ind_global(n_cmp, ns_ampa, rate_ampa/Hz, t_sim/second, rL, rG, jtr/second)
		spt_exc = spike_trains_hierarch(n_cmp, ns_ampa, rate_ampa/Hz, t_sim/second, rG, rG, jtr/second)


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

			syn_exc = Synapses(inp_exc, neuron_with_dendrite, on_pre = 'g_ampa += ' + str(weight_ampa) + '* dg_ampa')

			syn_exc.connect(i = indices_all_exc.astype(int), j = compartments_all_exc.astype(int))




		times_all_inh = []
		compartments_all_inh = []

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
			syn_inh = Synapses(inp_inh, neuron_with_dendrite, on_pre='g_gaba += ' + str(wgh) + '* dg_gaba')
			syn_inh.connect(i=indices_all_inh.astype(int), j=compartments_all_inh.astype(int))



		tr = time.time()
		run(t_sim, report = 'text')
		print('Time of run: ' + str(time.time() - tr) + 's')


		if mode == 'detailed':

			figure()

			plot(trace.t/second,trace.v[0].T/mV )
			xlabel('time (ms)')
			ylabel('membrane potential (mV)')

			#savefig(str(chosen_param) + '_' + str(param) + '/figure_V_t_wgh_' + str(wgh) + '_corr_' + str(rG) +'.png', dpi = 300)
			savefig('figure_V_t_wgh_' + str(wgh) + '_corr_' + str(rG) +'.png', dpi = 300)


			figure()

			contf = contourf(cumsum(neuron_with_dendrite.length) / cm, trace.t / ms, trace.v.T / mV, cmap='YlOrRd', alpha=1.,
							 levels=np.arange(-90, 10, 0.5))

			colorbar(contf)
			xlabel('Position [cm]')
			ylabel('Time [ms]')

			#savefig(str(chosen_param) + '_' + str(param) + '/figure_V_t_contour_wgh' + str(wgh) + '_corr_' + str(rG) +'.png', dpi=300)
			savefig('figure_V_t_contour_wgh' + str(wgh) + '_corr_' + str(rG) +'.png', dpi=300)


		# Counting spikes

		if len(spikes.i) != 0 and len(spikes.i) != 1:

			cmp_sp = [[int(spikes.i[0]), spikes.t[0]]]

			for j in range(1,len(spikes.i)):

				cmp_sp = np.append(cmp_sp, [[int(spikes.i[j]), spikes.t[j]]], axis = 0)

			ind = np.where([cmp_sp[:,0] == 0])[1]
			frq = len(ind) / (t_sim/second)

			print('Frequency of spiking: ' + str(frq))

		else:

			frq = 0.

			print('No spikes')


		corr_frq = np.append(corr_frq, [[rG,frq]], axis = 0)



	counter = counter + 1


	corr_frq = corr_frq[1:]
	corr_mean_frq = [mean(corr_frq[i:len(corr_frq):len(corr_scan)], axis = 0).tolist() for i in range(len(corr_scan))]

	# plots

	fig_n_sp = figure()

	plot(corr_frq[:,0], corr_frq[:,1], 'b+')
	plot(array(corr_mean_frq)[:,0], array(corr_mean_frq)[:,1], 'r+')

	savefig('figure_frq_weight_'  + str(wgh)  + '_t_ref_' + str(t_ref) + '_jtr_' + str(jtr) + '_rate_per_weight_' + str(rate_per_weight) + '_rL_' + str(rL) + '_b_' + str(b_tag) + '.png', dpi = 300)
	np.save('corr_frq_weight_'  + str(wgh)  + '_t_ref_' + str(t_ref) + '_jtr_' + str(jtr) + '_rate_per_weight_' + str(rate_per_weight) + '_rL_' + str(rL) + '_b_' + str(b_tag) + '.npy', corr_frq)



# time of simulation

tf = time.time()
t_calc = time.time() - t0


print('\nTotal time of run:')
print(str(round(t_calc / 60.,2)) + ' min')

print('\nTime of single run :')
t_calc_s = t_calc / (len(corr_scan) * n_rep * len(param_to_scan[chosen_param]))
print(str(round(t_calc_s / 60.,2)) + ' min')

print('\nEstimated time of full scan')
print(str(t_calc_s / 3600. * 2200.) + 'h')



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams["font.size"] = "13"



corr_frq = {}
corr_frq_mean = {}
corr_frq_std = {}


n_corr = len(corr_scan)

low_plot = 0
high_plot = len(param_to_scan[str(chosen_param)])


maxT = []

counter = 0

for param in param_to_scan[chosen_param]:

    globals()[chosen_param] = param

    b_tag = np.asarray(b)

    if chosen_param == 't_ref':
        rate_per_weight = rate_t_ref[counter]


    corr_frq[str(param)] = np.load('corr_frq_weight_' + str(wgh)  + '_t_ref_' + str(t_ref) + '_jtr_' + str(jtr) + '_rate_per_weight_' + str(rate_per_weight) + '_rL_' + str(rL) + '_b_' + str(b_tag) + '.npy')
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
color_array_reversed = color_array.reverse()


param_to_scan[chosen_param] = param_to_scan[chosen_param][low_plot:high_plot]

labels = array(param_to_scan[chosen_param])


if chosen_param == 'rate_per_weight':
    labels = array(param_to_scan[chosen_param]) * wgh


if chosen_param == 't_ref':
    labels = array(param_to_scan[chosen_param]) * 10**3



fig_corr = plt.figure(figsize=(10,5.5))

ax = fig_corr.add_subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.9, box.height])

ax.set_ylim(bottom = 0, top = 1.2 * maxmaxT )
ax.set_xlabel('ratio of shared spikes')
ax.set_ylabel('firing rate [Hz]')


clr = 0

for param in param_to_scan[chosen_param]:

    globals()[chosen_param] = param

    c = cm.YlOrRd(color_array[clr], 1)

    upper_limit = np.array(corr_frq_mean[str(param)])[:,1] + np.array(corr_frq_std[str(param)])[:,1]
    lower_limit = np.array(corr_frq_mean[str(param)])[:,1] - np.array(corr_frq_std[str(param)])[:,1]

    ax.plot(np.array(corr_frq_mean[str(param)])[:,0], np.array(corr_frq_mean[str(param)])[:,1], color = c, label = labels[clr])

    ax.fill_between(np.array(corr_frq_mean[str(param)])[:,0], lower_limit, upper_limit, color = c, alpha = 0.5)

    clr = clr + 1


ax.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))

plt.savefig('corr_sp_' + chosen_param + '.png', dpi = 300)
plt.savefig('corr_sp2_' + chosen_param + '.svg', dpi = 300)

np.save('corr_frq_mean.npy', corr_frq_mean)
np.save('corr_frq_std.npy', corr_frq_std)


os.chdir(main_path)






