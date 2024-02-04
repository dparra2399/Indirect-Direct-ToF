'''
	Base class for temporal coding schemes
'''
## Standard Library Imports

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace
from direct_toflib import direct_tof_utils

## Local Imports
from direct_toflib.coding import *


def init_coding_list(coding_ids, n_tbins, params, pulses_list=None):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	coding_list = []
	for i in range(len(coding_ids)):
		curr_coding_id = coding_ids[i]
		h_irf = None
		if(not (pulses_list is None)): h_irf = pulses_list[i].tirf.squeeze()
		curr_coding = create_coding_obj(curr_coding_id, n_tbins, params, h_irf)
		coding_list.append(curr_coding)
	return coding_list

def create_coding_obj(coding_id, n_tbins, params, h_irf=None):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''

	coding_obj = None
	freq_idx = params['freq_idx']
	n_gates = params['n_gates']
	gate_size = params['gate_size']
	(rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
		(direct_tof_utils.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
	ktaps = params['K']
	coding_id = coding_id.split('_')[0]
	if(coding_id == 'KTapSinusoid'):
		coding_obj = KTapSinusoidCoding(n_maxres=n_tbins, freq_idx=freq_idx, k=ktaps, account_irf=False, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK3'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=3, account_irf=False, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK4'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=4, account_irf=False, h_irf=h_irf)
	elif(coding_id == 'HamiltonianK5'):
		coding_obj = HamiltonianCoding(n_maxres=n_tbins, k=5, account_irf=False, h_irf=h_irf)
	elif(coding_id == 'Identity'):
		coding_obj = IdentityCoding(n_maxres=n_tbins, account_irf=False, h_irf=h_irf)
	elif (coding_id == 'Gated'):
		coding_obj = GatedCoding(n_maxres=n_tbins, n_gates=n_gates, account_irf=False, h_irf=h_irf)
	elif (coding_id == 'IntegratedGated'):
		coding_obj = IntegratedGatedCoding(n_maxres=n_tbins, n_gates=n_gates, tbin_res=tbin_res, gate_size=gate_size, account_irf=False, h_irf=h_irf)

	return coding_obj

