'''
	Base class for temporal coding schemes
'''
## Standard Library Imports

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from indirect_toflib.CodingFunctions  import *
from combined_toflib.combined_tof_utils import *


def init_coding_functions_list(coding_ids, n_tbins, params):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	coding_list = []
	params['num_measures'] = []
	for i in range(len(coding_ids)):
		curr_coding_id = coding_ids[i]
		curr_coding = create_coding_obj(curr_coding_id, n_tbins, params)
		coding_list.append(curr_coding)
	return coding_list

def create_coding_obj(coding_id, n_tbins, params):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	ktaps = params['K']
	peak_factor = params['peak_factor']
	ModFs = None
	DemodFs = None
	num_measures = ktaps
	if(coding_id == 'KTapSinusoid'):
		(ModFs, DemodFs) = GetCosCos(N=n_tbins, K=ktaps)
	elif(coding_id == 'HamiltonianK3'):
		#peak_factor = peak_factor/3
		(ModFs,DemodFs) = GetHamK3(N = n_tbins, PeakFactor=peak_factor)
	elif (coding_id == 'HamiltonianK3Gated'):
		#peak_factor = peak_factor/4
		num_measures = 4
		(ModFs, DemodFs) = GetHamK3(N=n_tbins, PeakFactor=peak_factor)
	elif(coding_id == 'HamiltonianK4'):
		#peak_factor = peak_factor/4
		(ModFs, DemodFs) = GetHamK4(N=n_tbins, PeakFactor=peak_factor)
	elif (coding_id == 'HamiltonianK4Gated'):
		#peak_factor = peak_factor/7
		num_measures = 7
		(ModFs, DemodFs) = GetHamK4(N=n_tbins, PeakFactor=peak_factor)
	elif(coding_id == 'HamiltonianK5'):
		#peak_factor = peak_factor/5
		(ModFs, DemodFs) = GetHamK5(N=n_tbins, PeakFactor=peak_factor)
	elif (coding_id == 'HamiltonianK5Gated'):
		#peak_factor = peak_factor/16
		num_measures = 16
		(ModFs, DemodFs) = GetHamK5(N=n_tbins, PeakFactor=peak_factor)

	params['num_measures'].append(num_measures)
	return (ModFs, DemodFs)

