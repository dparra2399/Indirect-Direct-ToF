'''
	Base class for temporal coding schemes
'''
## Standard Library Imports

## Library Imports
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from indirect_toflib.CodingFunctions  import *


def init_coding_functions_list(coding_ids, n_tbins, params):
	'''
		args are the input arguments object obtained from input_args_utils.py:add_coding_args
	'''
	coding_list = []
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
	if(coding_id == 'KTapSinusoid'):
		(ModFs, DemodFs) = GetCosCos(N=n_tbins, K=ktaps)
	elif(coding_id == 'HamiltonianK3'):
		(ModFs,DemodFs) = GetHamK3(N = n_tbins, PeakFactor=peak_factor)
	elif(coding_id == 'HamiltonianK4'):
		(ModFs, DemodFs) = GetHamK4(N=n_tbins, PeakFactor=peak_factor)
	elif(coding_id == 'HamiltonianK5'):
		(ModFs, DemodFs) = GetHamK5(N=n_tbins, PeakFactor=peak_factor)

	return (ModFs, DemodFs)

