'''
	Base class for temporal impulse response functions
'''
## Standard Library Imports
from abc import ABC, abstractmethod

## Library Imports
from IPython.core import debugger

breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.research_utils.shared_constants import *
from felipe_utils.research_utils.signalproc_ops import circular_conv, gaussian_pulse, expgaussian_pulse_conv
from felipe_utils.research_utils import np_utils
from felipe_utils.felipe_impulse_utils import tof_utils_felipe as tof_utils
import matplotlib as mpl

mpl.use('qt5agg')


def init_gauss_pulse_list(n_tbins, pulse_widths, mu=0, t_domain=None):
    '''
		For each pulse width generate a gaussian pulse.
		If mu is a list, then for each pulse width we generate multiple gaussian pulses for each mu.
	'''
    pulses_list = []
    for i in range(len(pulse_widths)):
        pulses = GaussianTIRF(n_tbins, mu=mu, sigma=pulse_widths[i], t_domain=t_domain)
        pulses_list.append(pulses)
    return pulses_list


class TemporalIRF(ABC):
    '''
		Class that holds temporal impulse response function data.
		The last dimension of the tirf_data should be time.
		The tirf data is considered a probability distribution from which we simulate
	'''

    def __init__(self, tirf_data, t_domain=None, sbr=None, ambient=None, T=0.1, mean_beta=1):
        self.tirf = tirf_data.astype(np.float32)
        self.n_tbins = self.tirf.shape[-1]
        self.n_tirfs = int(self.tirf.size / self.n_tbins)
        # Set a dummy time domain if needed
        self.set_t_domain(t_domain)
        # Set sbr to something if provided
        self.set_sbr(sbr)
        self.set_ambient(ambient)
        self.set_mean_beta(mean_beta)
        self.set_integration_time(T)
        # Find tirf pixels with no signal
        self.nosignal_mask = np.mean(self.tirf, axis=-1) < EPSILON
        self.nonzero_signal_mask = np.logical_not(self.nosignal_mask)
        self.tmp_tirf = np.zeros_like(self.tirf)

    def apply(self, v):
        assert (v.shape[-1] == self.n_tbins), "last dim of v needs to match n_tbins ({})".format(self.n_tbins)
        return circular_conv(self.tirf, v, axis=-1)

    def set_t_domain(self, t_domain):
        if (t_domain is None):
            self.t_domain = np.arange(0, self.n_tbins)
        else:
            assert (t_domain.shape[-1] == self.n_tbins), "Input t_domain need to match tirf_data last dim"
            self.t_domain = t_domain

    def set_sbr(self, input_sbr):
        if (input_sbr is None):
            self.sbr = None
            return
        self.sbr = np_utils.to_nparray(input_sbr)
        assert ((self.sbr.size == 1) or (self.sbr.shape == self.tirf.shape[
                                                           0:-1])), "input sbr should be a number OR should be an array that matches the first N-1 dims of self.tirf"

    def set_integration_time(self, input_t):
        if (input_t is None):
            self.t = None
            return
        self.t = input_t

    def set_ambient(self, input_amb, tau=1):
        if (input_amb is None):
            self.ambient = None
            return
        self.ambient = np_utils.to_nparray(input_amb) * tau
        assert ((self.ambient.size == 1) or (self.ambient.shape == self.tirf.shape[
                                                                   0:-1])), "input ambient light should be a number OR should be an array that matches the first N-1 dims of self.tirf"

    def set_mean_beta(self, input_beta):
        if (input_beta is None):
            self.mean_beta = 1
            return
        self.mean_beta = input_beta

    def simulate_exactly_n_photons(self, n_photons, n_mc_samples=1):
        '''
			Simulate exactly N photons. Regardless of sbr or signal shape the output signal here will always contain n_photons
		'''
        # Simulate the pulses for all depth n_mc_samples times.
        if (self.sbr.size == 1):
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_sbr(self.tirf[self.nonzero_signal_mask],
                                                                        sbr=self.sbr, axis=-1)
        else:
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_sbr(self.tirf[self.nonzero_signal_mask],
                                                                        sbr=self.sbr[self.nonzero_signal_mask], axis=-1)
        self.tmp_tirf[self.nosignal_mask] = 0
        return tof_utils.simulate_n_photons(self.tmp_tirf, n_photons=n_photons, n_mc_samples=n_mc_samples)

    def simulate_n_photons(self, n_photons=None, n_mc_samples=1, add_noise=True):
        '''
			Simulate a waveform with N total photons and add poisson noise.
		'''
        # Simulate the pulses for all depth n_mc_samples times.
        if (self.sbr.size == 1):
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_n_photons(self.tirf[self.nonzero_signal_mask],
                                                                              n_photons=n_photons, sbr=self.sbr,
                                                                              axis=-1)
        else:
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_n_photons(self.tirf[self.nonzero_signal_mask],
                                                                              n_photons=n_photons,
                                                                              sbr=self.sbr[self.nonzero_signal_mask],
                                                                              axis=-1)
        self.tmp_tirf[self.nosignal_mask] = 0
        if (add_noise):
            # the following is the most expensive step of this function
            return tof_utils.add_poisson_noise(self.tmp_tirf, n_mc_samples=n_mc_samples)
        else:
            return self.tmp_tirf

    def simulate_peak_power(self, peak_power, pAveSource=None, num_measures=1, n_mc_samples=1, dt=1, tau=1,
                            add_noise=True):
        self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_peak_power(self.tirf[self.nonzero_signal_mask],
                                                                           peak_power, pAveSource, ambient=self.ambient,
                                                                           num_measures=num_measures, sbr=self.sbr,
                                                                           T=self.t, mean_beta=self.mean_beta, dt=dt,
                                                                           tau=tau, axis=-1)
        self.tmp_tirf[self.nosignal_mask] = 0

        ret = self.tmp_tirf

        if add_noise is True:
            ret = tof_utils.add_poisson_noise(ret, n_mc_samples=n_mc_samples)

        return ret

    def simulate_avg_power(self, pAveSource, num_measurements=1, n_mc_samples=1, dt=1, tau=1, add_noise=True):
        self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_avg_power(self.tirf[self.nonzero_signal_mask],
                                                                          pAveSource, num_measurements=num_measurements,
                                                                          ambient=self.ambient,
                                                                          sbr=self.sbr, mean_beta=self.mean_beta,
                                                                          T=self.t, dt=dt, tau=tau, axis=-1)
        self.tmp_tirf[self.nosignal_mask] = 0

        ret = self.tmp_tirf
        if add_noise is True:
            ret = tof_utils.add_poisson_noise(ret, n_mc_samples=n_mc_samples)

        return ret

    def simulate_n_signal_photons(self, n_photons=None, n_mc_samples=1):
        '''
			Simulate a signal with n_photons. The ground truth signal will have a total
			number of photons =  n_photons + (n_photons/sbr)
			We then add poisson noise to this signal.
			For pixels without a signal, we do not perform simulation, simply set them to 0
		'''
        # if(not (n_photons is None)): assert(isinstance(n_photons, (int, float))), "n_photons should be a number"
        if (self.sbr.size == 1):
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_signal_n_photons(
                self.tirf[self.nonzero_signal_mask], n_photons=n_photons, sbr=self.sbr, axis=-1)
        else:
            self.tmp_tirf[self.nonzero_signal_mask] = tof_utils.set_signal_n_photons(
                self.tirf[self.nonzero_signal_mask], n_photons=n_photons, sbr=self.sbr[self.nonzero_signal_mask],
                axis=-1)
        self.tmp_tirf[self.nosignal_mask] = 0
        return tof_utils.add_poisson_noise(self.tmp_tirf, n_mc_samples=n_mc_samples)


class DepthImgTIRF(TemporalIRF):
    def __init__(self, depth_img, n_tbins, delta_depth=1., **kwargs):
        self.n_tbins = n_tbins
        self.delta_depth = delta_depth
        self.depth_img = depth_img
        super().__init__(tirf_data=tof_utils.depthmap2tirf(self.depth_img, self.n_tbins, self.delta_depth), **kwargs)


# def depthmap2tirf(self, depth_img):
# 	# Transform depths to non-zero indeces
# 	nonzero_indeces = np.round(self.depth_img / self.delta_depth).astype(np.int)
# 	loaded_tirf = np.zeros(self.depth_img.shape + (self.n_tbins,))
# 	for i in range(loaded_tirf.shape[0]):
# 		for j in range(loaded_tirf.shape[1]):
# 			loaded_tirf[i,j,nonzero_indeces[i,j]] = 1.
# 	return loaded_tirf

class ModelBasedTIRF(TemporalIRF):
    '''
		Generate TIRF using some model (e.g., Gaussian, Exp-mod gaussian, etc)
	'''

    def __init__(self, n_tbins, t_domain=None, **kwargs):
        self.n_tbins = n_tbins
        self.set_t_domain(t_domain)
        # Create a circular gaussian pulse
        modeled_tirf = self.generate_model_tirf()
        # Initialize the regular Temporal IRF
        super().__init__(tirf_data=modeled_tirf, t_domain=t_domain, **kwargs)

    @abstractmethod
    def generate_model_tirf(self):
        pass


class GaussianTIRF(ModelBasedTIRF):

    def __init__(self, n_tbins, mu=None, sigma=None, **kwargs):
        # Set mu, and sigma params
        (self.mu, self.sigma) = (0, 1.)
        if (not (mu is None)): self.mu = mu
        if (not (sigma is None)): self.sigma = sigma
        # Initialize the regular Temporal IRF
        super().__init__(n_tbins=n_tbins, **kwargs)

    def generate_model_tirf(self):
        # Create a circular gaussian pulse
        gaussian_tirf = gaussian_pulse(self.t_domain, self.mu, self.sigma, circ_shifted=True)
        return gaussian_tirf


class ExpModGaussianTIRF(GaussianTIRF):

    def __init__(self, n_tbins, mu=None, sigma=None, exp_lambda=None, **kwargs):
        # Set exp_lambda, and sigma params
        (self.exp_lambda) = 20 * (1. / n_tbins)
        if (not (exp_lambda is None)):
            self.exp_lambda = exp_lambda
        else:
            print("Warning: No lambda given, using default lambda = {}".format(self.exp_lambda))
        # Initialize the regular Temporal IRF
        super().__init__(n_tbins=n_tbins, mu=mu, sigma=sigma, **kwargs)

    def generate_model_tirf(self):
        # Create a circular gaussian pulse
        exp_mod_gaussian_tirf = expgaussian_pulse_conv(self.t_domain, self.mu, self.sigma, self.exp_lambda)
        return exp_mod_gaussian_tirf