from abc import ABC, abstractmethod
## Library Imports
from IPython.core import debugger

breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from spad_toflib.spad_tof_utils import calculate_ambient
from felipe_utils.felipe_cw_utils.CodingFunctionUtilsFelipe import ScaleMod
from felipe_utils.felipe_impulse_utils.tof_utils_felipe import *
from utils.file_utils import get_constrained_ham_codes
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


TotalEnergyDefault = 1.
TauDefault = 1.
AveragePowerDefault = TotalEnergyDefault / TauDefault


class LightSource(ABC):

    def __init__(self, light_source, depths, sbr=None, ave_source=None, ambient=None,
                 rep_tau=None, num_measures=None, t=None, mean_beta=None):
        self.light_source = light_source
        self.depths = depths
        self.set_all_params(sbr, ave_source, ambient, rep_tau, num_measures, t, mean_beta)
        # Find tirf pixels with no signal

    def set_integration_time(self, input_t):
        if (input_t is None):
            self.t = None
            return
        self.t = input_t

    def set_ambient(self, input_sbr, input_amb, dt=1, tau=1):
        if (input_amb is not None):
            self.ambient = input_amb
        elif (input_sbr is not None):
            self.sbr = input_sbr
        else:
            self.ambient = None
            self.sbr = None

    def set_ave_source(self, input_source):
        if (input_source is None):
            self.ave_source = None
            return
        self.ave_source = np_utils.to_nparray(input_source)

    def set_mean_beta(self, input_beta):
        if (input_beta is None):
            self.mean_beta = None
            return
        self.mean_beta = input_beta

    def set_tau(self, input_tau):
        if (input_tau is None):
            self.tau = None
            return
        self.tau = input_tau

    def set_num_measures(self, input_num):
        if (input_num is None):
            self.num_measures = None
            return
        self.num_measures = input_num

    def set_all_params(self, sbr=None, ave_source=None, ambient=None,
                       rep_tau=None, num_measures=None, t=None, mean_beta=None):
        self.set_ave_source(ave_source)
        self.set_mean_beta(mean_beta)
        self.set_integration_time(t)
        self.set_num_measures(num_measures)
        self.set_tau(rep_tau)
        self.set_ambient(sbr, ambient)

class SinglePhotonSource(LightSource):

    def __init__(self, n_tbins, binomial=False, light_source=None, t_domain=None, **kwargs):
        self.n_tbins = n_tbins
        self.binomial = binomial
        self.set_t_domain(t_domain)
        if light_source is None: light_source = self.generate_source()
        super().__init__(light_source=light_source, **kwargs)

    def set_t_domain(self, t_domain):
        if t_domain is None:
            self.t_domain = np.arange(0, self.n_tbins)
        else:
            assert (t_domain.shape[-1] == self.n_tbins), "Input t_domain need to match tirf_data last dim"
            self.t_domain = t_domain

    @abstractmethod
    def generate_source(self):
        pass

    @abstractmethod
    def simulate_photons(self):
        pass


class SinglePhotonContinousWave(SinglePhotonSource):
    def __init__(self, modfs=None, **kwargs):
        super().__init__(light_source=modfs, **kwargs)

    def phase_shifted(self, modfs):
        shifted_modfs = np.zeros((self.depths.shape[0], modfs.shape[1], modfs.shape[0]))
        tbin_depth_res = time2depth(self.tau / modfs.shape[0])
        for d in range(0, self.depths.shape[0]):
            for i in range(0, modfs.shape[-1]):
                shifted_modfs[d, i, :] = np.roll(modfs[:, i], int(self.depths[d] / tbin_depth_res))
        return shifted_modfs

    def simulate_photons(self):
        if self.binomial:
            return self.simulate_single_cycle()
        else:
            return self.simulate_n_cycles()

    def simulate_n_cycles(self):
        incident = np.zeros(self.light_source.shape)
        total_source_photons = self.ave_source * self.t
        total_amb_photons = total_source_photons / self.sbr
        scaled_modfs = np.copy(self.light_source)
        for i in range(0, self.light_source.shape[-1]):
            scaled_modfs[:, i] *= (total_source_photons / np.sum(self.light_source[:, i]))
            incident[:, i] = scaled_modfs[:, i] + (total_amb_photons / self.n_tbins)
        return self.phase_shifted(incident)

    def simulate_single_cycle(self):
        laser_cycles = (1. / self.tau) * self.t
        incident = self.simulate_n_cycles() / laser_cycles
        return incident



class KTapSinusoidSource(SinglePhotonContinousWave):
    def __init__(self, n_functions, modfs=None, **kwargs):
        if n_functions is None: n_functions = 3
        self.n_functions = n_functions
        super().__init__(modfs=modfs, **kwargs)

    def generate_source(self):
        modfs = np.zeros((self.n_tbins, self.n_functions))
        t = np.linspace(0, 2 * np.pi, self.n_tbins)
        cosfs = (0.5 * np.cos(t)) + 0.5
        for i in range(0, self.n_functions):
            modfs[:, i] = cosfs
        return modfs


class HamiltonianSource(SinglePhotonContinousWave):
    def __init__(self, n_functions, modfs=None, peak_factor=None, win_duty=None, **kwargs):
        self.n_functions = n_functions
        self.peak_factor = peak_factor
        self.win_duty = win_duty
        super().__init__(modfs=modfs, **kwargs)

    def set_peak_factor(self):
        if self.peak_factor is None:
            if self.n_functions == 3:
                self.peak_factor = 6.
            elif self.n_functions == 4:
                self.peak_factor = 12.
            elif self.n_functions == 5:
                self.peak_factor = 30.
            else:
                assert False
            return

    def generate_source(self):
        if self.peak_factor is not None:
            assert self.win_duty is not None, 'IRF Window is None when doing constrained Codes'
            return get_constrained_ham_codes(self.n_functions, self.peak_factor, self.win_duty, self.n_tbins)[0]
        self.set_peak_factor()
        modfs = np.zeros((self.n_tbins, self.n_functions))
        mod_duty = 1. / self.peak_factor
        for i in range(0, self.n_functions):
            modfs[0:math.floor(mod_duty * self.n_tbins), i] = self.peak_factor * AveragePowerDefault
        return modfs


class GaussianTIRF(SinglePhotonSource):

    def __init__(self, peak_factor=None, mu=None, sigma=None, **kwargs):
        # Set mu, and sigma params
        (self.mu, self.sigma) = (0, 1.)
        self.peak_factor = peak_factor
        if (not (mu is None)): self.mu = mu
        if (not (sigma is None)): self.sigma = sigma
        # Initialize the regular Temporal IRF
        super().__init__(**kwargs)

    def generate_source(self):
        # Create a circular gaussian pulse
        gaussian_tirf = gaussian_pulse(self.t_domain, self.mu, self.sigma, circ_shifted=True)
        return gaussian_tirf

    def simulate_photons(self):
        if self.peak_factor is None:
            if self.binomial:
                v_out = self.simulate_average_photons_single_cycle()
            else:
                v_out = self.simulate_average_photons_n_cycles()
        else:
            if self.binomial:
                v_out = self.simulate_peak_photons_single_cycle()
            else:
                v_out = self.simulate_peak_photons_n_cycles()

        return v_out


    def simulate_average_photons_n_cycles(self, inplace=False):
        v_out = np.copy(self.light_source)
        total_source_photons = self.ave_source * self.t
        total_amb_photons = total_source_photons / self.sbr
        v_out *= (total_source_photons / np.sum(v_out))
        v_out = v_out + (total_amb_photons / self.n_tbins)
        return v_out

    def simulate_peak_photons_n_cycles(self):
        v_out = np.copy(self.light_source)
        total_source_photons = self.ave_source * self.t
        total_amb_photons = total_source_photons / self.sbr
        peak_photons = total_source_photons * total_source_photons
        v_out = (v_out / v_out.max(axis=-1, keepdims=True)) * peak_photons
        v_out = v_out + (total_amb_photons / self.n_tbins)
        return v_out

    def simulate_average_photons_single_cycle(self, inplace=False):
        laser_cycles = (1. / self.tau) * self.t
        v_out = self.simulate_average_photons_n_cycles()
        v_out = v_out / laser_cycles
        return v_out

    def simulate_peak_photons_single_cycle(self):
        laser_cycles = (1. / self.tau) * self.t
        v_out = self.simulate_peak_photons_n_cycles()
        v_out = v_out / laser_cycles
        return v_out
