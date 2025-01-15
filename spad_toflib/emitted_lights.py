from abc import ABC, abstractmethod
## Library Imports
from IPython.core import debugger

breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse, smooth_codes
from felipe_utils.tof_utils_felipe import *
import math
import scipy as sp

TotalEnergyDefault = 1.
TauDefault = 1.
AveragePowerDefault = TotalEnergyDefault / TauDefault


class LightSource(ABC):

    def __init__(self, light_source, depths, t, rep_tau=None, num_measures=None):
        self.set_light_source(light_source)
        self.depths = depths
        self.set_tau(rep_tau)
        self.set_num_measures(num_measures)
        self.set_integration_time(t)
        # Find tirf pixels with no signal

    def set_ambient(self, input_sbr, input_amb, dt=1, tau=1):
        if (input_amb is not None):
            self.ambient = input_amb
        elif (input_sbr is not None):
            self.sbr = input_sbr
        else:
            self.ambient = None
            self.sbr = None

    def set_light_source(self, input_light_source):
        self.light_source = input_light_source
    def set_ave_source(self, input_source):
        if (input_source is None):
            self.ave_source = None
            return
        self.ave_source = np_utils.to_nparray(input_source)

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

    def set_integration_time(self, input_t):
        if (input_t is None):
            self.t = None
            return
        self.t = input_t


class SinglePhotonSource(LightSource):

    def __init__(self, n_tbins, n_functions, light_source=None, split=False, binomial=False
                 , t_domain=None, **kwargs):
        self.n_tbins = n_tbins
        self.binomial = binomial
        self.split = split
        self.set_t_domain(t_domain)
        self.n_functions = n_functions
        if light_source is None:
            light_source = self.generate_source()
        else:
             light_source = self.set_source(light_source)
        super().__init__(light_source=light_source, **kwargs)

    def set_source(self, input_light_source):
        x = np.arange(input_light_source.size)
        new_length = self.n_tbins
        new_x = np.linspace(x.min(), x.max(), new_length)
        new_y = sp.interpolate.interp1d(x, input_light_source, kind='cubic')(new_x)
        output_source = np.repeat(np.expand_dims(new_y, axis=-1), self.n_functions, axis=-1)
        return output_source

    def set_t_domain(self, t_domain):
        if t_domain is None:
            self.t_domain = np.arange(0, self.n_tbins)
        else:
            assert (t_domain.shape[-1] == self.n_tbins), "Input t_domain need to match tirf_data last dim"
            self.t_domain = t_domain

    @abstractmethod
    def generate_source(self):
        pass

    def simulate_peak_photons(self, peak_photons, ambient_photons):
        if self.binomial:
            laser_cycles = (1. / self.tau) * self.t
            v_out = self.simulate_peak_photons_n_cycles(peak_photons, ambient_photons)
            v_out = v_out / laser_cycles
        else:
            v_out = self.simulate_peak_photons_n_cycles(peak_photons, ambient_photons)
        v_out[v_out < 0] = 0
        return self.phase_shifted(v_out)

    def simulate_average_photons(self, total_photons, sbr):
        if self.binomial:
            laser_cycles = (1. / self.tau) * self.t
            v_out = self.simulate_average_photons_n_cycles(total_photons, sbr) / laser_cycles
        else:
            v_out = self.simulate_average_photons_n_cycles(total_photons, sbr)
        v_out[v_out < 0] = 0
        return self.phase_shifted(v_out)

    def simulate_average_photons_n_cycles(self, total_photons, sbr):
        incident = np.zeros(self.light_source.shape)

        if self.split: total_photons = total_photons / self.n_functions

        total_amb_photons = total_photons / sbr
        scaled_modfs = np.copy(self.light_source)
        for i in range(0, self.light_source.shape[-1]):
            scaled_modfs[:, i] *= (total_photons / np.sum(self.light_source[:, i]))
            incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / self.n_tbins))
        return incident

    def simulate_peak_photons_n_cycles(self, peak_photons, ambient_photons):
        incident = np.zeros(self.light_source.shape)
        peak_modfs = np.copy(self.light_source)

        if self.split:
            peak_photons = peak_photons / self.n_functions
            ambient_photons = ambient_photons / self.n_functions

        for k in range(self.light_source.shape[-1]):
            peak_modfs[:, k] = (peak_modfs[:, k] / peak_modfs[:, k].max()) * peak_photons
            incident[:, k] = peak_modfs[:, k] + ambient_photons
        return incident

    def phase_shifted(self, modfs):
        shifted_modfs = np.zeros((self.depths.shape[0], modfs.shape[1], modfs.shape[0]))
        tbin_depth_res = time2depth(self.tau / modfs.shape[0])
        for d in range(0, self.depths.shape[0]):
            for i in range(0, modfs.shape[-1]):
                shifted_modfs[d, i, :] = np.roll(modfs[:, i], int(self.depths[d] / tbin_depth_res))
        return shifted_modfs


class LearnedSource(SinglePhotonSource):
    def __init__(self, filename, win_duty=None, **kwargs):
        self.filename = filename
        self.win_duty = win_duty
        super().__init__(light_source=None, n_functions=None, **kwargs)

    def generate_source(self):
        light_source = np.load(self.filename)
        if self.win_duty is not None:
            dummy_var = np.zeros((self.n_tbins, 1))
            (light_source, _) = smooth_codes(light_source, dummy_var, window_duty=self.win_duty)
        self.n_functions = light_source.shape[-1]
        return light_source

class KTapSinusoidSource(SinglePhotonSource):
    def __init__(self, n_functions, modfs=None, **kwargs):
        if n_functions is None: n_functions = 3
        super().__init__(light_source=modfs, n_functions=n_functions, **kwargs)

    def generate_source(self):
        modfs = np.zeros((self.n_tbins, self.n_functions))
        t = np.linspace(0, 2 * np.pi, self.n_tbins)
        cosfs = (0.5 * np.cos(t)) + 0.5
        for i in range(0, self.n_functions):
            modfs[:, i] = cosfs
        return modfs


class HamiltonianSource(SinglePhotonSource):
    def __init__(self, n_functions, modfs=None, duty=None, win_duty=None, **kwargs):
        self.win_duty = win_duty
        self.set_duty(duty, n_functions)
        super().__init__(n_functions=n_functions, light_source=modfs,  **kwargs)

    def set_duty(self, duty, n_functions):
        if duty is None:
            if n_functions == 3:
                self.duty = 1. / 6.
            elif n_functions == 4:
                self.duty = 1. / 12.
            elif n_functions == 5:
                self.duty = 1. / 30.
            else:
                assert False
        else:
            self.duty = duty

    def generate_source(self):
        modfs = np.zeros((self.n_tbins, self.n_functions))
        dummy_var = np.zeros((self.n_tbins, self.n_functions))
        for i in range(0, self.n_functions):
            modfs[0:math.floor(self.duty * self.n_tbins), i] = AveragePowerDefault

        if self.win_duty is not None:
            (modfs, _) = smooth_codes(modfs, dummy_var, window_duty=self.win_duty)

        return modfs


class GaussianTIRF(SinglePhotonSource):

    def __init__(self, tirf=None, mu=None, sigma=None, win_duty=None, **kwargs):
        # Set mu, and sigma params
        (self.mu, self.sigma) = (0, 1.)
        if (not (mu is None)): self.mu = mu
        if (not (sigma is None)): self.sigma = sigma
        self.win_duty = win_duty
        # Initialize the regular Temporal IRF
        super().__init__(n_functions=1, light_source=tirf, **kwargs)

    def generate_source(self):
        # Create a circular gaussian pulse
        gaussian_tirf = gaussian_pulse(self.t_domain, 0, self.sigma, circ_shifted=True)
        gaussian_tirf = np.expand_dims(gaussian_tirf, axis=-1)
        if self.win_duty is not None:
             dummy_var = np.zeros((self.n_tbins, self.n_functions))
             (gaussian_tirf, _) = smooth_codes(gaussian_tirf, dummy_var, window_duty=self.win_duty)
        return gaussian_tirf
