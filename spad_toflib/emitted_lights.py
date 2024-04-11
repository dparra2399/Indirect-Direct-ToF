from abc import ABC, abstractmethod
## Library Imports
from IPython.core import debugger

breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from spad_toflib.spad_tof_utils import calculate_ambient
from felipe_utils.felipe_cw_utils.CodingFunctionUtilsFelipe import ScaleMod
from felipe_utils.felipe_impulse_utils.tof_utils_felipe import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

mpl.use('qt5agg')

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
        assert input_sbr is None or input_amb is None
        if (input_amb is not None):
            avg_amb = input_amb
        elif (input_sbr is not None):
            if self.ave_source is None:
                self.ambient = None
                return
            avg_amb = self.ave_source / input_sbr
        else:
            self.ambient = None
            return
        self.ambient = calculate_ambient(self.n_tbins, avg_amb, dt, tau)

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
        self.set_ambient(sbr, ambient)
        self.set_mean_beta(mean_beta)
        self.set_integration_time(t)
        self.set_num_measures(num_measures)
        self.set_tau(rep_tau)



class SinglePhotonSource(LightSource):

    def __init__(self, n_tbins, light_source=None, t_domain=None, **kwargs):
        self.n_tbins = n_tbins
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
        incident = np.zeros(self.light_source.shape)
        scaled_modfs = ScaleMod(self.light_source, tau=self.tau, pAveSource=self.ave_source)
        for i in range(0, self.light_source.shape[-1]):
            incident[:, i] = (self.t * self.mean_beta * (scaled_modfs[:, i] + self.ambient)) / self.num_measures
        return self.phase_shifted(incident)


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
    def __init__(self, n_functions, modfs=None, peak_factor=None, **kwargs):
        self.n_functions = n_functions
        self.set_peak_factor(peak_factor)
        super().__init__(modfs=modfs, **kwargs)

    def set_peak_factor(self, input_factor):
        if input_factor is None:
            if self.n_functions == 3:
                self.peak_factor = 6.
            elif self.n_functions == 4:
                self.peak_factor = 12.
            elif self.n_functions == 5:
                self.peak_factor = 30.
            else:
                assert False
            return
        self.peak_factor = input_factor

    def generate_source(self):
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
            v_out = self.simulate_average_photons()
        else:
            v_out = self.simulate_peak_photons()

        return v_out

    def set_average_area(self, v_in, inplace=False):
        if (not inplace):
            v_out = np.array(v_in)
        else:
            v_out = v_in

        dt = self.tau / self.n_tbins
        if (self.ave_source is not None):
            desired_area = self.ave_source * self.tau
            for i in range(0, self.depths.shape[0]):
                oldArea = np.sum(v_out[i, :]) * dt
                v_out[i, :] = v_out[i, :] * desired_area / oldArea
        return v_out

    def set_peak_power(self, v_in, inplace=False):
        if (not inplace):
            v_out = np.array(v_in)
        else:
            v_out = v_in
        if not (self.peak_factor is None):
            v_out = (v_out / v_out.max(axis=-1, keepdims=True)) * (self.peak_factor * self.ave_source)
        return v_out

    def simulate_average_photons(self, inplace=False):
        v_out = self.set_average_area(self.light_source, inplace)
        v_out = (v_out + self.ambient) * self.mean_beta * self.t
        v_out *= self.num_measures
        return v_out

    def simulate_peak_photons(self):
        v_out = self.set_peak_power(self.light_source)
        v_out = (v_out + self.ambient) * self.mean_beta * self.t
        v_out *= self.num_measures
        return v_out


class KTapSinusoidSWISSSPADSource(KTapSinusoidSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def simulate_photons(self):
        laser_cycles = (1. / self.tau) * self.t
        incident = np.zeros(self.light_source.shape)
        scaled_modfs = ScaleMod(self.light_source, tau=self.tau, pAveSource=self.ave_source)
        for i in range(0, self.light_source.shape[-1]):
            incident[:, i] = (self.mean_beta * (scaled_modfs[:, i] + self.ambient)) / laser_cycles
        return self.phase_shifted(incident)


class HamiltonianSWISSSPADSource(HamiltonianSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def simulate_photons(self):
        laser_cycles = (1. / self.tau) * self.t
        incident = np.zeros(self.light_source.shape)
        scaled_modfs = ScaleMod(self.light_source, tau=self.tau, pAveSource=self.ave_source)
        for i in range(0, self.light_source.shape[-1]):
            incident[:, i] = (self.mean_beta * (scaled_modfs[:, i] + self.ambient)) / laser_cycles
        return self.phase_shifted(incident)


class GaussianSWISSPADTIRF(GaussianTIRF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def simulate_average_photons(self, inplace=False):
        laser_cycles = (1. / self.tau) * self.t
        v_out = self.set_average_area(self.light_source, inplace)
        v_out = ((v_out + self.ambient) * self.mean_beta) / laser_cycles
        return v_out

    def simulate_peak_photons(self):
        laser_cycles = (1. / self.tau) * self.t
        v_out = self.set_peak_power(self.light_source)
        v_out = ((v_out + self.ambient) * self.mean_beta) / laser_cycles
        return v_out
