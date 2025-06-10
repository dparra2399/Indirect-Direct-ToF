from abc import ABC, abstractmethod
## Library Imports
from IPython.core import debugger

breakpoint = debugger.set_trace

## Local Imports
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse, smooth_codes
from felipe_utils.tof_utils_felipe import *
from felipe_utils.research_utils import signalproc_ops, np_utils
import numpy as np
import matplotlib.pyplot as plt

import math
import scipy as sp

TotalEnergyDefault = 1.
TauDefault = 1.
AveragePowerDefault = TotalEnergyDefault / TauDefault

#learned_folder = r'C:\Users\clwalker4\PycharmProjects\Indirect-Direct-ToF\learned_codes'
learned_folder = '/Users/Patron/PycharmProjects/Indirect-Direct-ToF/learned_codes'
class LightSource(ABC):

    def __init__(self, light_source, t, h_irf=None, rep_tau=None, num_measures=None):
        self.light_source = light_source
        self.set_tau(rep_tau)
        self.set_num_measures(num_measures)
        self.set_integration_time(t)
        self.update_irf(h_irf)
        self.update_illum()
        # Find tirf pixels with no signal

    def update_irf(self, h_irf=None):
        if h_irf is None:
            self.h_irf = gaussian_pulse(np.arange(0, self.light_source.shape[0]), 0, 1, circ_shifted=True)
        elif h_irf.shape[0] == self.light_source.shape[0]:
            self.h_irf = h_irf.squeeze()
            self.h_irf = np.roll(self.h_irf, -np.argmax(self.h_irf))
            self.h_irf = self.h_irf / self.h_irf.sum()
        else:
            w = h_irf
            x = np.arange(w.size)
            new_length = self.light_source.shape[0]
            new_x = np.linspace(x.min(), x.max(), new_length)
            new_y = sp.interpolate.interp1d(x, w, kind='cubic')(new_x)
            self.h_irf = new_y
            self.h_irf = np.roll(self.h_irf, -np.argmax(self.h_irf))
            self.h_irf = self.h_irf / self.h_irf.sum()

    def update_illum(self):
        if self.h_irf is not None:
            self.filtered_light_source = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], self.light_source, axis=0)
            return
        self.filtered_light_source = self.light_source


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

    def __init__(self, n_tbins, n_functions,split=False, binomial=False, t_domain=None, **kwargs):
        self.n_tbins = n_tbins
        self.binomial = binomial
        self.split = split
        self.set_t_domain(t_domain)
        self.n_functions = n_functions
        light_source = self.generate_source()
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

    def simulate_peak_photons(self, peak_photons, ambient_photons):
        if self.binomial:
            laser_cycles = (1. / self.tau) * self.t
            v_out = self.simulate_peak_photons_n_cycles(peak_photons, ambient_photons)
            v_out = v_out / laser_cycles
        else:
            v_out = self.simulate_peak_photons_n_cycles(peak_photons, ambient_photons)
        v_out[v_out < 0] = 0
        return self.phase_shifted(v_out)

    def simulate_average_photons(self, total_photons, sbr, depths, peak_factor=None):

        if self.binomial:
            laser_cycles = (1. / self.tau) * self.t
            (v_out, tmp_irf) = self.simulate_average_photons_n_cycles(total_photons, sbr, peak_factor=peak_factor) / laser_cycles
        else:
            (v_out, tmp_irf) = self.simulate_average_photons_n_cycles(total_photons, sbr, peak_factor=peak_factor)
        v_out[v_out < 0] = 0
        return self.phase_shifted(v_out, depths), tmp_irf


    def simulate_average_photons_sparse_indirect_reflections(self, total_photons, sbr,
                                                      total_photons_indirect, position, depths, peak_factor=None):
        incident = np.zeros(self.filtered_light_source.shape)
        light_source = self.filtered_light_source
        if self.split: total_photons = total_photons / self.n_functions

        total_amb_photons = total_photons / sbr
        scaled_modfs = np.copy(light_source)
        tmp_irf = np.copy(self.h_irf)
        #tmp_irf = np.copy(self.filtered_light_source.squeeze())
        for i in range(0, light_source.shape[-1]):
            scaled_modfs[:, i] *= (total_photons / np.sum(light_source[:, i]))
            tmp_irf *= (total_photons / np.sum(self.h_irf))
            incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / self.n_tbins))
        if peak_factor is not None:
            #peak_val = np.max(incident)
            incident = np.clip(incident, 0, (peak_factor * total_photons) + (total_amb_photons / self.n_tbins))
            tmp_irf = np.clip(tmp_irf, 0, (peak_factor * total_photons))
            if self.h_irf is not None:
                incident = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], incident, axis=0)
                tmp_irf = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], tmp_irf[:, np.newaxis], axis=0)

        k = np.zeros((self.n_tbins))
        k[position] = 1
        indirect = signalproc_ops.circular_conv(k, self.filtered_light_source[:, 0])
        indirect *= (total_photons_indirect / np.sum(indirect))
        incident += indirect[:, np.newaxis]

        incident[incident < 0] = 0
        return self.phase_shifted(incident, depths), tmp_irf

    def simulate_average_photons_dense_indirect_reflections(self, total_photons, sbr,
                                                      decay, A, depths, peak_factor=None):
        incident = np.zeros(self.filtered_light_source.shape)
        light_source = self.filtered_light_source
        if self.split: total_photons = total_photons / self.n_functions

        total_amb_photons = total_photons / sbr
        scaled_modfs = np.copy(light_source)
        tmp_irf = np.copy(self.h_irf)
        #tmp_irf = np.copy(self.filtered_light_source.squeeze())
        for i in range(0, light_source.shape[-1]):
            scaled_modfs[:, i] *= (total_photons / np.sum(light_source[:, i]))
            tmp_irf *= (total_photons / np.sum(self.h_irf))
            incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / self.n_tbins))
        if peak_factor is not None:
            #peak_val = np.max(incident)
            incident = np.clip(incident, 0, (peak_factor * total_photons) + (total_amb_photons / self.n_tbins))
            tmp_irf = np.clip(tmp_irf, 0, (peak_factor * total_photons))
            if self.h_irf is not None:
                incident = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], incident, axis=0)
                tmp_irf = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], tmp_irf[:, np.newaxis], axis=0)

        k = np.exp(-np.arange(self.n_tbins) / decay)
        k /= k.sum()
        indirect = signalproc_ops.circular_conv(k, self.filtered_light_source[:, 0])
        indirect *= (total_photons / np.sum(indirect))
        incident += A * indirect[:, np.newaxis]
        incident = np.roll(incident, -np.argmax(indirect), axis=-1)


        incident[incident < 0] = 0
        return self.phase_shifted(incident, depths), tmp_irf

    def simulate_average_photons_n_cycles(self, total_photons, sbr, peak_factor=None):
        incident = np.zeros(self.filtered_light_source.shape)

        light_source = self.filtered_light_source
        if self.split: total_photons = total_photons / self.n_functions

        total_amb_photons = total_photons / sbr
        scaled_modfs = np.copy(light_source)
        tmp_irf = np.copy(self.h_irf)
        #tmp_irf = np.copy(self.filtered_light_source.squeeze())
        for i in range(0, light_source.shape[-1]):
            scaled_modfs[:, i] *= (total_photons / np.sum(light_source[:, i]))
            tmp_irf *= (total_photons / np.sum(self.h_irf))
            incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / self.n_tbins))
        if peak_factor is not None:
            #peak_val = np.max(incident)
            incident = np.clip(incident, 0, (peak_factor * total_photons) + (total_amb_photons / self.n_tbins))
            tmp_irf = np.clip(tmp_irf, 0, (peak_factor * total_photons))
            if self.h_irf is not None:
                incident = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], incident, axis=0)
                tmp_irf = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], tmp_irf[:, np.newaxis], axis=0)
        return (incident, tmp_irf)

    def simulate_constant_pulse_energy(self, total_photons, sbr, depths, peak_factor=None):
        incident = np.zeros(self.filtered_light_source.shape)

        light_source = self.filtered_light_source
        total_amb_photons = total_photons / sbr

        if self.split: total_photons = total_photons / self.n_functions

        scaled_modfs = np.copy(light_source)
        for i in range(0, light_source.shape[-1]):
            sigma = 1
            while True:
                irf = gaussian_pulse(np.arange(self.n_tbins), 0, sigma, circ_shifted=True)
                new_light_source = signalproc_ops.circular_conv(irf[:, np.newaxis], light_source[:, i], axis=-1)
                new_light_source *= (total_photons/np.sum(new_light_source))
                if peak_factor is not None:
                    # peak_val = np.max(incident)
                    new_light_source = np.clip(new_light_source, 0, (peak_factor * total_photons))
                    new_light_source = signalproc_ops.circular_conv(irf[:, np.newaxis], new_light_source, axis=0)

                new_total_photons = np.sum(new_light_source)
                if abs(new_total_photons - total_photons) < 5:
                    break
                sigma += 1
            #new_light_source = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], new_light_source, axis=0)
            scaled_modfs[:, i] = new_light_source.squeeze()
            incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / self.n_tbins))
            tmp_irf = irf * (total_photons / np.sum(new_light_source))

        if peak_factor is not None:
            #peak_val = np.max(incident)
            incident = np.clip(incident, 0, (peak_factor * total_photons) + (total_amb_photons / self.n_tbins))
            tmp_irf = np.clip(tmp_irf, 0, (peak_factor * total_photons))
            if self.h_irf is not None:
                incident = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], incident, axis=0)
                tmp_irf = signalproc_ops.circular_conv(self.h_irf[:, np.newaxis], tmp_irf[:, np.newaxis], axis=0)
        incident[incident<0]=0
        return self.phase_shifted(incident, depths), tmp_irf

    def simulate_peak_photons_n_cycles(self, peak_photons, ambient_photons):
        incident = np.zeros(self.filtered_light_source.shape)
        peak_modfs = np.copy(self.filtered_light_source)

        if self.split:
            peak_photons = peak_photons / self.n_functions
            ambient_photons = ambient_photons / self.n_functions

        for k in range(self.filtered_light_source.shape[-1]):
            peak_modfs[:, k] = (peak_modfs[:, k] / peak_modfs[:, k].max()) * peak_photons
            incident[:, k] = peak_modfs[:, k] + ambient_photons
        return incident

    def phase_shifted(self, modfs, depths):
        shifted_modfs = np.zeros((depths.shape[0], modfs.shape[1], modfs.shape[0]))
        tbin_depth_res = time2depth(self.tau / modfs.shape[0])
        for d in range(0, depths.shape[0]):
            for i in range(0, modfs.shape[-1]):
                shifted_modfs[d, i, :] = np.roll(modfs[:, i], int(depths[d] / tbin_depth_res))
        return shifted_modfs


class LearnedSource(SinglePhotonSource):
    def __init__(self, model, n_functions, **kwargs):
        self.model = os.path.join(learned_folder, model, 'illum_model.npy')
        self.n_functions = n_functions
        super().__init__(n_functions=n_functions,  **kwargs)

    def generate_source(self):
        light_source = np.reshape(np.load(self.model), (self.n_tbins, 1))
        #output_source = np.repeat(light_source, self.n_functions, axis=-1)
        return light_source

class KTapSinusoidSource(SinglePhotonSource):
    def __init__(self, n_functions, **kwargs):
        if n_functions is None: n_functions = 3
        super().__init__(n_functions=n_functions, **kwargs)

    def generate_source(self):
        modfs = np.zeros((self.n_tbins, self.n_functions))
        t = np.linspace(0, 2 * np.pi, self.n_tbins)
        cosfs = (0.5 * np.cos(t)) + 0.5
        for i in range(0, self.n_functions):
            modfs[:, i] = cosfs
        return modfs


class HamiltonianSource(SinglePhotonSource):
    def __init__(self, n_functions, duty=None,  **kwargs):
        self.set_duty(duty, n_functions)
        super().__init__(n_functions=n_functions, **kwargs)

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
        for i in range(0, self.n_functions):
            modfs[0:math.floor(self.duty * self.n_tbins), i] = AveragePowerDefault

        return modfs


class GaussianTIRF(SinglePhotonSource):

    def __init__(self, mu=None, sigma=None, **kwargs):
        # Set mu, and sigma params
        (self.mu, self.sigma) = (0, 1.)
        if (not (mu is None)): self.mu = mu
        if (not (sigma is None)): self.sigma = sigma
        # Initialize the regular Temporal IRF
        super().__init__(n_functions=1, **kwargs)

    def generate_source(self):
        # Create a circular gaussian pulse
        gaussian_tirf = gaussian_pulse(self.t_domain, 0, self.sigma, circ_shifted=True)
        gaussian_tirf = np.expand_dims(gaussian_tirf, axis=-1)
        return gaussian_tirf
