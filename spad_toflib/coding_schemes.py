from abc import ABC, abstractmethod

from scipy import interpolate

from felipe_utils.felipe_cw_utils import CodingFunctionsFelipe
from felipe_utils.felipe_impulse_utils.tof_utils_felipe import zero_norm_t
from felipe_utils.research_utils import signalproc_ops, np_utils
from spad_toflib.emitted_lights import GaussianTIRF
from spad_toflib.spad_tof_utils import *
from utils.file_utils import get_constrained_ham_codes
import matplotlib.pyplot as plt


class Coding(ABC):

    def __init__(self, total_laser_cycles=None, binomial=False, num_measures=None, h_irf=None, account_irf=False):

        self.binomial = binomial
        self.set_laser_cycles(total_laser_cycles)
        if self.correlations is None: self.set_coding_scheme()
        (self.n_tbins, self.n_functions) = (self.correlations.shape[-2], self.correlations.shape[-1])
        self.set_num_measures(num_measures)
        self.update_irf(h_irf)
        self.account_irf = account_irf

    @abstractmethod
    def set_coding_scheme(self):
        pass

    ''' Felipe's Code'''

    def zncc_reconstruction(self, intensities):
        norm_int = normalize_measure_vals(intensities)
        return np.matmul(self.norm_corrfs, norm_int[..., np.newaxis]).squeeze(-1)

    def update_irf(self, h_irf=None):
        # If nothing is given set to gaussian
        if (h_irf is None):
            # print("hirf is NONE")
            self.h_irf = GaussianTIRF(n_tbins=self.n_tbins, mu=0, sigma=1, depths=None).light_source.squeeze()
        else:
            self.h_irf = h_irf.squeeze()
        self.h_irf = self.h_irf / self.h_irf.sum()

    ''' Felipe's Code'''

    def get_rec_algo_func(self, rec_algo_id):
        rec_algo_func_name = '{}_reconstruction'.format(rec_algo_id)
        rec_algo_function = getattr(self, rec_algo_func_name, None)
        assert (
                rec_algo_function is not None), "Reconstruction algorithm {} is NOT available. Please choose from the following algos: {}".format(
            rec_algo_func_name, self.rec_algos_avail)
        return rec_algo_function

    ''' Felipe's Code'''

    def reconstruction(self, intensities, rec_algo_id='zncc', **kwargs):
        rec_algo_function = self.get_rec_algo_func(rec_algo_id)
        lookup = rec_algo_function(intensities, **kwargs)
        return lookup

    ''' Felipe's Code'''

    def max_peak_decoding(self, intensities, rec_algo_id='zncc', **kwargs):
        lookup = self.reconstruction(intensities, rec_algo_id, **kwargs)
        return np.argmax(lookup, axis=-1)

    ''' Felipe's Code'''

    def maxgauss_peak_decoding(self, intensities, gauss_sigma, rec_algo_id='zncc', **kwargs):
        lookup = self.reconstruction(intensities, rec_algo_id, **kwargs)
        return signalproc_ops.max_gaussian_center_of_mass_mle(lookup, sigma_tbins=gauss_sigma)

    def set_correlations(self, modfs, demodfs):
        self.correlations = np.fft.ifft(np.fft.fft(modfs, axis=0).conj() * np.fft.fft(demodfs, axis=0), axis=0).real
        self.norm_corrfs = normalize_measure_vals(self.correlations, axis=1)

    def encode_cw(self, incident, trials, after=False):
        if self.binomial:
            return self.encode_cw_bin(incident, trials)
        else:
            return self.encode_cw_pois(incident, trials, after)

    def encode_cw_pois(self, incident, trials, after):
        if not after: incident = poisson_noise_array(incident, trials)
        intensities = np.matmul(incident, self.demodfs)[..., 0, :]
        if after: intensities = poisson_noise_array(intensities, trials)
        return intensities

    def encode_cw_bin(self, incident, trials):
        photons = np.matmul(incident, self.demodfs)[..., 0, :]
        probabilities = 1 - np.exp(-photons)
        rng = np.random.default_rng()
        new_shape = (trials,) + probabilities.shape
        photon_counts = rng.binomial(int(self.laser_cycles / self.n_functions), probabilities, size=new_shape)
        return photon_counts

    ''' Felipe's Code'''

    def encode_impulse(self, transient_img, trials):
        if self.binomial:
            return self.encode_cw_bin(transient_img, trials)
        else:
            return self.encode_impulse_pois(transient_img, trials)

    def encode_impulse_pois(self, transient_img, trials):
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        transient_img = poisson_noise_array(transient_img, trials)
        return np.matmul(transient_img[..., np.newaxis, :], self.correlations).squeeze(-2)

    def encode_impulse_bin(self, transient_img, trials):
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        transient_img = poisson_noise_array(transient_img, trials)
        photons = np.matmul(transient_img[..., np.newaxis, :], self.correlations).squeeze(-2)
        probabilities = 1 - np.exp(-photons)
        rng = np.random.default_rng()
        new_shape = (trials,) + probabilities.shape
        photon_counts = rng.binomial(int(self.laser_cycles / self.n_tbins), probabilities, size=new_shape)

        return photon_counts

    def set_laser_cycles(self, input_cycles):
        if input_cycles is None:
            self.laser_cycles = None
            return
        assert self.binomial is True, 'To set laser cycles must be binomial poisson model'
        self.laser_cycles = input_cycles

    def set_num_measures(self, input_num):
        if input_num is None:
            self.num_measures = self.n_functions
            return
        self.num_measures = input_num

    def get_num_measures(self):
        return self.num_measures

class KTapSinusoidCoding(Coding):

    def __init__(self, n_tbins, ktaps, **kwargs):
        if (ktaps is None): ktaps = 3
        self.n_functions = ktaps
        self.set_coding_scheme(n_tbins, ktaps)
        super().__init__(**kwargs)

    def set_coding_scheme(self, n_tbins, ktaps):
        (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetCosCos(N=n_tbins, K=ktaps)
        self.set_correlations(self.modfs, self.demodfs)


class HamiltonianCoding(Coding):
    def __init__(self, n_tbins, k, peak_factor=None, win_duty=None, **kwargs):
        self.n_functions = k
        self.peak_factor = peak_factor
        self.win_duty = win_duty
        self.set_coding_scheme(n_tbins, k, peak_factor)
        super().__init__(**kwargs)

    def set_num_measures(self, input_num):
        if input_num is None and self.binomial:
            if self.n_functions == 3:
                self.num_measures = 4
            elif self.n_functions == 4:
                self.num_measures = 7
            elif self.n_functions == 5:
                self.num_measures = 16
            else:
                assert False, 'not implemented for k>5'
            return
        elif input_num is None:
            self.num_measures = self.n_functions
        else:
            self.num_measures = input_num

    def set_coding_scheme(self, n_tbins, k, peak_factor):
        if peak_factor is not None:
            assert self.win_duty is not None, 'IRF Window is None when doing constrained Codes'
            (self.modfs, self.demodfs) = get_constrained_ham_codes(k, self.peak_factor, self.win_duty, n_tbins)
        else:
            if (k == 3):
                (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK3(n_tbins)
            elif (k == 4):
                (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK4(n_tbins)
            elif (k == 5):
                (self.modfs, self.demodfs) = CodingFunctionsFelipe.GetHamK5(n_tbins)
            else:
                assert False
        self.set_correlations(self.modfs, self.demodfs)


''' Felipe's Code'''


class GatedCoding(Coding):
    '''
        Gated coding class. Coding is applied like a gated camera or a coarse histogram in SPADs
        In the extreme case that we have a gate for every time bin then the C matrix is an (n_maxres x n_maxres) identity matrix
    '''

    def __init__(self, n_tbins, n_gates=None, **kwargs):
        if (n_gates is None): n_gates = n_tbins
        assert ((n_tbins % n_gates) == 0), "Right now GatedCoding required n_maxres to be divisible by n_gates"
        assert ((n_tbins >= n_gates)), "n_gates should always be smaller than n_maxres"
        self.n_gates = n_gates
        self.set_coding_scheme(n_tbins, n_gates)
        super().__init__(**kwargs)

    def set_coding_scheme(self, n_tbins, n_gates):
        self.gate_len = int(n_tbins / n_gates)
        self.correlations = np.zeros((n_tbins, n_gates))
        for i in range(n_gates):
            start_tbin = i * self.gate_len
            end_tbin = start_tbin + self.gate_len
            self.correlations[start_tbin:end_tbin, i] = 1.

    def encode_impulse(self, transient_img, trials):
        if self.binomial:
            return self.encode_impulse_bin(transient_img, trials)
        else:
            return self.encode_impulse_pois(transient_img, trials)

    def encode_impulse_pois(self, transient_img, trials):
        '''
        Encode the transient image using the n_codes inside the self.C matrix
        For GatedCoding with many n_gates, encoding through matmul is quite slow, so we assign it differently
        '''
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        transient_img = poisson_noise_array(transient_img, trials)
        c_vals = np.array(transient_img[..., 0::self.gate_len])
        for i in range(self.gate_len - 1):
            start_idx = i + 1
            c_vals += transient_img[..., start_idx::self.gate_len]
        return c_vals

    ''' MY CODE '''

    def encode_impulse_bin(self, transient_img, trials):
        assert (transient_img.shape[-1] == self.n_tbins), "Input c_vec does not have the correct dimensions"
        photons = np.array(transient_img[..., 0::self.gate_len])
        for i in range(self.gate_len - 1):
            start_idx = i + 1
            photons += transient_img[..., start_idx::self.gate_len]

        probabilities = 1 - np.exp(-photons)
        rng = np.random.default_rng()
        new_shape = (trials,) + probabilities.shape
        photon_counts = rng.binomial(int(self.laser_cycles / self.n_tbins), probabilities, size=new_shape)

        return photon_counts

    def matchfilt_reconstruction(self, c_vals):
        template = self.h_irf
        zn_template = zero_norm_t(template, axis=-1)
        zn_c_vals = zero_norm_t(c_vals, axis=-1)
        shifts = signalproc_ops.circular_matched_filter(zn_c_vals, zn_template)
        # vectorize tensors
        (c_vals, c_vals_shape) = np_utils.vectorize_tensor(c_vals, axis=-1)
        shifts = shifts.reshape((c_vals.shape[0],))
        h_rec = np.zeros(c_vals.shape, dtype=template.dtype)
        for i in range(shifts.size): h_rec[i, :] = np.roll(template, shift=shifts[i], axis=-1)
        c_vals = c_vals.reshape(c_vals_shape)
        return h_rec.reshape(c_vals_shape)

    def linear_reconstruction(self, c_vals):
        if (self.n_gates == self.n_tbins): return c_vals
        if (self.account_irf):
            print(
                "Warning: Linear Reconstruction in Gated does not account for IRF, so unless the IRF spreads across time bins, this will produce quantized depths")
        x_fullres = np.arange(0, self.n_tbins)
        # Create a circular x axis by concatenating the first element to the end and the last element to the begining
        circular_x_lres = np.arange((0.5 * self.gate_len) - 0.5 - self.gate_len, self.n_tbins + self.gate_len,
                                    self.gate_len)
        circular_c_vals = np.concatenate((c_vals[..., -1][..., np.newaxis], c_vals, c_vals[..., 0][..., np.newaxis]),
                                         axis=-1)
        f = interpolate.interp1d(circular_x_lres, circular_c_vals, axis=-1, kind='linear')
        return f(x_fullres)


''' Felipe's Code'''


class IdentityCoding(GatedCoding):
    '''
        Identity coding class. GatedCoding in the extreme case where n_maxres == n_gates
    '''

    def __init__(self, n_tbins, **kwargs):
        super().__init__(n_tbins=n_tbins, **kwargs)
