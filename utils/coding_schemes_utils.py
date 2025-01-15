from IPython.core import debugger

breakpoint = debugger.set_trace
from spad_toflib.coding_schemes import *
from spad_toflib.emitted_lights import *
from dataclasses import dataclass
import numpy as np

learned_folder = 'learned_codes'

def init_coding_list(n_tbins, depths, params, t_domain=None):
    coding_list = []
    imaging_schemes = params['imaging_schemes']
    tbin_res = params['rep_tau'] / n_tbins
    t = params['T']
    tau = params['rep_tau']
    for i in range(len(imaging_schemes)):
        current_coding_scheme = imaging_schemes[i]
        current_coding_scheme.coding_obj = create_coding_obj(current_coding_scheme, n_tbins, tbin_res,
                                                             t_domain=t_domain)
        current_coding_scheme.light_obj = create_light_obj(current_coding_scheme, n_tbins, tbin_res, depths,
                                                           t, tau, t_domain=t_domain)


def create_light_obj(coding_system, n_tbins, tbin_res, depths, t, tau, t_domain=None):
    light_obj = None
    light_id = coding_system.light_id

    n_functions = coding_system.ktaps
    pw = coding_system.pulse_width
    binomial = coding_system.binomial
    win_duty = coding_system.freq_window
    h_irf = coding_system.h_irf
    split = coding_system.split
    duty = coding_system.duty

    if light_id == 'KTapSinusoid':
        light_obj = KTapSinusoidSource(modfs=h_irf, n_functions=n_functions, split=split,
                                       t=t, rep_tau=tau, binomial=binomial, n_tbins=n_tbins, depths=depths)
    elif light_id == 'HamiltonianK3':
        light_obj = HamiltonianSource(modfs=h_irf, n_functions=3, duty=duty, split=split, binomial=binomial,
                                      t=t, rep_tau=tau, win_duty=win_duty, n_tbins=n_tbins, depths=depths)
    elif light_id == 'HamiltonianK4':
        light_obj = HamiltonianSource(modfs=h_irf, n_functions=4, duty=duty, split=split, binomial=binomial,
                                      t=t, rep_tau=tau, win_duty=win_duty, n_tbins=n_tbins, depths=depths)
    elif light_id == 'HamiltonianK5':
        light_obj = HamiltonianSource(modfs=h_irf, n_functions=5, duty=duty, split=split, binomial=binomial,
                                      t=t, rep_tau=tau, win_duty=win_duty, n_tbins=n_tbins, depths=depths)
    elif light_id == 'Gaussian':
        if pw is None: pw = 1
        light_obj = GaussianTIRF(tirf=h_irf, n_tbins=n_tbins, binomial=binomial, sigma=pw * tbin_res,
                                 t=t, rep_tau=tau, depths=depths, t_domain=t_domain, win_duty=win_duty)

    elif light_id == 'Learned':
        filename = os.path.join(os.path.join(learned_folder, 'illumination'), coding_system.checkpoint_file)
        light_obj = LearnedSource(filename=filename, split=split, binomial=binomial,
                                      t=t, rep_tau=tau, win_duty=win_duty, n_tbins=n_tbins, depths=depths)

    return light_obj


def create_coding_obj(coding_system, n_tbins, tbin_res, t_domain=None):
    coding_obj = None
    coding_id = coding_system.coding_id
    ktaps = coding_system.ktaps
    n_gates = coding_system.n_gates
    n_bits = coding_system.n_bits
    laser_cycles = coding_system.total_laser_cycles
    win_duty = coding_system.freq_window
    binomial = coding_system.binomial
    gated = coding_system.gated
    account_irf = coding_system.account_irf
    h_irf = coding_system.h_irf
    pw = coding_system.pulse_width
    cw_tof = coding_system.cw_tof
    freq_idx = coding_system.freq_idx
    n_freqs = coding_system.n_freqs
    n_codes = coding_system.n_codes

    if gated is True:
        split = True
    else:
        split = coding_system.split
    if pw is None: pw = 1
    duty = coding_system.duty
    if (coding_id == 'KTapSinusoid'):
        coding_obj = KTapSinusoidCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial,
                                        ktaps=ktaps, split=split, after=cw_tof, account_irf=account_irf, t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK3'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial,k=3,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK4'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial, k=4,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK5'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial, k=5,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'ModifiedHamiltonianK3'):
        coding_obj = ModifiedHamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial,k=3,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'ModifiedHamiltonianK4'):
        coding_obj = ModifiedHamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial, k=4,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'ModifiedHamiltonianK5'):
        coding_obj = ModifiedHamiltonianCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, gated=gated, binomial=binomial, k=5,
                                                split=split, duty=duty, win_duty=win_duty, account_irf=account_irf,
                                                t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'Identity'):
        coding_obj = IdentityCoding(n_tbins=n_tbins, sigma=pw * tbin_res, gated=gated, binomial=binomial, total_laser_cycles=laser_cycles, account_irf=account_irf,
                                            t_domain=t_domain, h_irf=h_irf, win_duty=win_duty)
    elif (coding_id == 'Gated'):
        assert n_gates != None, 'Need to declare number of gates for gated coding'
        coding_obj = GatedCoding(n_tbins=n_tbins, sigma=pw * tbin_res, binomial=binomial, gated=gated, total_laser_cycles=laser_cycles, n_gates=n_gates,
                                         account_irf=False, t_domain=t_domain, h_irf=h_irf, win_duty=win_duty)
    elif (coding_id == 'Greys'):
        assert n_bits != None, 'Need to declare number of bits for greys coding'
        coding_obj = GrayCoding(n_tbins=n_tbins, sigma=pw * tbin_res, binomial=binomial, gated=gated, total_laser_cycles=laser_cycles, n_bits=n_bits,
                                         account_irf=account_irf, t_domain=t_domain, h_irf=h_irf, win_duty=win_duty)
    elif (coding_id == 'Fourier'):
        coding_obj = FourierCoding(n_tbins=n_tbins, sigma=pw * tbin_res, binomial=binomial, gated=gated,
                                total_laser_cycles=laser_cycles, freq_idx=freq_idx, n_codes=n_codes,
                                account_irf=account_irf, t_domain=t_domain, h_irf=h_irf, win_duty=win_duty)

    elif (coding_id == 'TruncatedFourier'):
        coding_obj = TruncatedFourierCoding(n_tbins=n_tbins, sigma=pw * tbin_res, binomial=binomial, gated=gated,
                                total_laser_cycles=laser_cycles, n_freqs=n_freqs, n_codes=n_codes,
                                account_irf=account_irf, t_domain=t_domain, h_irf=h_irf, win_duty=win_duty)

    elif (coding_id == 'GrayTruncatedFourier'):
        coding_obj = GrayTruncatedFourierCoding(n_tbins=n_tbins, sigma=pw * tbin_res, binomial=binomial, gated=gated,
                                total_laser_cycles=laser_cycles, n_codes=n_codes,
                                account_irf=account_irf, t_domain=t_domain, h_irf=h_irf)
    elif (coding_id == 'Learned'):
        filename = os.path.join(os.path.join(learned_folder, 'coding_matrices'), coding_system.checkpoint_file)
        coding_obj = LearnedCoding(n_tbins=n_tbins, sigma=pw * tbin_res, checkpoints=filename,
                                   binomial=binomial, gated=gated, total_laser_cycles=laser_cycles,
                                   account_irf=False, t_domain=t_domain, h_irf=h_irf,
                                   win_duty=win_duty)
    return coding_obj


def get_levels_list_montecarlo(params):
    level_one = params['levels_one']
    level_one_exp = params['levels_one_exp']
    level_two = params['levels_two']
    level_two_exp = params['levels_two_exp']
    n_level_one = params['num_levels_one']
    n_level_two = params['num_levels_two']
    levels_one = None
    levels_two = None

    if level_one in ['laser cycles', 'ave power']:
        levels_one = np.round(np.power(10, np.linspace(level_one_exp[0], level_one_exp[1], n_level_one)))
    elif level_one in ['integration time', 'sbr']:
        levels_one = np.power(10, np.linspace(level_one_exp[0], level_one_exp[1], n_level_one))
    elif level_one in ['peak power', 'amb photons']:
        levels_one = np.linspace(level_one_exp[0], level_one_exp[1], n_level_one)
    else:
        assert False, f'MonteCarlo exp not implemented for {level_one} parameter'

    if level_two in ['laser cycles', 'ave power']:
        levels_two = np.round(np.power(10, np.linspace(level_two_exp[0], level_two_exp[1], n_level_two)))
    elif level_two in ['integration time', 'sbr']:
        levels_two = np.power(10, np.linspace(level_two_exp[0], level_two_exp[1], n_level_two))
    elif level_two in ['peak power', 'amb photons']:
        levels_two = np.linspace(level_two_exp[0], level_two_exp[1], n_level_two)
    else:
        assert False, f'MonteCarlo exp not implemented for {level_two} parameter'

    return np.meshgrid(levels_one, levels_two)


@dataclass
class ImagingSystemParams:
    coding_id: str
    light_id: str
    rec_algo: str
    coding_obj: Coding = None
    light_obj: LightSource = None
    h_irf: np.array = None
    account_irf: bool = False
    binomial: bool = False
    cw_tof: bool = False
    gated: bool = False
    split: bool = False
    pulse_width: int = None
    duty: float = None
    freq_window: float = None
    ktaps: int = None
    n_gates: int = None
    n_bits: int = None
    freq_idx: list = None
    n_freqs: int = None
    n_codes: int = None
    total_laser_cycles: int = None
    mean_absolute_error: float = None
    checkpoint_file: str = None
