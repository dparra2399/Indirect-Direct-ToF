from IPython.core import debugger

breakpoint = debugger.set_trace
from spad_toflib.coding_schemes import *
from spad_toflib.emitted_lights import *
from dataclasses import dataclass
import numpy as np


def init_coding_list(n_tbins, depths, params, t_domain=None, pulses_list=None):
    coding_list = []
    imaging_schemes = params['imaging_schemes']
    tbin_res = params['rep_tau'] / n_tbins
    for i in range(len(imaging_schemes)):
        current_coding_scheme = imaging_schemes[i]
        h_irf = None
        if (not (pulses_list is None)): h_irf = pulses_list[i].tirf.squeeze()
        current_coding_scheme.coding_obj = create_coding_obj(current_coding_scheme, n_tbins, h_irf)
        current_coding_scheme.light_obj = create_light_obj(current_coding_scheme, n_tbins, tbin_res, depths, h_irf=None,
                                                           t_domain=t_domain)


def create_light_obj(coding_system, n_tbins, tbin_res, depths, h_irf=None, t_domain=None):
    light_obj = None
    light_id = coding_system.light_id
    n_functions = coding_system.ktaps
    peak_factor = coding_system.peak_factor
    pw = coding_system.pulse_width
    if (light_id == 'KTapSinusoid'):
        light_obj = KTapSinusoidSource(n_functions=n_functions, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'KTapSinusoidSWISSSPAD'):
        light_obj = KTapSinusoidSWISSSPADSource(n_functions=n_functions, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK3'):
        light_obj = HamiltonianSource(n_functions=3, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK3SWISSPAD'):
        light_obj = HamiltonianSWISSSPADSource(n_functions=3, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK4'):
        light_obj = HamiltonianSource(n_functions=4, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK4SWISSPAD'):
        light_obj = HamiltonianSWISSSPADSource(n_functions=4, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK5'):
        light_obj = HamiltonianSource(n_functions=5, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'HamiltonianK5SWISSPAD'):
        light_obj = HamiltonianSWISSSPADSource(n_functions=5, peak_factor=peak_factor, n_tbins=n_tbins, depths=depths)
    elif (light_id == 'Gaussian'):
        light_obj = GaussianTIRF(n_tbins=n_tbins, peak_factor=peak_factor, mu=depth2time(depths), sigma=pw * tbin_res,
                                 depths=depths, t_domain=t_domain)
    elif (light_id == 'GaussianSWISSPAD'):
        light_obj = GaussianSWISSPADTIRF(n_tbins=n_tbins, peak_factor=peak_factor, mu=depth2time(depths),
                                         sigma=pw * tbin_res, depths=depths, t_domain=t_domain)

    return light_obj


def create_coding_obj(coding_system, n_tbins, h_irf=None):
    coding_obj = None
    coding_id = coding_system.coding_id
    ktaps = coding_system.ktaps
    peak_factor = coding_system.peak_factor
    n_gates = coding_system.n_gates
    laser_cycles = coding_system.total_laser_cycles
    if (coding_id == 'KTapSinusoid'):
        coding_obj = KTapSinusoidCoding(n_tbins=n_tbins, ktaps=ktaps, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'KTapSinusoidSWISSSPAD'):
        coding_obj = KTapSinusoidSWISSSPADCoding(total_laser_cycles=laser_cycles, n_tbins=n_tbins, ktaps=ktaps,
                                                 account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK3'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, k=3, peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK3SWISSPAD'):
        coding_obj = HamiltonianSWISSSPADCoding(total_laser_cycles=laser_cycles, n_tbins=n_tbins, k=3,
                                                peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK4'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, k=4, peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK4SWISSPAD'):
        coding_obj = HamiltonianSWISSSPADCoding(total_laser_cycles=laser_cycles, n_tbins=n_tbins, k=4,
                                                peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK5'):
        coding_obj = HamiltonianCoding(n_tbins=n_tbins, k=5, peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'HamiltonianK5SWISSPAD'):
        coding_obj = HamiltonianSWISSSPADCoding(total_laser_cycles=laser_cycles, n_tbins=n_tbins, k=5,
                                                peak_factor=peak_factor, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'Identity'):
        coding_obj = IdentityCoding(n_tbins=n_tbins, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'Gated'):
        coding_obj = GatedCoding(n_tbins=n_tbins, n_gates=n_gates, account_irf=False, h_irf=h_irf)
    elif (coding_id == 'IdentitySWISSPAD'):
        coding_obj = IdentitySWISSPADCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, account_irf=False,
                                            h_irf=h_irf)
    elif (coding_id == 'GatedSWISSPAD'):
        coding_obj = GatedSWISSPADCoding(n_tbins=n_tbins, total_laser_cycles=laser_cycles, n_gates=n_gates,
                                         account_irf=False, h_irf=h_irf)

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
    else:
        assert False, f'MonteCarlo exp not implemented for {level_one} parameter'

    if level_two in ['laser cycles', 'ave power']:
        levels_two = np.round(np.power(10, np.linspace(level_two_exp[0], level_two_exp[1], n_level_two)))
    elif level_two in ['integration time', 'sbr']:
        levels_two = np.power(10, np.linspace(level_two_exp[0], level_two_exp[1], n_level_two))
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
    pulse_width: int = None
    peak_factor: float = None
    ktaps: int = None
    n_gates: int = None
    total_laser_cycles: int = None
    mean_absolute_error: float = None