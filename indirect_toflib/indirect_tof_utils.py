### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
import matplotlib.pyplot as plt

breakpoint = debugger.set_trace



def GetIncident(ModFs, n_photons, meanBeta, sbr=None):

    (n_tbins, K) = ModFs.shape
    if not(sbr is None):
        n_ambient_photons = n_photons / sbr
        ambient = n_ambient_photons / n_tbins
    else:
        ambient = 0

    ModFs *= n_photons / np.sum(ModFs, axis=0, keepdims=True)
    Incident = meanBeta * (ModFs + ambient)
    return Incident



def ApplyKPhaseShifts(x, shifts):
    K = 0
    if (type(shifts) == np.ndarray):
        K = shifts.size
    elif (type(shifts) == list):
        K = len(shifts)
    else:
        K = 1
    for i in range(0, K):
        x[:, i] = np.roll(x[:, i], int(round(shifts[i])))

    return x

def NormalizeMeasureVals(b_vals, axis=-1):
    """
        b_vals = n x k numpy matrix where each row corresponds to a set of k brightness measurements
    """
    ## Normalized correlation functions, zero mean, unit variance. We have to transpose so that broadcasting works.
    norm_bvals = (b_vals - np.mean(b_vals, axis=axis, keepdims=True)) / np.std(b_vals, axis=axis, keepdims=True)

    return norm_bvals

def AddPoissonNoiseLam(lam, trials=1000):
    new_size = (trials,) + lam.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=lam, size=new_size).astype(lam.dtype)

def GetMeasurements(Incident, DemodFs, dt=1):
    (n_tbins, K) = Incident.shape
    measures = np.zeros(Incident.shape)

    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, n_tbins):
            cc = np.roll(Incident[:, j], l)
            measures[l, j] = np.inner(cc, demod)

    measures = measures * dt
    return measures



def ITOF(Incident, DemodFs, depths, trials, dt=1):
    (n_tbins, K) = Incident.shape
    measures = np.zeros((depths.shape[0], K, trials))
    depths = depths.astype(int)

    for j in range(0, K):
        demod = DemodFs[:, j]

        for l in range(0, depths.shape[0]):
            cc = np.roll(Incident[:, j], depths[l])
            measures[l, j, :] = AddPoissonNoiseLam(np.inner(cc, demod), trials)

    measures = measures * dt
    return measures


def ScaleIncidentAreaUnderCurve(x, dx=1., desiredArea=1.):
    oldArea = np.sum(x) * dx
    y = x * desiredArea / oldArea
    return y


def ScaleIncident(ModFs, dx=1., desiredArea=1.):
    (N, K) = ModFs.shape
    for i in range(0, K):
        ModFs[:, i] = ScaleIncidentAreaUnderCurve(x=ModFs[:, i], dx=dx, desiredArea=desiredArea)

    return ModFs

def ComputeMetrics(depths, decoded_depths):
    errors = np.abs(decoded_depths - depths[np.newaxis, :])
    mae = np.mean(np.mean(errors, axis=0))
    return mae


def ScaleAreaUnderCurve(x, dx=0., desiredArea=1.):
    """ScaleAreaUnderCurve: Scale the area under the curve x to some desired area.

    Args:
        x (TYPE): Discrete set of points that lie on the curve. Numpy vector
        dx (float): delta x. Set to 1/length of x by default.
        desiredArea (float): Desired area under the curve.

    Returns:
        numpy.ndarray: Scaled vector x with new area.
    """
    #### Validate Input
    # assert(UtilsTesting.IsVector(x)),'Input Error - ScaleAreaUnderCurve: x should be a vector.'
    #### Calculate some parameters
    N = x.size
    #### Set default value for dc
    if (dx == 0): dx = 1. / float(N)
    #### Calculate new area
    oldArea = np.sum(x) * dx
    y = x * desiredArea / oldArea
    #### Return scaled vector
    return y


def ScaleMod(ModFs, tau=1., pAveSource=1., dt=1.):
    """ScaleMod: Scale modulation appropriately given the beta of the scene point, the average
    source power and the repetition frequency.

    Args:
        ModFs (np.ndarray): N x K matrix. N samples, K modulation functions
        tau (float): Repetition frequency of ModFs
        pAveSource (float): Average power emitted by the source
        beta (float): Average reflectivity of scene point

    Returns:
        np.array: ModFs
    """
    (N, K) = ModFs.shape
    if (dt is None): dt = tau / float(N)
    eTotal = tau * pAveSource  # Total Energy
    for i in range(0, K):
        ModFs[:, i] = ScaleAreaUnderCurve(x=ModFs[:, i], dx=dt, desiredArea=eTotal)

    return ModFs

def plot(Measures, ITOF, IDTOF):
    plt.plot(ITOF)
    plt.plot(Measures, marker='o',
     linewidth=1)
    plt.plot(IDTOF)