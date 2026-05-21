import numpy as np
from .params import BridgeParams, INAParams, BPFParams, ADCParams, RefSupplyParams

k_B = 1.380649e-23  # Boltzmann constant [J/K]


def _johnson(R: float, T: float) -> float:
    """White Johnson-Nyquist noise spectral density [V/√Hz]."""
    return float(np.sqrt(4.0 * k_B * T * R))


def _shape_1f(f: np.ndarray, fc: float) -> np.ndarray:
    """Combined white + 1/f shaping: sqrt(1 + fc/f)."""
    return np.sqrt(1.0 + fc / f)


def bridge_noise(f: np.ndarray, p: BridgeParams) -> np.ndarray:
    """
    TMR bridge Johnson + 1/f noise referred to bridge differential output [V/√Hz].

    For a balanced Wheatstone bridge with 4 equal arms of resistance R, the
    total differential output noise is sqrt(4kTR) — equivalent to one arm's
    Johnson noise (the cross-terms cancel for uncorrelated resistors).
    The 1/f term models magnetic/material excess noise with corner fc.
    """
    en_white = _johnson(p.R, p.T)
    return en_white * _shape_1f(f, p.fc)


def ina_voltage_noise(f: np.ndarray, p: INAParams) -> np.ndarray:
    """INA input-referred voltage noise [V/√Hz]."""
    en = p.en * 1e-9
    return en * _shape_1f(f, p.fc_v)


def ina_current_noise(f: np.ndarray, p: INAParams, R_source: float) -> np.ndarray:
    """
    INA current noise referred to INA input [V/√Hz].

    Two uncorrelated current noise sources (one per input) each flow through
    the Thevenin source resistance.  Combined differential contribution:
    sqrt(2) * in * R_source.  We use the differential factor sqrt(2) implicitly
    by noting that for a bridge R_source = R_bridge (one arm), which already
    accounts for the full bridge impedance seen by the INA.
    """
    in_ = p.in_ * 1e-12
    return in_ * R_source * _shape_1f(f, p.fc_i)


def bpf_transfer(f: np.ndarray, p: BPFParams) -> np.ndarray:
    """
    2nd-order MFB bandpass magnitude |H(f)|.

    Transfer function: H(jω) = G·(ω₀/Q)·jω / (ω₀² − ω² + j·ω·ω₀/Q)
    At f = f₀: |H| = G (passband gain).
    """
    w0 = 2.0 * np.pi * p.f0
    w = 2.0 * np.pi * f
    H_num = p.gain * (w0 / p.Q) * 1j * w
    H_den = w0**2 - w**2 + 1j * w * (w0 / p.Q)
    return np.abs(H_num / H_den)


def bpf_noise_output(f: np.ndarray, p: BPFParams, T: float = 298.0) -> np.ndarray:
    """
    BPF stage total noise referred to BPF output [V/√Hz].

    Op-amp voltage noise + current noise through feedback resistor + Johnson
    noise of R_in and R_fb are combined at the op-amp input, then shaped by
    |H_bpf(f)| to produce output-referred noise.  This is an approximation:
    the noise transfer function of the MFB topology is assumed equal to the
    signal transfer function, which is accurate near the passband.
    """
    en_v = p.en * 1e-9 * _shape_1f(f, p.fc_v)
    en_i_rfb = p.in_ * 1e-12 * p.R_fb
    en_Rin = _johnson(p.R_in, T)
    en_Rfb = _johnson(p.R_fb, T)
    en_total_input = np.sqrt(en_v**2 + en_i_rfb**2 + en_Rin**2 + en_Rfb**2)
    return en_total_input * bpf_transfer(f, p)


def adc_quantization_noise(f: np.ndarray, p: ADCParams) -> np.ndarray:
    """
    ADC quantization noise spectral density [V/√Hz].

    Flat (white) spectrum: e_q = LSB/sqrt(12), spread over Nyquist bandwidth.
    """
    lsb = p.Vref / (2**p.N_bits)
    e_q_rms = lsb / np.sqrt(12.0)
    e_q_density = e_q_rms / np.sqrt(p.fs / 2.0)
    return np.full_like(f, e_q_density)


def ref_supply_noise(f: np.ndarray, p: RefSupplyParams) -> np.ndarray:
    """
    Bridge reference supply noise referred to bridge output [V/√Hz].

    The supply noise is specified directly as a density at the bridge output
    (the user accounts for the bridge sensitivity / supply ratio when entering
    the parameter).  1/f shaping uses the supply reference corner fc.
    """
    en = p.en * 1e-9
    return en * _shape_1f(f, p.fc)
