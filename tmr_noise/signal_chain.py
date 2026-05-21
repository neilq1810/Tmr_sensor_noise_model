from dataclasses import dataclass
import numpy as np
from .params import NoiseModelParams
from .noise_sources import (
    bridge_noise, ina_voltage_noise, ina_current_noise,
    bpf_transfer, bpf_noise_output, adc_quantization_noise, ref_supply_noise,
)


@dataclass
class NoiseResult:
    # Frequency axis [Hz]
    f: np.ndarray

    # ---- Individual sources — INPUT-REFERRED [V/√Hz] ----
    en_bridge: np.ndarray        # Bridge Johnson + 1/f
    en_ina_v: np.ndarray         # INA voltage noise
    en_ina_i: np.ndarray         # INA current noise × source impedance
    en_bpf: np.ndarray           # BPF op-amp + resistor noise
    en_adc: np.ndarray           # ADC quantization noise
    en_ref: np.ndarray           # Bridge reference supply noise

    # ---- Total input-referred noise [V/√Hz] ----
    en_total_input: np.ndarray

    # ---- Output-referred equivalents [V/√Hz] ----
    en_bridge_out: np.ndarray
    en_ina_v_out: np.ndarray
    en_ina_i_out: np.ndarray
    en_bpf_out: np.ndarray
    en_adc_out: np.ndarray
    en_ref_out: np.ndarray
    en_total_output: np.ndarray

    # ---- Chain gain profile ----
    G_chain: np.ndarray          # G_ina × |H_bpf(f)|

    # ---- RMS metrics [V_rms] over the configured integration band ----
    rms_input: float
    rms_output: float

    # ---- Per-source RMS [V_rms] over integration band (input-referred) ----
    rms_bridge: float
    rms_ina_v: float
    rms_ina_i: float
    rms_bpf: float
    rms_adc: float
    rms_ref: float


def _rms(en: np.ndarray, f: np.ndarray, f_low: float, f_high: float) -> float:
    """Integrate en²(f) over [f_low, f_high] and return sqrt."""
    mask = (f >= f_low) & (f <= f_high)
    if mask.sum() < 2:
        return 0.0
    return float(np.sqrt(np.trapezoid(en[mask] ** 2, f[mask])))


def compute_noise(p: NoiseModelParams) -> NoiseResult:
    """
    Compute noise spectral density for all sources across the signal chain.

    Referral strategy:
      - Bridge and INA sources are already at the INA input — no division needed.
      - Ref supply noise is at the bridge output = INA input — no division needed.
      - BPF noise is referred to BPF output; divide by G_ina to get INA input.
      - ADC noise is at ADC output; divide by G_chain(f) to get INA input.
    """
    f = np.geomspace(p.analysis.f_start, p.analysis.f_end, p.analysis.N_points)

    # Chain gain profile
    H_bpf = bpf_transfer(f, p.bpf)
    G_chain = p.ina.gain * H_bpf

    # --- Noise at their native chain positions ---

    # Bridge output = INA input
    en_bridge = bridge_noise(f, p.bridge)
    en_ina_v = ina_voltage_noise(f, p.ina)
    en_ina_i = ina_current_noise(f, p.ina, R_source=p.bridge.R)
    en_ref = ref_supply_noise(f, p.ref)

    # BPF output → refer back to INA input (÷ G_ina)
    en_bpf_at_output = bpf_noise_output(f, p.bpf, T=p.bridge.T)
    en_bpf = en_bpf_at_output / p.ina.gain

    # ADC output → refer back to INA input (÷ G_chain)
    # Guard against near-zero G_chain at band edges
    G_safe = np.where(np.abs(G_chain) < 1e-12, 1e-12, np.abs(G_chain))
    en_adc_at_output = adc_quantization_noise(f, p.adc)
    en_adc = en_adc_at_output / G_safe

    # --- Total input-referred ---
    en_total_input = np.sqrt(
        en_bridge**2 + en_ina_v**2 + en_ina_i**2 +
        en_bpf**2 + en_adc**2 + en_ref**2
    )

    # --- Output-referred (multiply input-referred by G_chain) ---
    def _to_out(en_in):
        return en_in * np.abs(G_chain)

    en_bridge_out = _to_out(en_bridge)
    en_ina_v_out = _to_out(en_ina_v)
    en_ina_i_out = _to_out(en_ina_i)
    en_bpf_out = _to_out(en_bpf)
    en_adc_out = _to_out(en_adc)
    en_ref_out = _to_out(en_ref)
    en_total_output = _to_out(en_total_input)

    # --- RMS over integration band ---
    fl, fh = p.analysis.f_rms_low, p.analysis.f_rms_high
    rms_input = _rms(en_total_input, f, fl, fh)
    rms_output = _rms(en_total_output, f, fl, fh)

    rms_bridge = _rms(en_bridge, f, fl, fh)
    rms_ina_v = _rms(en_ina_v, f, fl, fh)
    rms_ina_i = _rms(en_ina_i, f, fl, fh)
    rms_bpf = _rms(en_bpf, f, fl, fh)
    rms_adc = _rms(en_adc, f, fl, fh)
    rms_ref = _rms(en_ref, f, fl, fh)

    return NoiseResult(
        f=f,
        en_bridge=en_bridge, en_ina_v=en_ina_v, en_ina_i=en_ina_i,
        en_bpf=en_bpf, en_adc=en_adc, en_ref=en_ref,
        en_total_input=en_total_input,
        en_bridge_out=en_bridge_out, en_ina_v_out=en_ina_v_out,
        en_ina_i_out=en_ina_i_out, en_bpf_out=en_bpf_out,
        en_adc_out=en_adc_out, en_ref_out=en_ref_out,
        en_total_output=en_total_output,
        G_chain=G_chain,
        rms_input=rms_input, rms_output=rms_output,
        rms_bridge=rms_bridge, rms_ina_v=rms_ina_v, rms_ina_i=rms_ina_i,
        rms_bpf=rms_bpf, rms_adc=rms_adc, rms_ref=rms_ref,
    )
