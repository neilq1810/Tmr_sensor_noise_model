from dataclasses import dataclass, field


@dataclass
class BridgeParams:
    R: float = 5000.0         # Bridge arm resistance [Ω]
    T: float = 298.0          # Temperature [K]
    fc: float = 100.0         # 1/f corner frequency [Hz]
    sensitivity: float = 1.0  # Sensitivity [mV/V/mT] — informational only


@dataclass
class INAParams:
    gain: float = 100.0   # Gain [V/V]
    en: float = 7.0       # Input voltage noise density [nV/√Hz] @ 1 kHz
    in_: float = 0.4      # Input current noise density [pA/√Hz] @ 1 kHz
    fc_v: float = 10.0    # Voltage noise 1/f corner [Hz]
    fc_i: float = 1.0     # Current noise 1/f corner [Hz]


@dataclass
class BPFParams:
    gain: float = 10.0       # Passband gain [V/V]
    f0: float = 1000.0       # Centre frequency [Hz]
    Q: float = 5.0           # Quality factor (BW = f0/Q)
    en: float = 10.0         # Op-amp input voltage noise [nV/√Hz]
    in_: float = 0.01        # Op-amp input current noise [pA/√Hz]
    fc_v: float = 100.0      # Voltage noise 1/f corner [Hz]
    R_in: float = 10000.0    # Input resistor [Ω]
    R_fb: float = 100000.0   # Feedback resistor [Ω]


@dataclass
class ADCParams:
    N_bits: int = 16       # Resolution [bits]
    Vref: float = 5.0      # Reference voltage [V]
    fs: float = 10000.0    # Sampling rate [Hz]


@dataclass
class RefSupplyParams:
    V_supply: float = 5.0   # Bridge excitation voltage [V]
    en: float = 10.0        # Supply noise density [nV/√Hz]
    fc: float = 10.0        # 1/f corner [Hz]


@dataclass
class AnalysisParams:
    f_start: float = 0.1      # Frequency sweep start [Hz]
    f_end: float = 10000.0    # Frequency sweep end [Hz]
    N_points: int = 1000      # Number of log-spaced frequency points
    f_rms_low: float = 1.0    # RMS integration lower bound [Hz]
    f_rms_high: float = 1000.0  # RMS integration upper bound [Hz]


@dataclass
class NoiseModelParams:
    bridge: BridgeParams = field(default_factory=BridgeParams)
    ina: INAParams = field(default_factory=INAParams)
    bpf: BPFParams = field(default_factory=BPFParams)
    adc: ADCParams = field(default_factory=ADCParams)
    ref: RefSupplyParams = field(default_factory=RefSupplyParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
