"""
TMR Bridge Sensor & Signal Chain Noise Model — Interactive Dash Dashboard.

Run:  python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from tmr_noise import (
    NoiseModelParams, BridgeParams, INAParams, BPFParams,
    ADCParams, RefSupplyParams, AnalysisParams,
    compute_noise,
)
from tmr_noise.figures import (
    make_spectral_density_figure,
    make_rms_accumulation_figure,
    make_noise_budget_figure,
    make_gain_figure,
)

# ── App ──────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="TMR Noise Model",
)


# ── Helper: labelled numeric input ───────────────────────────────────────────

def _inp(label: str, id_: str, value, unit: str = "", step=None, min_=None):
    step_kw = {"step": step} if step is not None else {}
    min_kw = {"min": min_} if min_ is not None else {}
    return dbc.Row(
        [
            dbc.Col(html.Label(label, style={"fontSize": "0.82rem", "whiteSpace": "nowrap"}), width=7),
            dbc.Col(
                dbc.Input(
                    id=id_, type="number", value=value, debounce=True,
                    size="sm", style={"fontSize": "0.82rem"},
                    **step_kw, **min_kw,
                ),
                width=5,
            ),
            dbc.Col(html.Span(unit, style={"fontSize": "0.75rem", "color": "#777"}), width=12),
        ],
        className="mb-1 g-1",
    )


def _section(title: str, children, colour: str = "#1976D2"):
    return dbc.Card(
        [
            dbc.CardHeader(
                title,
                style={"background": colour, "color": "#fff",
                       "fontWeight": "600", "fontSize": "0.85rem", "padding": "6px 12px"},
            ),
            dbc.CardBody(children, style={"padding": "8px 12px"}),
        ],
        className="mb-2",
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

_defaults = NoiseModelParams()

sidebar = html.Div(
    [
        html.H5("Parameters", className="mb-3 mt-1", style={"fontWeight": "700"}),

        _section("Bridge Sensor", [
            _inp("Arm resistance R", "br-R", _defaults.bridge.R, "Ω"),
            _inp("Temperature T", "br-T", _defaults.bridge.T, "K"),
            _inp("1/f corner fc", "br-fc", _defaults.bridge.fc, "Hz"),
            _inp("Sensitivity", "br-sens", _defaults.bridge.sensitivity, "mV/V/mT"),
        ], colour="#1565C0"),

        _section("Instrumentation Amp (INA)", [
            _inp("Gain", "ina-gain", _defaults.ina.gain, "V/V"),
            _inp("Input voltage noise en", "ina-en", _defaults.ina.en, "nV/√Hz"),
            _inp("Input current noise in", "ina-in", _defaults.ina.in_, "pA/√Hz"),
            _inp("Voltage noise 1/f corner", "ina-fcv", _defaults.ina.fc_v, "Hz"),
            _inp("Current noise 1/f corner", "ina-fci", _defaults.ina.fc_i, "Hz"),
        ], colour="#2E7D32"),

        _section("Op-Amp BPF (MFB 2nd-order)", [
            _inp("Passband gain", "bpf-gain", _defaults.bpf.gain, "V/V"),
            _inp("Centre frequency f₀", "bpf-f0", _defaults.bpf.f0, "Hz"),
            _inp("Quality factor Q", "bpf-Q", _defaults.bpf.Q, ""),
            _inp("Op-amp voltage noise en", "bpf-en", _defaults.bpf.en, "nV/√Hz"),
            _inp("Op-amp current noise in", "bpf-in", _defaults.bpf.in_, "pA/√Hz"),
            _inp("Voltage noise 1/f corner", "bpf-fcv", _defaults.bpf.fc_v, "Hz"),
            _inp("Input resistor R_in", "bpf-Rin", _defaults.bpf.R_in, "Ω"),
            _inp("Feedback resistor R_fb", "bpf-Rfb", _defaults.bpf.R_fb, "Ω"),
        ], colour="#E65100"),

        _section("ADC", [
            _inp("Resolution N", "adc-bits", _defaults.adc.N_bits, "bits", step=1, min_=1),
            _inp("Reference voltage Vref", "adc-vref", _defaults.adc.Vref, "V"),
            _inp("Sampling rate fs", "adc-fs", _defaults.adc.fs, "Hz"),
        ], colour="#6A1B9A"),

        _section("Bridge Bias Reference Supply", [
            _inp("Supply voltage Vs", "ref-vs", _defaults.ref.V_supply, "V"),
            _inp("Supply noise en", "ref-en", _defaults.ref.en, "nV/√Hz"),
            _inp("Supply 1/f corner fc", "ref-fc", _defaults.ref.fc, "Hz"),
        ], colour="#B71C1C"),

        _section("Analysis Band", [
            _inp("Sweep start", "an-fstart", _defaults.analysis.f_start, "Hz", min_=1e-6),
            _inp("Sweep end", "an-fend", _defaults.analysis.f_end, "Hz"),
            _inp("Frequency points", "an-pts", _defaults.analysis.N_points, "", step=10, min_=50),
            _inp("RMS band low", "an-rmsl", _defaults.analysis.f_rms_low, "Hz"),
            _inp("RMS band high", "an-rmsh", _defaults.analysis.f_rms_high, "Hz"),
        ], colour="#004D40"),
    ],
    style={
        "overflowY": "auto",
        "height": "calc(100vh - 60px)",
        "padding": "8px 12px",
        "borderRight": "1px solid #e0e0e0",
        "background": "#f5f5f5",
    },
)


# ── Metrics strip ─────────────────────────────────────────────────────────────

def _metric_card(title, id_):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, style={"fontSize": "0.75rem", "margin": "0", "color": "#555"}),
            html.H5(id=id_, children="—", style={"margin": "0", "fontWeight": "700"}),
        ], style={"padding": "8px 14px"}),
        style={"minWidth": "160px"},
    )


metrics_strip = dbc.Row(
    [
        dbc.Col(_metric_card("RMS Input-Referred", "metric-rms-in"), width="auto"),
        dbc.Col(_metric_card("RMS Output-Referred", "metric-rms-out"), width="auto"),
        dbc.Col(_metric_card("Dominant Source", "metric-dom"), width="auto"),
        dbc.Col(_metric_card("Chain Gain @ f₀", "metric-gain"), width="auto"),
        dbc.Col(_metric_card("BPF Bandwidth", "metric-bw"), width="auto"),
    ],
    className="g-2 mb-3",
    align="center",
)


# ── Plots area ────────────────────────────────────────────────────────────────

plots_area = html.Div(
    [
        metrics_strip,
        dbc.Tabs(
            [
                dbc.Tab(
                    dcc.Graph(id="fig-input", style={"height": "500px"}),
                    label="Input-Referred NSD",
                ),
                dbc.Tab(
                    dcc.Graph(id="fig-output", style={"height": "500px"}),
                    label="Output-Referred NSD",
                ),
                dbc.Tab(
                    dcc.Graph(id="fig-rms", style={"height": "500px"}),
                    label="Cumulative RMS",
                ),
                dbc.Tab(
                    dcc.Graph(id="fig-budget", style={"height": "500px"}),
                    label="Noise Budget",
                ),
                dbc.Tab(
                    dcc.Graph(id="fig-gain", style={"height": "500px"}),
                    label="Chain Gain",
                ),
            ],
            id="tabs",
        ),
    ],
    style={"padding": "12px 18px"},
)


# ── Top navbar ────────────────────────────────────────────────────────────────

navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(html.Span("⚡", style={"fontSize": "1.4rem"}), width="auto"),
                dbc.Col(
                    html.Span(
                        "TMR Bridge Sensor & Signal Chain Noise Model",
                        style={"fontWeight": "700", "fontSize": "1.05rem", "color": "#fff"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Span(
                        "INA → BPF → ADC  |  input- & output-referred NSD  |  integrated RMS",
                        style={"fontSize": "0.8rem", "color": "#b0c4de"},
                    ),
                    width="auto",
                ),
            ],
            align="center",
        ),
        fluid=True,
    ),
    color="dark",
    dark=True,
    style={"height": "52px"},
)


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(plots_area, width=9),
            ],
            className="g-0",
            style={"height": "calc(100vh - 52px)"},
        ),
    ]
)


# ── Callback: update all outputs when any input changes ──────────────────────

_ALL_INPUTS = [
    # Bridge
    Input("br-R", "value"), Input("br-T", "value"), Input("br-fc", "value"),
    Input("br-sens", "value"),
    # INA
    Input("ina-gain", "value"), Input("ina-en", "value"), Input("ina-in", "value"),
    Input("ina-fcv", "value"), Input("ina-fci", "value"),
    # BPF
    Input("bpf-gain", "value"), Input("bpf-f0", "value"), Input("bpf-Q", "value"),
    Input("bpf-en", "value"), Input("bpf-in", "value"), Input("bpf-fcv", "value"),
    Input("bpf-Rin", "value"), Input("bpf-Rfb", "value"),
    # ADC
    Input("adc-bits", "value"), Input("adc-vref", "value"), Input("adc-fs", "value"),
    # Ref supply
    Input("ref-vs", "value"), Input("ref-en", "value"), Input("ref-fc", "value"),
    # Analysis
    Input("an-fstart", "value"), Input("an-fend", "value"), Input("an-pts", "value"),
    Input("an-rmsl", "value"), Input("an-rmsh", "value"),
]


def _safe(val, default, min_val=None):
    """Return val if valid, otherwise default.  Clamp to min_val if given."""
    try:
        v = float(val) if not isinstance(val, int) else val
        if v != v:  # NaN
            return default
        if min_val is not None and v < min_val:
            return min_val
        return v
    except (TypeError, ValueError):
        return default


@app.callback(
    Output("fig-input", "figure"),
    Output("fig-output", "figure"),
    Output("fig-rms", "figure"),
    Output("fig-budget", "figure"),
    Output("fig-gain", "figure"),
    Output("metric-rms-in", "children"),
    Output("metric-rms-out", "children"),
    Output("metric-dom", "children"),
    Output("metric-gain", "children"),
    Output("metric-bw", "children"),
    _ALL_INPUTS,
    prevent_initial_call=False,
)
def update_all(
    br_R, br_T, br_fc, br_sens,
    ina_gain, ina_en, ina_in, ina_fcv, ina_fci,
    bpf_gain, bpf_f0, bpf_Q, bpf_en, bpf_in, bpf_fcv, bpf_Rin, bpf_Rfb,
    adc_bits, adc_vref, adc_fs,
    ref_vs, ref_en, ref_fc,
    an_fstart, an_fend, an_pts, an_rmsl, an_rmsh,
):
    d = _defaults

    params = NoiseModelParams(
        bridge=BridgeParams(
            R=_safe(br_R, d.bridge.R, 1.0),
            T=_safe(br_T, d.bridge.T, 1.0),
            fc=_safe(br_fc, d.bridge.fc, 1e-3),
            sensitivity=_safe(br_sens, d.bridge.sensitivity),
        ),
        ina=INAParams(
            gain=_safe(ina_gain, d.ina.gain, 1e-3),
            en=_safe(ina_en, d.ina.en, 1e-6),
            in_=_safe(ina_in, d.ina.in_, 0.0),
            fc_v=_safe(ina_fcv, d.ina.fc_v, 1e-3),
            fc_i=_safe(ina_fci, d.ina.fc_i, 1e-3),
        ),
        bpf=BPFParams(
            gain=_safe(bpf_gain, d.bpf.gain, 1e-3),
            f0=_safe(bpf_f0, d.bpf.f0, 1e-3),
            Q=_safe(bpf_Q, d.bpf.Q, 0.1),
            en=_safe(bpf_en, d.bpf.en, 1e-6),
            in_=_safe(bpf_in, d.bpf.in_, 0.0),
            fc_v=_safe(bpf_fcv, d.bpf.fc_v, 1e-3),
            R_in=_safe(bpf_Rin, d.bpf.R_in, 1.0),
            R_fb=_safe(bpf_Rfb, d.bpf.R_fb, 1.0),
        ),
        adc=ADCParams(
            N_bits=max(1, int(_safe(adc_bits, d.adc.N_bits))),
            Vref=_safe(adc_vref, d.adc.Vref, 1e-3),
            fs=_safe(adc_fs, d.adc.fs, 1.0),
        ),
        ref=RefSupplyParams(
            V_supply=_safe(ref_vs, d.ref.V_supply, 1e-3),
            en=_safe(ref_en, d.ref.en, 0.0),
            fc=_safe(ref_fc, d.ref.fc, 1e-3),
        ),
        analysis=AnalysisParams(
            f_start=_safe(an_fstart, d.analysis.f_start, 1e-6),
            f_end=_safe(an_fend, d.analysis.f_end, 1.0),
            N_points=max(50, int(_safe(an_pts, d.analysis.N_points))),
            f_rms_low=_safe(an_rmsl, d.analysis.f_rms_low, 1e-6),
            f_rms_high=_safe(an_rmsh, d.analysis.f_rms_high, 1.0),
        ),
    )

    # Ensure f_end > f_start and rms band is valid
    if params.analysis.f_end <= params.analysis.f_start:
        params.analysis.f_end = params.analysis.f_start * 10.0
    if params.analysis.f_rms_high <= params.analysis.f_rms_low:
        params.analysis.f_rms_high = params.analysis.f_rms_low * 10.0

    result = compute_noise(params)

    # ---- Figures ----
    fig_in = make_spectral_density_figure(result, referred="input")
    fig_out = make_spectral_density_figure(result, referred="output")
    fig_rms = make_rms_accumulation_figure(result)
    fig_budget = make_noise_budget_figure(result)
    fig_gain = make_gain_figure(result)

    # ---- Metrics ----
    def _fmt_rms(v_rms):
        v_nv = v_rms * 1e9
        if v_nv >= 1000:
            return f"{v_nv/1000:.2f} µV_rms"
        return f"{v_nv:.1f} nV_rms"

    rms_in_str = _fmt_rms(result.rms_input)
    rms_out_str = _fmt_rms(result.rms_output)

    source_rms = {
        "Bridge": result.rms_bridge,
        "INA voltage": result.rms_ina_v,
        "INA current": result.rms_ina_i,
        "BPF": result.rms_bpf,
        "ADC": result.rms_adc,
        "Ref supply": result.rms_ref,
    }
    dom = max(source_rms, key=source_rms.get)

    G_at_f0 = params.ina.gain * params.bpf.gain  # gain at BPF centre
    gain_str = f"{20 * __import__('math').log10(G_at_f0):.1f} dB"

    bw_3db = params.bpf.f0 / params.bpf.Q
    bw_str = f"{bw_3db:.1f} Hz"

    return (fig_in, fig_out, fig_rms, fig_budget, fig_gain,
            rms_in_str, rms_out_str, dom, gain_str, bw_str)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
