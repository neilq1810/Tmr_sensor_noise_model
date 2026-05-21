import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .signal_chain import NoiseResult

# Consistent colour palette per source
_COLOURS = {
    "Bridge": "#2196F3",
    "INA voltage": "#4CAF50",
    "INA current": "#8BC34A",
    "BPF": "#FF9800",
    "ADC": "#9C27B0",
    "Ref supply": "#F44336",
    "Total": "#212121",
}

_DASH_STYLES = {
    "Bridge": "solid",
    "INA voltage": "solid",
    "INA current": "dash",
    "BPF": "solid",
    "ADC": "dot",
    "Ref supply": "dashdot",
    "Total": "solid",
}


def _add_trace(fig, f, en, name, row=1, col=1, visible=True):
    fig.add_trace(
        go.Scatter(
            x=f, y=en * 1e9,
            name=name,
            mode="lines",
            line=dict(color=_COLOURS[name], dash=_DASH_STYLES[name],
                      width=2.5 if name == "Total" else 1.5),
            visible=visible,
        ),
        row=row, col=col,
    )


def make_spectral_density_figure(result: NoiseResult, referred: str = "input") -> go.Figure:
    """
    Log-log noise spectral density plot (nV/√Hz vs Hz).
    referred: 'input' or 'output'
    """
    f = result.f
    if referred == "input":
        sources = [
            ("Bridge", result.en_bridge),
            ("INA voltage", result.en_ina_v),
            ("INA current", result.en_ina_i),
            ("BPF", result.en_bpf),
            ("ADC", result.en_adc),
            ("Ref supply", result.en_ref),
            ("Total", result.en_total_input),
        ]
        title = "Input-Referred Noise Spectral Density"
    else:
        sources = [
            ("Bridge", result.en_bridge_out),
            ("INA voltage", result.en_ina_v_out),
            ("INA current", result.en_ina_i_out),
            ("BPF", result.en_bpf_out),
            ("ADC", result.en_adc_out),
            ("Ref supply", result.en_ref_out),
            ("Total", result.en_total_output),
        ]
        title = "Output-Referred Noise Spectral Density"

    fig = go.Figure()
    for name, en in sources:
        fig.add_trace(go.Scatter(
            x=f, y=en * 1e9,
            name=name,
            mode="lines",
            line=dict(color=_COLOURS[name], dash=_DASH_STYLES[name],
                      width=2.5 if name == "Total" else 1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Frequency (Hz)", type="log", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Noise Density (nV/√Hz)", type="log", showgrid=True, gridcolor="#e0e0e0"),
        legend=dict(x=1.01, y=1, xanchor="left"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=160, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def make_rms_accumulation_figure(result: NoiseResult) -> go.Figure:
    """
    Cumulative RMS noise [nV_rms] from f_start up to each frequency point.
    Shows how the total RMS noise builds across the spectrum.
    """
    f = result.f
    en2 = result.en_total_input ** 2

    # Cumulative integral using the trapezoidal rule
    df = np.diff(f)
    en2_mid = 0.5 * (en2[:-1] + en2[1:])
    cumulative = np.concatenate([[0.0], np.cumsum(en2_mid * df)])
    rms_accum = np.sqrt(cumulative) * 1e9  # nV_rms

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, y=rms_accum,
        mode="lines",
        line=dict(color=_COLOURS["Total"], width=2),
        name="Cumulative RMS",
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
    ))
    fig.update_layout(
        title="Cumulative Input-Referred RMS Noise",
        xaxis=dict(title="Upper Integration Frequency (Hz)", type="log",
                   showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Cumulative RMS Noise (nV_rms)", showgrid=True,
                   gridcolor="#e0e0e0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified",
    )
    return fig


def make_noise_budget_figure(result: NoiseResult) -> go.Figure:
    """
    Horizontal bar chart of per-source RMS noise contribution over the
    integration band (input-referred, nV_rms).
    """
    sources = ["Bridge", "INA voltage", "INA current", "BPF", "ADC", "Ref supply"]
    values = [
        result.rms_bridge * 1e9,
        result.rms_ina_v * 1e9,
        result.rms_ina_i * 1e9,
        result.rms_bpf * 1e9,
        result.rms_adc * 1e9,
        result.rms_ref * 1e9,
    ]
    colours = [_COLOURS[s] for s in sources]

    # Sort descending for readability
    order = np.argsort(values)[::-1]
    sources_s = [sources[i] for i in order]
    values_s = [values[i] for i in order]
    colours_s = [colours[i] for i in order]

    fig = go.Figure(go.Bar(
        x=values_s,
        y=sources_s,
        orientation="h",
        marker_color=colours_s,
        text=[f"{v:.2f} nV" for v in values_s],
        textposition="outside",
    ))
    fig.update_layout(
        title="Noise Budget (Input-Referred RMS per Source)",
        xaxis=dict(title="RMS Noise (nV_rms)", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        margin=dict(l=120, r=80, t=50, b=60),
        showlegend=False,
    )
    return fig


def make_gain_figure(result: NoiseResult) -> go.Figure:
    """
    Chain gain profile |G_chain(f)| = G_ina × |H_bpf(f)| in dB.
    """
    f = result.f
    G_db = 20.0 * np.log10(np.abs(result.G_chain) + 1e-300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, y=G_db,
        mode="lines",
        line=dict(color="#3F51B5", width=2),
        name="Chain Gain",
    ))
    fig.update_layout(
        title="Signal Chain Gain Profile",
        xaxis=dict(title="Frequency (Hz)", type="log", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Gain (dB)", showgrid=True, gridcolor="#e0e0e0"),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified",
    )
    return fig
