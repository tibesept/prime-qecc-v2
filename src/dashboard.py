import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bruhat_tits import BruhatTitsTree


def create_dashboard(results_weil: dict,
                     results_graph: dict,
                     output_file: str = "dashboard.html"):
    """
    Creates an interactive Plotly dashboard.
    Panel 1: W(sigma)
    Panel 2: Resonance Spectrum (Delta W_p)
    Panel 3 is actually rendered via tree.visualize() separately, but we could try to combine them.
    We will just write Panel 1 and Panel 2 here.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Weil Functional Robustness W(σ)", "Zeta Resonance Spectrum (Broken RH)"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )

    # Panel 1: W(sigma)
    sigmas = results_weil['sigma_values']
    w_values = results_weil['w_values']

    fig.add_trace(
        go.Scatter(
            x=sigmas, y=w_values,
            mode='markers+lines',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=2),
            name='W(f)'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.update_xaxes(title_text="σ (Gaussian width)", row=1, col=1)
    fig.update_yaxes(title_text="W(f)", row=1, col=1)

    # Panel 2: Resonance Spectrum
    if 'prime_data' in results_graph:
        primes = list(results_graph['prime_data'].keys())
        deltas = [results_graph['prime_data'][p]['delta'] for p in primes]
        
        # Color red for negative shift, green for positive
        colors = ['red' if d < 0 else 'green' for d in deltas]

        fig.add_trace(
            go.Bar(
                x=[str(p) for p in primes], y=deltas,
                marker=dict(color=colors),
                name='Δ W_p'
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Prime p", type='category', row=1, col=2)
        fig.update_yaxes(title_text="Δ W_p (Shift Impact)", row=1, col=2)

    fig.update_layout(
        height=500, width=1200,
        title_text="Prime Number QECC: Zeta Resonance Analysis",
        showlegend=False,
    )

    fig.write_html(output_file)
    print(f"✓ Saved dashboard to {output_file}")


def plot_weil_components(components: dict, output_file: str = "weil_components.html"):
    """
    Plots the components of the Weil functional for deeper introspection.
    """
    labels = ['W_R (Archimedean)', 'W_zeros (Zeros)', 'W_primes (Primes)', 'W_TOTAL']
    values = [
        float(components['W_archimedean']),
        float(components['W_zeros']),
        float(components['W_primes']),
        float(components['W_total'])
    ]

    colors = ['blue', 'green', 'orange', 'red' if values[3] < 0 else 'darkgreen']

    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker=dict(color=colors),
               text=[f"{v:.4e}" for v in values], textposition="auto")
    ])

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title="Weil Functional Decomposition",
        yaxis_title="Contribution",
        height=500, width=800
    )

    fig.write_html(output_file)
    print(f"✓ Saved components plot to {output_file}")
