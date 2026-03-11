import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dashboard(
    results_weil: dict,
    results_graph: dict,
    results_positivity: dict = None,
    results_intrinsic: dict = None,
    results_tensor: dict = None,
    results_coupled: dict = None,
    output_file: str = "dashboard.html",
):
    """
    Creates an interactive Plotly dashboard.
    """
    titles = [
        "Weil Functional Robustness W(sigma)",
        "Resonance Spectrum (Synthetic Prime Deltas)",
    ]
    specs = [[{"type": "scatter"}, {"type": "bar"}]]

    if results_positivity:
        titles.append("Weil Positivity Criterion")
        specs[0].append({"type": "scatter"})
    if results_intrinsic:
        titles.append("Intrinsic Transfer Kernel")
        specs[0].append({"type": "scatter"})
    if results_tensor:
        titles.append("Tensor Ansatz Diagnostics")
        specs[0].append({"type": "scatter"})
    if results_coupled:
        titles.append("Coupled Ansatz: Prime to Tensor")
        specs[0].append({"type": "bar"})

    ncols = len(titles)

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles, specs=specs)

    # Panel 1: W(sigma)
    sigmas = results_weil["sigma_values"]
    w_values = results_weil["w_values"]
    fig.add_trace(
        go.Scatter(
            x=sigmas,
            y=w_values,
            mode="markers+lines",
            marker=dict(size=10, color="blue"),
            line=dict(color="blue", width=2),
            name="W",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.update_xaxes(title_text="sigma", row=1, col=1)
    fig.update_yaxes(title_text="W(F)", row=1, col=1)

    # Panel 2: Prime deltas
    primes = list(results_graph["prime_data"].keys())
    deltas = [results_graph["prime_data"][p]["delta"] for p in primes]
    colors = ["red" if d < 0 else "green" for d in deltas]
    fig.add_trace(
        go.Bar(
            x=[str(p) for p in primes],
            y=deltas,
            marker=dict(color=colors),
            name="delta W_p",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="prime p", type="category", row=1, col=2)
    fig.update_yaxes(title_text="delta W_p", row=1, col=2)

    current_col = 3

    if results_positivity:
        scan = results_positivity.get("scan_results", [])
        delta_scan = [0.0] + [item["delta"] for item in scan]
        w_scan = [results_positivity["W_healthy"]] + [item["W_broken"] for item in scan]
        marker_colors = ["green" if w >= -1e-10 else "red" for w in w_scan]

        fig.add_trace(
            go.Scatter(
                x=delta_scan,
                y=w_scan,
                mode="markers+lines",
                marker=dict(size=11, color=marker_colors, line=dict(width=1, color="black")),
                line=dict(color="gray", width=2, dash="dot"),
                name="W positivity",
            ),
            row=1,
            col=current_col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=current_col)
        fig.update_xaxes(title_text="delta shift", row=1, col=current_col)
        fig.update_yaxes(title_text="W(F)", row=1, col=current_col)
        current_col += 1

    if results_intrinsic:
        scan = results_intrinsic.get("scan_results", [])
        x_vals = [0.0] + [item["delta"] for item in scan]
        herm = [results_intrinsic["healthy"]["hermiticity_defect"]] + [
            item["hermiticity_defect"] for item in scan
        ]
        mineig = [results_intrinsic["healthy"]["min_hermitian_eigenvalue"]] + [
            item["min_hermitian_eigenvalue"] for item in scan
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=herm,
                mode="markers+lines",
                marker=dict(size=9, color="crimson"),
                line=dict(color="crimson", width=2),
                name="Hermiticity defect",
            ),
            row=1,
            col=current_col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mineig,
                mode="markers+lines",
                marker=dict(size=9, color="teal"),
                line=dict(color="teal", width=2),
                name="Min eig(H)",
            ),
            row=1,
            col=current_col,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=current_col)
        fig.update_xaxes(title_text="delta shift", row=1, col=current_col)
        fig.update_yaxes(title_text="kernel diagnostics", row=1, col=current_col)
        current_col += 1

    if results_tensor:
        diagnostics = results_tensor.get("diagnostics", [])
        delta_values = [item["delta"] for item in diagnostics]
        iso = [item["isometry_defect"] for item in diagnostics]
        tp = [item["trace_preservation_defect"] for item in diagnostics]

        fig.add_trace(
            go.Scatter(
                x=delta_values,
                y=iso,
                mode="markers+lines",
                marker=dict(size=9, color="darkblue"),
                line=dict(color="darkblue", width=2),
                name="isometry defect",
            ),
            row=1,
            col=current_col,
        )
        fig.add_trace(
            go.Scatter(
                x=delta_values,
                y=tp,
                mode="markers+lines",
                marker=dict(size=9, color="orange"),
                line=dict(color="orange", width=2),
                name="tp defect",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(title_text="delta", row=1, col=current_col)
        fig.update_yaxes(title_text="defect norm", row=1, col=current_col)
        current_col += 1

    if results_coupled:
        rows = results_coupled.get("prime_diagnostics", [])
        labels = [str(item["prime"]) for item in rows]
        values = [item["tensor"]["isometry_defect"] for item in rows]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color="purple"),
                name="coupled isometry defect",
            ),
            row=1,
            col=current_col,
        )
        fig.update_xaxes(title_text="prime", type="category", row=1, col=current_col)
        fig.update_yaxes(title_text="isometry defect", row=1, col=current_col)

    fig.update_layout(
        height=560,
        width=max(1200, 420 * ncols),
        title_text="Prime-QECC Toy Model Dashboard",
        showlegend=False,
    )
    fig.write_html(output_file)
    print(f"Saved dashboard to {output_file}")


def plot_weil_components(components: dict, output_file: str = "weil_components.html"):
    labels = ["W_arch", "W_zeros", "W_primes", "W_total"]
    values = [
        float(components["W_archimedean"]),
        float(components["W_zeros"]),
        float(components["W_primes"]),
        float(components["W_total"]),
    ]
    colors = ["blue", "green", "orange", "red" if values[3] < 0 else "darkgreen"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors),
                text=[f"{v:.4e}" for v in values],
                textposition="auto",
            )
        ]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(title="Weil Functional Decomposition", yaxis_title="Contribution", height=500, width=900)
    fig.write_html(output_file)
    print(f"Saved components plot to {output_file}")
