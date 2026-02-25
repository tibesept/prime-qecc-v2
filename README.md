# Prime Number QECC

A numerical toy model linking the Weil explicit formula with p-adic holographic codes.

## Project Structure

- `data/` : Caches for Riemann zeros data.
- `src/` : Python source files for the implementation.
  - `data_loader.py`: Fetches dataset from Odlyzko tables.
  - `weil_*.py`: Computes terms for the Weil explicit functional formula.
  - `bruhat_tits.py`: Constructs the p-adic Bruhat-Tits tree.
  - `connection.py`: Connects structural properties evaluated.
  - `dashboard.py`: Renders Plotly outputs.
- `tests/`: Basic validation scripts.
- `requirements.txt`: Python package requirements.

## Running

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cd src && python main.py`

## Visualizations

The model produces HTML visualization dashboards demonstrating the mathematical concepts studied. Open `dashboard.html`, `weil_components.html`, and `tree_initial.html` in your browser.
