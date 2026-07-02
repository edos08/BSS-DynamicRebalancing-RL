"""Main Dash application."""

import argparse
import logging
import dash
import dash_bootstrap_components as dbc

from pathlib import Path
from dash import dcc, html

from results_webapp.callbacks import register_callbacks
from results_webapp.layouts import create_header, create_controls, PLOT_CONFIG


# ---------------------------------------------------------------------------
# Tab factories — one per mode
# ---------------------------------------------------------------------------

def _training_tabs():
    """Tabs shown when Training mode is selected."""
    return dbc.Tabs(id='mode-tabs', children=[

        # ── Overview ────────────────────────────────────────────────────────
        dbc.Tab(label='📊 Overview', tab_id='tab-overview', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id='mean-failures-plot', config=PLOT_CONFIG), width=6),
                dbc.Col(dcc.Graph(id='rewards-plot',       config=PLOT_CONFIG), width=6),
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='epsilon-plot',        config=PLOT_CONFIG), width=6),
                dbc.Col(dcc.Graph(id='total-failures-plot', config=PLOT_CONFIG), width=6),
            ], className='mt-3'),
        ]),

        # ── Episode Details ──────────────────────────────────────────────────
        dbc.Tab(label='🔍 Episode Details', tab_id='tab-episode', children=[
            _episode_detail_rows(prefix='train'),
        ]),
    ])


def _validation_tabs():
    """Tabs shown when Validation mode is selected."""
    return dbc.Tabs(id='mode-tabs', active_tab='tab-best', children=[

        # ── Best ────────────────────────────────────────────────────────────
        dbc.Tab(label='🏆 Best', tab_id='tab-best', children=[
            dbc.Row([
                dbc.Col(html.Div(id='best-episode-banner'), width=12)
            ], className='mt-3'),
            _episode_detail_rows(prefix='best'),
        ]),

        # ── Episode Details ──────────────────────────────────────────────────
        dbc.Tab(label='🔍 Episode Details', tab_id='tab-episode', children=[
            _episode_detail_rows(prefix='val'),
        ]),
    ])


def _benchmark_tabs():
    """Tabs shown when Benchmark mode is selected — single tab, stripped down."""
    return dbc.Tabs(id='mode-tabs', active_tab='tab-bench',children=[
        dbc.Tab(label='📋 Results', tab_id='tab-bench', children=[
            dbc.Row([
                dbc.Col(html.Div(id='bench-stats-cards'), width=12)
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='bench-failures-plot',       config=PLOT_CONFIG), width=6),
                dbc.Col(dcc.Graph(id='bench-demand-plot', config=PLOT_CONFIG), width=6),
            ], className='mt-3'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='bench-rebalance-times-plot', config=PLOT_CONFIG), width=6),
            ], className='mt-3'),
        ]),
    ])


def _episode_detail_rows(prefix: str):
    """
    Return the episode detail plot grid.

    `prefix` is used to namespace component IDs so training, validation and
    best tabs don't clash:
        'train'  → train-timeslot-rewards-plot, …
        'val'    → val-timeslot-rewards-plot, …
        'best'   → best-timeslot-rewards-plot, …
    """
    p = prefix
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id=f'{p}-episode-stats-cards'), width=12)
        ], className='mt-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{p}-timeslot-rewards-plot',  config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(id=f'{p}-timeslot-failures-plot', config=PLOT_CONFIG), width=6),
        ], className='mt-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{p}-action-distribution-plot', config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(id=f'{p}-reward-tracking-plot',     config=PLOT_CONFIG), width=6),
        ], className='mt-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{p}-on-road-bikes-plot',       config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(id=f'{p}-depot-load-plot',          config=PLOT_CONFIG), width=6),
        ], className='mt-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{p}-truck-load-plot',          config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(id=f'{p}-inside-system-bikes-plot', config=PLOT_CONFIG), width=6),
        ], className='mt-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{p}-outside-system-bikes-plot', config=PLOT_CONFIG), width=6),
            dbc.Col(dcc.Graph(id=f'{p}-demand',                    config=PLOT_CONFIG), width=6),
        ], className='mt-3'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label('Select metric:', style={'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id=f'{p}-graph-metric-selector',
                        options=[
                            {'label': 'Visits (sum)',        'value': 'visits_sum'},
                            {'label': 'Operations (sum)',    'value': 'ops_sum'},
                            {'label': 'Failures (sum)',      'value': 'failure_sum'},
                            {'label': 'Failure rate',        'value': 'failure_rate'},
                            {'label': 'Critic score (mean)', 'value': 'critic_mean'},
                            {'label': 'Eligibility (mean)',  'value': 'eligibility_mean'},
                            {'label': 'Bikes (mean)',        'value': 'bikes_mean'},
                        ],
                        value='visits_sum',
                        clearable=False,
                        style={'width': '40%', 'marginBottom': '10px'}
                    ),
                    html.Div(id=f'{p}-graph-heatmap-status',
                             style={'color': '#888', 'fontSize': '13px',
                                    'fontFamily': 'monospace', 'marginBottom': '6px'}),
                    html.Img(
                        id=f'{p}-graph-heatmap-plot',
                        style={'width': '100%', 'display': 'block',
                               'border': '1px solid #d3d3d3'}
                    ),
                ], style={'padding': '10px',
                          'box-shadow': '0px 1px 3px rgba(0,0,0,0.2)',
                          'background-color': 'white'}),
            ], width=6),
        ], className='mt-3'),
    ])


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(results_path: str, data_path: str = None, port: int = 8050,
               update_interval_ms: int = 5000):
    """
    Create and configure the Dash app.

    Args:
        results_path: Path to results directory
        data_path: Path to data directory containing network.graphml
        port: Port to run server on
        update_interval_ms: Auto-refresh interval in milliseconds

    Returns:
        Configured Dash app
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,   # needed: tabs render dynamically
    )

    app.title = 'Train-Val Dashboard'

    app.results_path = Path(results_path)
    app.data_path = Path(data_path) if data_path else None
    app.port = port
    app.update_interval_ms = update_interval_ms

    app.layout = dbc.Container([
        dcc.Store(id='run-data-store'),

        create_header(),

        dcc.Interval(
            id='interval-component',
            interval=update_interval_ms,
            n_intervals=0
        ),

        create_controls(),

        # Config display
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Run Configuration'),
                    dbc.CardBody(id='config-display')
                ])
            ])
        ], className='mb-4'),

        # Dynamic tab area — rendered by the mode-tabs-container callback
        html.Div(id='mode-tabs-container'),

    ], fluid=True, style={'padding': '20px'})

    register_callbacks(app)

    return app


def main():
    """Main entry point for the webapp."""
    parser = argparse.ArgumentParser(description='BSS Results WebApp')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Path to results directory')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data directory (containing utils/network.graphml)')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run server on')
    parser.add_argument('--update-interval', type=int, default=5*60*1000,
                        help='Auto-refresh interval in milliseconds')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    app = create_app(
        results_path=args.results_path,
        data_path=args.data_path,
        port=args.port,
        update_interval_ms=args.update_interval
    )

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print(f"Starting BSS Results WebApp on http://0.0.0.0:{args.port}")
    print(f"Results path: {args.results_path}")
    if args.data_path:
        print(f"Data path: {args.data_path}")

    app.run(debug=args.debug, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()