import dash_bootstrap_components as dbc
from dash import html, dcc

def create_layout():
    return dbc.Container([
        html.H1("Enhanced Video Dashboard", className="mb-4"),
        dcc.Tabs([
            dcc.Tab(
                label="Enhanced Videos",
                children=[
                    dbc.Row(
                        dbc.Col(
                            dcc.Dropdown(
                                id="video-dropdown",
                                options=[],
                                placeholder="Select a video",
                                clearable=False
                            ),
                            width=8
                        ),
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                "Object Detection",
                                id="obj-detection-btn",
                                color="primary",
                                className="me-2",
                                outline=True
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Brightness +",
                                id="brightness-plus-btn",
                                color="info",
                                className="me-2",
                                outline=True
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Brightness -",
                                id="brightness-minus-btn",
                                color="info",
                                className="me-2",
                                outline=True
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Reset Filters",
                                id="reset-filters-btn",
                                color="danger",
                                outline=True
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Loop Video",
                                id="loop-btn",
                                color="warning",
                                className="me-2",
                                outline=True
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Restart Video",
                                id="restart-btn",
                                color="success",
                                className="me-2",
                                outline=True
                            ),
                            width="auto"
                        )
                    ], className="mb-3"),
                    dcc.Store(id="obj-detect-store", data=False),
                    dcc.Store(id="brightness-store", data=1.0),
                    dcc.Store(id="pause-store", data=False),
                    dcc.Store(id="loop-store", data=False),
                    html.Div(id="video-container", className="mt-3"),
                    html.Div(
                        dbc.Button(
                            "Pause",
                            id="pause-btn",
                            color="secondary",
                            className="mt-2"
                        ),
                        id="pause-button-container",
                        className="text-center"
                    ),
                    html.Div(id="status-message", className="text-center mt-2"),
                    html.Div(id="active-filters-display", className="text-center mt-2")
                ]
            ),
            dcc.Tab(
                label="Live Surveillance",
                children=[
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                "Start Live Feed",
                                id="live-btn",
                                color="primary",
                                className="me-2",
                                outline=True,
                                style={
                                    'margin-top': '10px',
                                    'margin-bottom': '10px'
                                }
                            ),
                            width="auto"
                        )
                    ], className="mb-3"),
                    html.Div(id="live-feed-container")
                ]
            )
        ]),
        dcc.Interval(id="update-interval", interval=5000),
        html.Div(id="dummy-output", style={"display": "none"}),
        html.Div(id="dummy-restart-output", style={"display": "none"})
    ], fluid=True)
