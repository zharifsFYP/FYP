import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import flask
import time
import cv2
import numpy as np
import threading
import logging
from queue import Queue
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ESP32_IP = "192.168.68.115"
SURVEILLANCE_URL = f"http://{ESP32_IP}/surCapture"
VIDEO_DIR = "enhanced_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Create Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, 
               external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

# Global state for video streaming
class VideoStreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.paused = False
        self.current_video = None
        self.frame_queue = Queue(maxsize=10)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.video_element = None
        self.video_id = "video-element"
        self.brightness_level = 1.0
        self.active_filters = {
            'object_detection': False,
            'brightness': False
        }
        self.loop_video = False
        self.restart_requested = False
        self.processing_thread = None
        self.playback_ended = False
        self.filter_update_needed = False
        self.live_active = False
        self.live_thread = None
        self.live_frame_queue = Queue(maxsize=2)

video_state = VideoStreamState()

# Flask Routes
@server.route('/videos/<path:filename>')
def serve_video(filename):
    return flask.send_from_directory(VIDEO_DIR, filename)

@server.route('/filtered_feed/<filename>')
def filtered_feed(filename):
    def generate():
        try:
            while video_state.active and video_state.current_video == filename:
                if not video_state.frame_queue.empty():
                    frame = video_state.frame_queue.get()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    time.sleep(0.01)
        except Exception as e:
            logger.error(f"Stream generation error: {e}")

    return flask.Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@server.route('/live_feed')
def live_feed():
    def generate():
        while video_state.live_active:
            if not video_state.live_frame_queue.empty():
                frame = video_state.live_frame_queue.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
                
    return flask.Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Video Processing Functions
def apply_brightness_adjustment(frame, brightness):
    if brightness == 1.0:
        return frame
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, np.array([brightness]))
    v = np.clip(v, 0, 255).astype('uint8')
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def detect_objects(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        fg_mask = video_state.bg_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return frame

def apply_filters(frame):
    if video_state.active_filters['brightness']:
        frame = apply_brightness_adjustment(frame, video_state.brightness_level)
    if video_state.active_filters['object_detection']:
        frame = detect_objects(frame)
    return frame

def video_processing_thread(filename):
    try:
        video_path = os.path.join(VIDEO_DIR, filename)
        
        while video_state.active and video_state.current_video == filename:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Failed to open video: {filename}")
                    break

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = 1.0 / fps if fps > 0 else 0.033
                video_state.playback_ended = False

                while video_state.active and video_state.current_video == filename:
                    if video_state.restart_requested:
                        with video_state.lock:
                            video_state.restart_requested = False
                            video_state.playback_ended = False
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        with video_state.frame_queue.mutex:
                            video_state.frame_queue.queue.clear()
                        continue
                        
                    if not video_state.paused:
                        start_time = time.time()
                        ret, frame = cap.read()
                        
                        if not ret:
                            video_state.playback_ended = True
                            if video_state.loop_video:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                time.sleep(0.1)
                                continue

                        processed_frame = apply_filters(frame)
                        _, jpeg = cv2.imencode('.jpg', processed_frame)
                        
                        if video_state.frame_queue.full():
                            try:
                                video_state.frame_queue.get_nowait()
                            except:
                                pass
                        
                        video_state.frame_queue.put(jpeg.tobytes())

                        elapsed = time.time() - start_time
                        time.sleep(max(0, frame_delay - elapsed))
                    else:
                        time.sleep(0.1)
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            
            if not video_state.loop_video:
                break

    except Exception as e:
        logger.error(f"Video processing error: {e}")
    finally:
        with video_state.lock:
            if video_state.current_video == filename:
                video_state.active = False
        logger.info(f"Stopped processing: {filename}")

def live_feed_thread():
    while video_state.live_active:
        try:
            # Use requests to get the frame instead of cv2.VideoCapture
            response = requests.get(SURVEILLANCE_URL, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Apply filters if needed
                    if video_state.active_filters['brightness']:
                        frame = apply_brightness_adjustment(frame, video_state.brightness_level)
                    if video_state.active_filters['object_detection']:
                        frame = detect_objects(frame)
                    
                    # Convert to JPEG
                    _, jpeg = cv2.imencode('.jpg', frame)
                    
                    # Put frame in queue
                    if video_state.live_frame_queue.full():
                        video_state.live_frame_queue.get_nowait()
                    video_state.live_frame_queue.put(jpeg.tobytes())
            
            time.sleep(0.05)  # ~20fps
            
        except Exception as e:
            logger.error(f"Live feed error: {e}")
            time.sleep(0.5)

# Dashboard Layout
app.layout = dbc.Container([
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
                dbc.Row([  # Wrap button in a Row for consistent spacing
                    dbc.Col(
                        dbc.Button(
                            "Start Live Feed",
                            id="live-btn",
                            color="primary",
                            className="me-2",
                            outline=True,  # Add outline to match other buttons
                            style={
                                'margin-top': '10px',  # Match vertical spacing
                                'margin-bottom': '10px'
                            }
                        ),
                        width="auto"  # Match other button columns
                    )
                ], className="mb-3"),  # Same bottom margin as other button rows
                html.Div(id="live-feed-container")
            ]
        )
    ]),
    dcc.Interval(id="update-interval", interval=5000),
    html.Div(id="dummy-output", style={"display": "none"}),
    html.Div(id="dummy-restart-output", style={"display": "none"})
], fluid=True)

# Dash Callbacks
@app.callback(
    Output("video-dropdown", "options"),
    Input("update-interval", "n_intervals")
)
def update_video_list(n):
    try:
        videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
        return [{'label': f, 'value': f} for f in sorted(videos)]
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return []

@app.callback(
    [Output("video-container", "children"),
     Output("pause-button-container", "style")],
    [Input("video-dropdown", "value"),
     Input("obj-detect-store", "data"),
     Input("brightness-store", "data")],
    prevent_initial_call=True
)
def display_video(filename, obj_detect_enabled, brightness_level):
    if not filename:
        return dbc.Alert("Please select a video", color="secondary"), {'display': 'none'}

    with video_state.lock:
        # Update filter states
        video_state.active_filters['object_detection'] = obj_detect_enabled
        video_state.active_filters['brightness'] = brightness_level != 1.0
        video_state.brightness_level = brightness_level

        # Only restart processing if switching videos or changing to/from filtered mode
        needs_restart = (
            (video_state.current_video != filename) or
            (video_state.video_element == "unfiltered" and (obj_detect_enabled or brightness_level != 1.0)) or
            (video_state.video_element == "filtered" and not obj_detect_enabled and brightness_level == 1.0)
        )

        if needs_restart:
            if video_state.active and video_state.processing_thread:
                video_state.active = False
                video_state.processing_thread.join(timeout=0.5)
            
            if obj_detect_enabled or brightness_level != 1.0:
                video_state.active = True
                video_state.current_video = filename
                video_state.video_element = "filtered"
                video_state.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
                
                video_state.processing_thread = threading.Thread(
                    target=video_processing_thread,
                    args=(filename,),
                    name=f"video_thread_{filename}"
                )
                video_state.processing_thread.daemon = True
                video_state.processing_thread.start()

                return html.Div([
                    html.Img(src=f"/filtered_feed/{filename}", 
                            style={'width': '100%'},
                            id=video_state.video_id)
                ]), {'display': 'block'}
            else:
                video_state.active = False
                video_state.video_element = "unfiltered"
                return html.Video(
                    src=f"/videos/{filename}",
                    controls=True,
                    id=video_state.video_id,
                    style={'width': '100%'}
                ), {'display': 'block'}
        else:
            # Just update the filters without restarting
            if video_state.video_element == "filtered":
                return html.Div([
                    html.Img(src=f"/filtered_feed/{filename}", 
                            style={'width': '100%'},
                            id=video_state.video_id)
                ]), {'display': 'block'}
            else:
                return html.Video(
                    src=f"/videos/{filename}",
                    controls=True,
                    id=video_state.video_id,
                    style={'width': '100%'}
                ), {'display': 'block'}

@app.callback(
    [Output("obj-detect-store", "data"),
     Output("obj-detection-btn", "outline")],
    [Input("obj-detection-btn", "n_clicks")],
    [State("obj-detect-store", "data")],
    prevent_initial_call=True
)
def toggle_object_detection(n_clicks, current_state):
    if n_clicks is None:
        return current_state, True
    new_state = not current_state
    return new_state, not new_state

@app.callback(
    [Output("brightness-store", "data"),
     Output("brightness-plus-btn", "outline"),
     Output("brightness-minus-btn", "outline")],
    [Input("brightness-plus-btn", "n_clicks"),
     Input("brightness-minus-btn", "n_clicks"),
     Input("reset-filters-btn", "n_clicks")],
    [State("brightness-store", "data")],
    prevent_initial_call=True
)
def handle_brightness_adjustment(plus_clicks, minus_clicks, reset_clicks, current_level):
    ctx = callback_context
    if not ctx.triggered:
        return current_level, True, True
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == "reset-filters-btn":
        return 1.0, True, True
    elif triggered_id == "brightness-plus-btn":
        new_level = min(2.0, current_level + 0.2)
        return new_level, new_level == 1.0, new_level == 1.0
    elif triggered_id == "brightness-minus-btn":
        new_level = max(0.2, current_level - 0.2)
        return new_level, new_level == 1.0, new_level == 1.0
    
    return current_level, True, True

@app.callback(
    [Output("loop-store", "data"),
     Output("loop-btn", "outline")],
    [Input("loop-btn", "n_clicks")],
    [State("loop-store", "data")],
    prevent_initial_call=True
)
def toggle_loop(n_clicks, current_state):
    if n_clicks is None:
        return current_state, True
    
    new_state = not current_state
    with video_state.lock:
        video_state.loop_video = new_state
    return new_state, not new_state

@app.callback(
    Output("active-filters-display", "children"),
    [Input("obj-detect-store", "data"),
     Input("brightness-store", "data"),
     Input("loop-store", "data")],
    prevent_initial_call=True
)
def update_active_filters_display(obj_detect_enabled, brightness_level, loop_enabled):
    active_filters = []
    if obj_detect_enabled:
        active_filters.append("Object Detection")
    if brightness_level != 1.0:
        active_filters.append(f"Brightness: {brightness_level*100:.0f}%")
    if loop_enabled:
        active_filters.append("Loop Enabled")
    
    if not active_filters:
        return "No active filters"
    
    return html.Div([
        html.Strong("Active Filters: "),
        ", ".join(active_filters)
    ])

@app.callback(
    [Output("pause-store", "data"),
     Output("pause-btn", "children"),
     Output("status-message", "children")],
    [Input("pause-btn", "n_clicks"),
     Input("video-dropdown", "value")],
    [State("pause-store", "data")],
    prevent_initial_call=True
)
def toggle_pause(n_clicks, filename, current_state):
    ctx = callback_context
    if not ctx.triggered:
        return current_state, "Pause", ""
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == "video-dropdown" and filename:
        with video_state.lock:
            video_state.paused = False
        return False, "Pause", "Video changed - playback resumed"
    
    if n_clicks is None:
        return current_state, "Pause", ""
    
    new_state = not current_state
    with video_state.lock:
        video_state.paused = new_state
    
    btn_text = "Resume" if new_state else "Pause"
    status_msg = "Video paused" if new_state else "Video resumed"
    
    return new_state, btn_text, status_msg

@app.callback(
    Output("dummy-restart-output", "children"),
    [Input("restart-btn", "n_clicks")],
    [State("video-dropdown", "value"),
     State("obj-detect-store", "data"),
     State("brightness-store", "data")],
    prevent_initial_call=True
)
def restart_video(n_clicks, current_video, obj_detect_enabled, brightness_level):
    if n_clicks is None or not current_video:
        return ""
    
    with video_state.lock:
        if video_state.current_video == current_video:
            # If playback ended and we're in filtered mode, reactivate
            if video_state.playback_ended and (obj_detect_enabled or brightness_level != 1.0):
                if not video_state.active:
                    video_state.active = True
                    video_state.processing_thread = threading.Thread(
                        target=video_processing_thread,
                        args=(current_video,),
                        name=f"video_thread_{current_video}"
                    )
                    video_state.processing_thread.daemon = True
                    video_state.processing_thread.start()
            
            video_state.restart_requested = True
            video_state.playback_ended = False
            # Clear the frame queue immediately
            with video_state.frame_queue.mutex:
                video_state.frame_queue.queue.clear()
    
    return ""

@app.callback(
    [Output("live-feed-container", "children"),
     Output("live-btn", "children"),
     Output("live-btn", "color")],
    [Input("live-btn", "n_clicks")],
    prevent_initial_call=True
)
def toggle_live_feed(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    if n_clicks % 2 == 1:  # Start live feed
        video_state.live_active = True
        video_state.live_thread = threading.Thread(
            target=live_feed_thread,
            daemon=True
        )
        video_state.live_thread.start()
        
        return (
            html.Img(
                src="/live_feed",
                style={"width": "100%"},
                id="live-feed"
            ),
            "Stop Live Feed",
            "danger"
        )
    else:  # Stop live feed
        video_state.live_active = False
        if video_state.live_thread:
            video_state.live_thread.join(timeout=1)
        return (
            dbc.Alert("Live feed stopped", color="secondary"),
            "Start Live Feed",
            "primary"
        )

app.clientside_callback(
    """
    function(is_paused) {
        const video = document.getElementById("video-element");
        if (video) {
            if (is_paused) {
                if (video.pause) video.pause();
            } else {
                if (video.play) video.play();
            }
        }
        return '';
    }
    """,
    Output("dummy-output", "children"),
    [Input("pause-store", "data")],
    prevent_initial_call=True
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)