import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import flask
import time as t
import cv2
import numpy as np
import threading
import logging
from queue import Queue
import requests

#Import 
from layout import create_layout

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#constants and directories
ESP32_IP = "Insert IP here"
SURVEILLANCE_URL = f"http://{ESP32_IP}/surCapture"
VID_DIR = "enhanced_videos"
os.makedirs(VID_DIR, exist_ok=True)

#flask server and dash app
flaskSrv = flask.Flask(__name__)
dashApp = dash.Dash(
    __name__, 
    server=flaskSrv,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

#layout
dashApp.layout = create_layout()

#global states for video streaming
class VidStreamSt:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.paused = False
        self.currVid = None
        self.fQueue = Queue(maxsize=10)
        # Background subtractor for object detection
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.vidElement = None
        self.vidId = "video-element"
        self.brightLvl = 1.0
        # Maintain original keys for dash ID references
        self.activeFltrs = {
            'object_detection': False,
            'brightness': False
        }
        self.loopVid = False
        self.restartRequested = False
        self.processingThread = None
        self.playbackEnded = False
        self.filterUpdateNeeded = False
        # Live feed references
        self.liveActive = False
        self.liveThread = None
        self.liveFQueue = Queue(maxsize=2)

vidSt = VidStreamSt()

######### video processing helpers (used by both vid processing & live feed)

def applyBrightAdjustment(fData, bright):
    #mods brightness by scaling V channel 
    if bright ==1.0:
        return fData
    hsv = cv2.cvtColor(fData, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, np.array([bright]))
    v = np.clip(v, 0, 255).astype('uint8')
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def dtctObjects(fData):
    try:
        grayF = cv2.cvtColor(fData, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayF, (5, 5), 0)
        fgMask = vidSt.bgSubtractor.apply(blurred)
        _, thresh = cv2.threshold(fgMask, 25, 255, cv2.THRESH_BINARY)
        
        # Increased dilation size and iterations to enlarge the binary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # larger kernel
        dilated = cv2.dilate(thresh, kernel, iterations=4)  # increased iterations
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 20000:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(fData, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return fData
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return fData


def applyFltrs(fData):
    #applies brightness and/or object detection filters if active
    if vidSt.activeFltrs['brightness']:
        fData = applyBrightAdjustment(fData, vidSt.brightLvl)
    if vidSt.activeFltrs['object_detection']:
        fData = dtctObjects(fData)
    return fData


############################# vidProcessingThread + dspVid callback

def vidProcessingThread(fileName):
    # Continuously reads frames from a specified video file, applies filters, streams them out
    try:
        vidPath = os.path.join(VID_DIR, fileName)
        while vidSt.active and vidSt.currVid == fileName:
            try:
                cap = cv2.VideoCapture(vidPath)
                if not cap.isOpened():
                    logger.error(f"Failed to open vid: {fileName}")
                    break

                fps =cap.get(cv2.CAP_PROP_FPS)
                fDelay = 1.0/ fps if fps > 0 else 0.033
                vidSt.playbackEnded = False

                # Main loop for reading frames
                while vidSt.active and vidSt.currVid == fileName:
                    if vidSt.restartRequested:
                        # Restart from beginning if requested
                        with vidSt.lock:
                            vidSt.restartRequested = False
                            vidSt.playbackEnded = False
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        with vidSt.fQueue.mutex:
                            vidSt.fQueue.queue.clear()
                        continue
                        
                    if not vidSt.paused:
                        strtTime = t.time()
                        ret, rawF = cap.read()
                        
                        if not ret:
                            #reached video end
                            vidSt.playbackEnded = True
                            if vidSt.loopVid:
                                #loop to start if needed
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                t.sleep(0.1)
                                continue

                        processedF = applyFltrs(rawF)
                        _, jpeg = cv2.imencode('.jpg', processedF)
                        
                        #Keep queue from blocking if full
                        if vidSt.fQueue.full():
                            try:
                                vidSt.fQueue.get_nowait()
                            except:
                                pass
                        vidSt.fQueue.put(jpeg.tobytes())
                        elapsed = t.time() - strtTime
                        t.sleep(max(0, fDelay - elapsed))
                    else:
                        
                        t.sleep(0.1)
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            
            #If not looping, end after single read
            if not vidSt.loopVid:
                break

    except Exception as e:
        logger.error(f"Vid processing error: {e}")
    finally:
        #Cleanup 
        with vidSt.lock:
            if vidSt.currVid == fileName:
                vidSt.active = False
        logger.info(f"Stopped processing: {fileName}")

@dashApp.callback(
    [Output("video-container", "children"),
     Output("pause-button-container", "style")],
    [Input("video-dropdown", "value"),
     Input("obj-detect-store", "data"),
     Input("brightness-store", "data")],
    prevent_initial_call=True
)
def dspVid(selectedFile, objDtctEnabled, brightLvl):
    #decide whether to display a raw <video> or the filtered feed
    if not selectedFile:
        return dbc.Alert("Please select a video", color="secondary"), {'display': 'none'}

    with vidSt.lock:
        #update filter states
        vidSt.activeFltrs['object_detection'] = objDtctEnabled
        vidSt.activeFltrs['brightness'] = (brightLvl != 1.0)
        vidSt.brightLvl =brightLvl

        needsRestart =(
            (vidSt.currVid != selectedFile) or
            (vidSt.vidElement == "unfiltered" and (objDtctEnabled or brightLvl != 1.0)) or
            (vidSt.vidElement == "filtered" and not objDtctEnabled and brightLvl == 1.0))

        if needsRestart:
            # Stop current thread if active
            if vidSt.active and vidSt.processingThread:
                vidSt.active = False
                vidSt.processingThread.join(timeout=0.5)
            

            # If filters are enabled, switch to a processed feed
            if objDtctEnabled or brightLvl != 1.0:
                vidSt.active = True
                vidSt.currVid = selectedFile
                vidSt.vidElement = "filtered"
                # Re-init background subtractor for object detection
                vidSt.bgSubtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
                
                vidSt.processingThread = threading.Thread(
                    target=vidProcessingThread,
                    args=(selectedFile,),
                    name=f"vid_thread_{selectedFile}"
                )
                vidSt.processingThread.daemon = True
                vidSt.processingThread.start()

                return (
                    html.Div([
                        html.Img(
                            src=f"/filtered_feed/{selectedFile}", 
                            style={'width': '100%'},
                            id=vidSt.vidId
                        )]),
                    {'display': 'block'}
                )
            else:
                #Use a raw if no filters are active
                vidSt.active = False
                vidSt.vidElement = "unfiltered"
                return (
                    html.Video(
                        src=f"/videos/{selectedFile}",
                        controls=True,
                        id=vidSt.vidId,
                        style={'width': '100%'}
                    ),
                    {'display': 'block'}
                )
        else:
            #No need to restart processing; just adjust current feed if curerent filtered
            if vidSt.vidElement == "filtered":
                return (
                    html.Div([
                        html.Img(
                            src=f"/filtered_feed/{selectedFile}", 
                            style={'width': '100%'},
                            id=vidSt.vidId
                        )
                        ]),
                    {'display': 'block'}
                )
            else:
                return (
                    html.Video(
                        src=f"/videos/{selectedFile}",
                         controls=True,
                        id=vidSt.vidId,
                        style={'width': '100%'}
                    ),
                    {'display': 'block'}
                )

######################### liveFeedThread + tggLiveFeed callback

def liveFeedThread():
    #Continuously fetc frames from SURVEILLANCE_URL
    while vidSt.liveActive:
        try:
            resp=requests.get(SURVEILLANCE_URL, stream=True, timeout=5)
            if resp.status_code == 200:
                imgArr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                rawF = cv2.imdecode(imgArr, cv2.IMREAD_COLOR)
                
                if rawF is not None:
                    # Apply filters if active
                    if vidSt.activeFltrs['brightness']:
                        rawF= applyBrightAdjustment(rawF, vidSt.brightLvl)
                    if vidSt.activeFltrs['object_detection']:
                        rawF = dtctObjects(rawF)
                    
                    _, jpeg = cv2.imencode('.jpg', rawF)
                    
                    # Use limited buffer queue
                    if vidSt.liveFQueue.full():
                        vidSt.liveFQueue.get_nowait()
                    vidSt.liveFQueue.put(jpeg.tobytes())
            
            # Attempt ~20 fps
            t.sleep(0.05)
        except Exception as e:
            logger.error(f"Live feed error: {e}")
            t.sleep(0.5)

@dashApp.callback(
    [Output("live-feed-container", "children"),
     Output("live-btn", "children"),
     Output("live-btn", "color")],
    [Input("live-btn", "n_clicks")],
    prevent_initial_call=True
)
def tggLiveFeed(nClks):
    # Toggle live feed from ESP32
    if nClks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    if nClks % 2 ==1:  #start live feed 
        vidSt.liveActive = True
        vidSt.liveThread = threading.Thread(
            target=liveFeedThread,
            daemon=True
        )
        vidSt.liveThread.start()
        
        return (
            html.Img(
                src="/live_feed",
                style={"width": "100%"},
                id="live-feed"
            ),
            "Stop Live Feed",
            "danger"
        )
    else:  #stop feed
        vidSt.liveActive =False
        if vidSt.liveThread:
            vidSt.liveThread.join(timeout=1)
        return (
            dbc.Alert("Live feed stopped", color="secondary"),
            "Start Live Feed",
            "primary"
        )

############################## Callbacks ##############################################


@dashApp.callback(
    Output("video-dropdown", "options"),
    Input("update-interval", "n_intervals")
)
def updVidList(nIntr):
    #refresh the list of MP4 files for the dropdown
    try:
        allVids = [f for f in os.listdir(VID_DIR) if f.endswith('.mp4')]
        return [{'label': f, 'value': f} for f in sorted(allVids)]
    except Exception as e:
        logger.error(f"Error listing vids: {e}")
        return []

@dashApp.callback(
    [Output("obj-detect-store", "data"),
     Output("obj-detection-btn", "outline")],
    [Input("obj-detection-btn", "n_clicks")],
    [State("obj-detect-store", "data")],
    prevent_initial_call=True
)
def tggObjDtct(nClks, currSt):
    #toggle object detection on/off
    if nClks is None:
        return currSt, True
    newSt = not currSt
    return newSt, not newSt

@dashApp.callback(
    [Output("brightness-store", "data"),
     Output("brightness-plus-btn", "outline"),
     Output("brightness-minus-btn", "outline")],
    [Input("brightness-plus-btn", "n_clicks"),
     Input("brightness-minus-btn", "n_clicks"),
     Input("reset-filters-btn", "n_clicks")],
    [State("brightness-store", "data")],
    prevent_initial_call=True
)
def handleBrightAdjust(plusClks, minusClks, resetClks, currLvl):
    #manage brightness-level adjustments and reset
    cbCtx = callback_context
    if not cbCtx.triggered:
        return currLvl, True, True
    
    trgId = cbCtx.triggered[0]['prop_id'].split('.')[0]
    
    if trgId == "reset-filters-btn":
        return 1.0, True, True
    elif trgId == "brightness-plus-btn":
        newLvl = min(2.0, currLvl + 0.2)
        return newLvl, (newLvl == 1.0), (newLvl == 1.0)
    elif trgId == "brightness-minus-btn":
        newLvl = max(0.2, currLvl - 0.2)
        return newLvl, (newLvl == 1.0), (newLvl == 1.0)
    
    return currLvl, True, True

@dashApp.callback(
    [Output("loop-store", "data"),
     Output("loop-btn", "outline")],
    [Input("loop-btn", "n_clicks")],
    [State("loop-store", "data")],
    prevent_initial_call=True
)
def tggLoop(nClks, currSt):
    #toggle looping of  video
    if nClks is None:
        return currSt, True
    
    newSt = not currSt
    with vidSt.lock:
        vidSt.loopVid = newSt
    return newSt, not newSt

@dashApp.callback(
    Output("active-filters-display", "children"),
    [Input("obj-detect-store", "data"),
     Input("brightness-store", "data"),
     Input("loop-store", "data")],
    prevent_initial_call=True
)
def updActiveFltrsDisplay(objDtctEn, brightLvl, loopEn):
    #\ list of which filters are active
    active = []
    if objDtctEn:
        active.append("Object Detection")
    if brightLvl != 1.0:
        active.append(f"Brightness: {brightLvl*100:.0f}%")
    if loopEn:
        active.append("Loop Enabled")
    
    if not active:
        return "No active filters"
    
    return html.Div([
        html.Strong("Active Filters: "),
        ", ".join(active)
    ])

@dashApp.callback(
    [Output("pause-store", "data"),
     Output("pause-btn", "children"),
     Output("status-message", "children")],
    [Input("pause-btn", "n_clicks"),
     Input("video-dropdown", "value")],
    [State("pause-store", "data")],
    prevent_initial_call=True
)
def tggPause(nClks, selectedFile, currSt):
    #pause/unpause the video or auto-resume on video change
    cbCtx = callback_context
    if not cbCtx.triggered:
        return currSt, "Pause", ""
    
    trgId = cbCtx.triggered[0]['prop_id'].split('.')[0]
    
    if trgId == "video-dropdown" and selectedFile:
        with vidSt.lock:
            vidSt.paused = False
        return False, "Pause", "Video changed - playback resumed"
    if nClks is None:
        return currSt, "Pause", ""
    newSt = not currSt
    with vidSt.lock:
        vidSt.paused = newSt
    
    btnTxt = "Resume" if newSt else "Pause"
    statusTxt = "Video paused" if newSt else "Video resumed"
    
    return newSt, btnTxt, statusTxt

@dashApp.callback(
    Output("dummy-restart-output", "children"),
    [Input("restart-btn", "n_clicks")],
    [State("video-dropdown", "value"),
     State("obj-detect-store", "data"),
     State("brightness-store", "data")],
    prevent_initial_call=True
)
def rstrtVid(nClks, currFile, objDtct, brLvl):
    #Restart the current video from the begin
    if nClks is None or not currFile:
        return ""
    
    with vidSt.lock:
        if vidSt.currVid == currFile:
            #video ended and filters are active, re-init processing if needed
            if vidSt.playbackEnded and (objDtct or brLvl != 1.0):
                if not vidSt.active:
                    vidSt.active = True
                    vidSt.processingThread = threading.Thread(
                        target=vidProcessingThread,
                        args=(currFile,),
                        name=f"vid_thread_{currFile}"
                    )
                    vidSt.processingThread.daemon = True
                    vidSt.processingThread.start()
            
            vidSt.restartRequested = True
            vidSt.playbackEnded = False
            # Clears any remaining frames
            with vidSt.fQueue.mutex:
                vidSt.fQueue.queue.clear()
    return ""

#clientside callback to pause/resume the unfiltered element
dashApp.clientside_callback(
    """
    function(is_paused) {
        const vidEl = document.getElementById("video-element");
        if (vidEl) {
            if (is_paused) {
                if (vidEl.pause) vidEl.pause();
            } else {
                if (vidEl.play) vidEl.play();
            }
        }
        return '';
    }
    """,
    Output("dummy-output", "children"),
    [Input("pause-store", "data")],
    prevent_initial_call=True
)
####################flask routes to serve video files#################

@flaskSrv.route('/videos/<path:fileName>')
def serveVid(fileName):
    return flask.send_from_directory(VID_DIR, fileName)

@flaskSrv.route('/filtered_feed/<fileName>')
def filteredFeed(fileName):
    #streams processed frames for the 'filtered' feed
    def generate():
        try:
            while vidSt.active and vidSt.currVid == fileName:
                if not vidSt.fQueue.empty():
                    fData = vidSt.fQueue.get()
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + fData + b'\r\n'
                    )
                else:
                    t.sleep(0.01)
        except Exception as e:
            logger.error(f"Stream generation error: {e}")

    return flask.Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@flaskSrv.route('/live_feed')
def liveFeed():
    # Streams frames from the live feed
    def generate():
        while vidSt.liveActive:
            if not vidSt.liveFQueue.empty():
                fData = vidSt.liveFQueue.get()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + fData + b'\r\n'
                )
            else:
                t.sleep(0.01)
    return flask.Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

#start app
if __name__ == '__main__':
    dashApp.run(host='0.0.0.0', port=5000, debug=True)
