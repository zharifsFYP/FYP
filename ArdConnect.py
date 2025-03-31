import os
import glob
import requests
import time
import cv2
import numpy as np
import threading
import torch
import subprocess

from Gmodel import GenerateImage
from Fmodel import InterpNet
from pipeline import preprocessingPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
ipAd = "192.168.68.115"
url = f"http://{ipAd}/capture"
polling = 1
timeout = 10
vidFps = 20
vidCodec = "avc1"
rawDir = "raw_videos"
enhDir = "enhanced_videos"
finFps = 20

os.makedirs(rawDir, exist_ok=True)
os.makedirs(enhDir, exist_ok=True)

genWeights = "generator_model.pth"
interpWeights = "best_interp_model.pth"
upscaleRes = (512, 512)
interpRes = (256, 256)
batch = 2

def convert_to_h264(inputPath, outputPath):
    command = [
        "ffmpeg", "-y", "-i", inputPath, "-c:v", "libx264",
        "-preset", "fast", "-crf", "23", outputPath
    ]
    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(res.stderr.decode())

def capture_video():
    frames = []
    finCap = None
    print("Starting continuous capture from ESP32...")
    while True:
        now = time.time()
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print("Error retrieving image:", e)
            time.sleep(polling)
            continue

        conType = resp.headers.get("Content-Type", "")
        if "image/jpeg" in conType:
            img_array = np.frombuffer(resp.content, np.uint8)
            f = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if f is not None:
                frames.append(f)
                finCap = now
                print(f"Captured f #{len(frames)}")
            else:
                print("Failed to decode image")
        else:
            print("ESP32 message:", resp.text)

        if finCap and (now - finCap) > timeout and frames:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(finCap))
            rawFile = os.path.join(rawDir, f"{timestamp}.mp4")
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*vidCodec)
            out = cv2.VideoWriter(rawFile, fourcc, vidFps, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"Saved raw video: {rawFile}")

            t = threading.Thread(target=process_raw_video, args=(rawFile,))
            t.start()

            frames = []
            finCap = None

        time.sleep(polling)

def process_raw_video(rawVidPath):
    timestamp = os.path.splitext(os.path.basename(rawVidPath))[0]
    tempFolder = os.path.join("temp_frames", timestamp)
    os.makedirs(tempFolder, exist_ok=True)

    cap = cv2.VideoCapture(rawVidPath)
    fCount = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(tempFolder, f"frame_{fCount:04d}.png"), f)
        fCount += 1
    cap.release()
    print(f"Extracted {fCount} frames from {rawVidPath}")

    outputFDir = os.path.join("enhanced_frames", timestamp)
    os.makedirs(outputFDir, exist_ok=True)

    run_pipeline(tempFolder, outputFDir, genWeights, interpWeights, upscaleRes, interpRes, batch)

    intVidPath = os.path.join(enhDir, f"intermediate_enhanced_video_{timestamp}.mp4")
    frames_to_video(outputFDir, intVidPath, fps=finFps)
    print(f"Intermediate enhanced video saved: {intVidPath}")

    finVidPath = os.path.join(enhDir, f"{timestamp}.mp4")
    convert_to_h264(intVidPath, finVidPath)
    print(f"Final enhanced video saved: {finVidPath}")

def run_pipeline(inDir, outputFDir, genWeights, interpWeights, upscaleRes, interpRes, batch=8):
    frames = load_frames_from_folder(inDir)
    if not frames:
        print("No images found in the input folder!")
        return

    gen = GenerateImage(in_channels=3, out_channels=3, feat_channels=64,
                        num_stacks=3, grow=32, kernel_size=3, upscale_factor=4)
    gen.load_state_dict(torch.load(genWeights, map_location=device))
    gen.to(device)
    gen.eval()
    upsFrames = upscale_frames_batched(frames, gen, upscaleRes, batch)

    interMod = InterpNet()
    interMod.load_state_dict(torch.load(interpWeights, map_location=device))
    interMod.to(device)
    interMod.eval()
    finFrame = interpolate_frames_batched(upsFrames, interMod, interpRes, batch)
    save_frames_to_folder(finFrame, outputFDir)

def load_frames_from_folder(folder):
    imgF = sorted(glob.glob(os.path.join(folder, "*.png")))
    images = []
    for file in imgF:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
    return images

def upscale_frames_batched(frames, gen, resTarget, batch=8):
    ups = []
    for i in range(0, len(frames), batch):
        batchF = frames[i : i + batch]
        preprocTensor = []
        for img in batchF:
            processed = preprocessingPipeline(img, resTarget[0], resTarget[1], gamma=1.0)
            tensor = torch.from_numpy(np.transpose(processed, (2, 0, 1)))
            preprocTensor.append(tensor)
        batchTensor = torch.stack(preprocTensor).to(device)
        with torch.no_grad():
            outBatch = gen(batchTensor)
        outBatch = outBatch.cpu().numpy()
        for out in outBatch:
            out = np.transpose(out, (1, 2, 0))
            out = np.clip(out, 0, 1)
            upsImg = (out * 255).astype(np.uint8)
            ups.append(upsImg)
    return ups

def prepare_interpolation_batches(upsFrames, modRes=(256,256), batch=8):
    batchList = []
    currBatch = []
    for i in range(len(upsFrames) - 1):
        fARgb = cv2.cvtColor(upsFrames[i], cv2.COLOR_BGR2RGB)
        fBRgb = cv2.cvtColor(upsFrames[i+1], cv2.COLOR_BGR2RGB)
        imgA = cv2.resize(fARgb, modRes, interpolation=cv2.INTER_LINEAR)
        imgB = cv2.resize(fBRgb, modRes, interpolation=cv2.INTER_LINEAR)
        imgA = imgA.astype(np.float32) / 255.0
        imgB = imgB.astype(np.float32) / 255.0
        tensorA = torch.from_numpy(np.transpose(imgA, (2, 0, 1)))
        tensorB = torch.from_numpy(np.transpose(imgB, (2, 0, 1)))
        exTensor = torch.full((1, modRes[1], modRes[0]), 0.5, dtype=torch.float32)
        inp = torch.cat([tensorA, tensorB, exTensor], dim=0)
        currBatch.append(inp)
        if len(currBatch) == batch:
            batchTensor = torch.stack(currBatch).to(device)
            batchList.append(batchTensor)
            currBatch = []
    if currBatch:
        batchTensor = torch.stack(currBatch).to(device)
        batchList.append(batchTensor)
    return batchList

def interpolate_frames_batched(upsFrames, interMod, modRes=(256,256), batch=8):
    batchList = prepare_interpolation_batches(upsFrames, modRes, batch)
    interpFrames = []
    for batchTensor in batchList:
        with torch.no_grad():
            preds = interMod(batchTensor)
        preds = preds.cpu().numpy()
        for pred in preds:
            pred = np.transpose(pred, (1, 2, 0))
            pred = np.clip(pred, 0, 1)
            pred = (pred * 255.0).astype(np.uint8)
            predBgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            h, w, _ = upsFrames[0].shape
            predBgr = cv2.resize(predBgr, (w, h), interpolation=cv2.INTER_LINEAR)
            interpFrames.append(predBgr)
    finFrame = []
    for i in range(len(upsFrames) - 1):
        finFrame.append(upsFrames[i])
        finFrame.append(interpFrames[i])
    finFrame.append(upsFrames[-1])
    return finFrame

def save_frames_to_folder(frames, outFold):
    os.makedirs(outFold, exist_ok=True)
    for idx, f in enumerate(frames):
        filename = os.path.join(outFold, f"frame_{idx:04d}.png")
        cv2.imwrite(filename, f)
    print(f"Saved {len(frames)} frames to folder: {outFold}")

def frames_to_video(framesFold, outVidPath, fps=30):
    frameFiles = sorted(glob.glob(os.path.join(framesFold, "frame_*.png")))
    if not frameFiles:
        print("No frames found in the folder!")
        return
    f = cv2.imread(frameFiles[0])
    height, width, _ = f.shape
    vidSize = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outVidPath, fourcc, fps, vidSize)
    for fFile in frameFiles:
        f = cv2.imread(fFile)
        if f is None:
            print(f"Warning: Unable to read {fFile}. Skipping...")
            continue
        out.write(f)
    out.release()
    print(f"Video saved as {outVidPath}")

if __name__ == "__main__":
    capture_video()
