# Goku Instant Cam

A lightweight, Python-only Windows virtual webcam effect inspired by Goku's **Instant Transmission**.

The visual is intentionally restrained and close to the technique: the two-finger forehead gesture, a very brief cool-white/blue flash, a narrow afterimage/speed smear, and the subject disappearing or reappearing. **There are no circular energy rings, orbiting sparks, or continuous aura effects.**

## Designed for a low-spec PC

- Python 3.11, 64-bit
- 640x480 webcam input/output by default
- AI inference at 320x240
- MediaPipe Selfie Segmentation + lightweight Face Detection + one-hand tracking
- Hand landmarks are processed every second frame; face detection every third frame
- No browser, Node.js, Electron, HTML, CSS, or JavaScript
- Webcam frames are processed locally

## Windows setup

1. Install Python 3.11 (64-bit). Enable **Add Python to PATH**.
2. Install OBS Studio.
3. Download this repository as a ZIP and extract it.
4. Open OBS and click **Start Virtual Camera**.
5. Double-click `RUN_GOKU_CAM.bat`. The first run installs Python packages and may take a few minutes.
6. In the Goku Instant Cam preview, step out of camera view for about 2 seconds. The empty room is captured as the background.
7. Return to the camera. Bring your index and middle fingertips together and touch them to your forehead. Hold for about half a second.
8. Repeat the same gesture to reappear.

In Zoom, Google Meet, or Microsoft Teams, choose **OBS Virtual Camera** as the camera.

## Controls

- **B**: recapture the empty-room background. Step out of frame afterward.
- **T**: test the disappear/reappear effect without making the gesture.
- **Q** or **Esc**: quit.

## Manual setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-goku.txt
python goku_instant_cam.py
```

Preview only, without virtual camera output:

```powershell
python goku_instant_cam.py --no-virtual-camera
```

If your webcam is not camera 0:

```powershell
python goku_instant_cam.py --camera 1
```

## Performance notes

The defaults are deliberately conservative for a 4 GB RAM computer. Close unnecessary apps before starting a meeting. If the preview stutters, reduce output size to 480x360 and frame rate to 15:

```powershell
python goku_instant_cam.py --width 480 --height 360 --fps 15
```

Virtual camera output requires OBS Studio's Virtual Camera to be started before launching this app. If the app says the virtual camera is unavailable, the preview still runs; start OBS Virtual Camera and restart Goku Instant Cam.

## Notes

- The effect is a webcam illusion. It uses the captured empty-room background to hide the subject when they teleport.
- Gesture detection works best with the face and one hand clearly visible, even lighting, and the hand not covering the whole face.
- This repository contains a standalone implementation and does not import the previous VanishCam project.
