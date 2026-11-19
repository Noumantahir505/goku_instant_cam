# Goku Instant Cam

A standalone Windows webcam effect for Goku-style Instant Transmission. Hold your index and middle fingertips together against your forehead to disappear. Repeat the gesture to reappear.

## What it does

- Processes webcam frames locally on your PC.
- Captures the empty room as the background when you step out of frame.
- Detects the two-finger forehead gesture using MediaPipe.
- Adds a blue-white energy aura, moving streaks and sparks during teleportation.
- Sends the processed feed to a virtual camera for Zoom, Google Meet, Microsoft Teams and other desktop apps.
- Includes a keyboard trigger for testing.

## Quick start on Windows

1. Install **Python 3.11 (64-bit)** and tick **Add Python to PATH** during installation.
2. Install **OBS Studio** if you want virtual camera output.
3. Download or clone this repository.
4. Double-click `RUN_GOKU_CAM.bat`.
5. Move out of the webcam frame briefly. The app captures the empty background automatically.
6. Return to the frame. Hold your index and middle fingertips together against your forehead for about 0.7 seconds.
7. Repeat the gesture to reappear.

In Zoom, Meet or Teams, select **OBS Virtual Camera** as your camera.

## Manual setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-goku.txt
python goku_instant_cam.py
```

## Controls

- **B**: reset and recapture the empty background. Step out of frame after pressing it.
- **T**: trigger a teleport manually for testing.
- **Q** or **Esc**: quit.

Choose another webcam if needed:

```powershell
python goku_instant_cam.py --camera 1
```

Run preview only, without virtual camera output:

```powershell
python goku_instant_cam.py --no-virtual-camera
```

## Notes

- This app processes video locally. It does not upload webcam frames.
- Gesture recognition works best with even lighting, a visible face and one hand clearly visible.
- Virtual camera output requires a compatible provider. OBS Studio's Virtual Camera is the recommended Windows option.
- The standalone Goku Instant Cam files are a new implementation and do not import the original VanishCam code.
