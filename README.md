# Goku Instant Cam

A lightweight, Python-only Windows webcam effect inspired by Goku's **Instant Transmission**.

The effect is a quick snap, not a fade: a short cool-white/blue flash, a bright subject edge, one directional afterimage, then an abrupt disappearance or reappearance. No circular energy rings, orbiting sparks, or continuous aura.

## Instant Transmission visual effects

The current transition combines four lightweight OpenCV effects:
- Vertical blue-white energy streak overlays around the subject.
- Localized wave/noise displacement to make the subject area look spatially warped.
- Rapid opacity flicker during vanish and reappearance instead of a smooth fade.
- A short vertical directional blur at the teleport snap.

Additional burst effects now include an expanding shockwave ring, short electric edge sparks, two displaced afterimage echoes, and a brief RGB channel-split glitch. All are procedural OpenCV effects and require no extra AI models or heavy video assets.

## Windows / low-spec defaults

- Python 3.11 (64-bit)
- 640x480 output at 20 FPS
- AI inference at 320x240
- MediaPipe selfie segmentation, face detection and two-hand tracking
- Hand tracking every second frame; face detection every third frame
- No browser, Node.js, Electron, HTML, CSS or JavaScript
- Webcam frames are processed locally

## Important: correct OBS workflow

**Do not click "Start Virtual Camera" in OBS while Goku Instant Cam is running.**

The Python app sends its processed frames directly to the installed **OBS Virtual Camera** device using pyvirtualcam. OBS's built-in virtual camera is a single output device, so OBS cannot own that same output at the same time as the Python app. This is why starting OBS Virtual Camera first causes a conflict.

For Zoom / Google Meet / Teams:

1. Install OBS Studio once. You do not need to keep OBS open.
2. If OBS is open, make sure its **Start Virtual Camera** button is stopped. It should say "Start Virtual Camera", not "Stop Virtual Camera".
3. Close Zoom, Teams and browser tabs that may already be using the camera.
4. Run `RUN_GOKU_CAM.bat`.
5. Wait for the terminal to say `Virtual camera connected: OBS Virtual Camera`.
6. Now open Zoom, Google Meet or Teams.
7. Choose **OBS Virtual Camera** in that app's camera settings. This is the processed Goku feed from Python.

Do not select the physical webcam in Zoom. Do not start OBS's own virtual camera output.

If you want to compose scenes inside OBS itself, you need a second virtual-camera device (for example Unity Capture) for Python, then OBS can capture that separate device and output its own OBS Virtual Camera. That is a different workflow and uses more resources. For a 4 GB RAM PC, the direct Python-to-OBS-virtual-camera workflow above is recommended.

## Install and run

1. Install Python 3.11 (64-bit) and tick **Add Python to PATH**.
2. Install OBS Studio.
3. Download this repository as ZIP and extract it.
4. Double-click `RUN_GOKU_CAM.bat`. First run installs dependencies.
5. Step out of camera view for about 2 seconds to capture the empty room.
6. Return and touch your index + middle fingertips together to your forehead. Hold for about half a second.
7. Repeat the gesture to reappear.

## Super Saiyan transformation

A second, separate two-hand gesture activates a persistent Goku-inspired golden aura:

1. Keep both hands visible to the webcam.
2. Bring both wrists/hands together and cross your forearms in front of your head.
3. Quickly spread both arms outward until your hands are clearly far apart.
4. The app toggles the golden aura on. Repeat the same close-then-spread gesture to turn it off.

The aura uses the existing person segmentation mask to add a pulsing golden rim, soft glow, animated energy arcs and small rising sparks. It is a stylized webcam effect, not a full-body anime transformation. The camera needs to see both hands; good lighting and a clear background help recognition. During the gesture, the preview shows `TRANSFORM HANDS DETECTED` when both hands are being tracked, and `HANDS DETECTED - NOW SPREAD YOUR ARMS!` after the close pose is armed.

## Controls

- **B**: reset and recapture the empty-room background.
- **T**: test the teleport effect without the gesture.
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

If the PC stutters, reduce output size and frame rate:

```powershell
python goku_instant_cam.py --width 480 --height 360 --fps 15
```

## Notes

- The effect uses a captured empty-room background to hide the subject.
- Gesture recognition works best with even lighting, face and hand visible.
- This is a webcam illusion, not actual movement or scene teleportation.
- The project is a standalone Python implementation.
