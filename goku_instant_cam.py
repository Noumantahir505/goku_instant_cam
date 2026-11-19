"""
Goku Instant Cam
A standalone webcam effect: hold your index and middle fingertips together
against your forehead to teleport out; repeat to teleport back in.

Original implementation. Uses MediaPipe for segmentation and hand/face landmarks.
"""
import argparse
import time
import sys

import cv2
import mediapipe as mp
import numpy as np

WINDOW = "Goku Instant Cam"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 24
GESTURE_HOLD_SECONDS = 0.65
TRANSITION_SECONDS = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description="Goku-style instant transmission virtual camera")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--no-virtual-camera", action="store_true",
                        help="Run the preview only, without virtual camera output")
    return parser.parse_args()


def make_mask(segmentation, shape):
    h, w = shape[:2]
    mask = cv2.resize(segmentation, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip((mask - 0.12) / 0.72, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    return np.clip(mask, 0.0, 1.0)


def detect_forehead_gesture(hand_result, face_result):
    """Detect two close fingertips held near the upper forehead."""
    if not hand_result.multi_hand_landmarks or not face_result.multi_face_landmarks:
        return False

    hand = hand_result.multi_hand_landmarks[0].landmark
    face = face_result.multi_face_landmarks[0].landmark

    index_tip = np.array([hand[8].x, hand[8].y])
    middle_tip = np.array([hand[12].x, hand[12].y])
    forehead = np.array([face[10].x, face[10].y + 0.055])

    finger_gap = float(np.linalg.norm(index_tip - middle_tip))
    index_distance = float(np.linalg.norm(index_tip - forehead))
    middle_distance = float(np.linalg.norm(middle_tip - forehead))

    fingers_together = finger_gap < 0.075
    touching_forehead = index_distance < 0.17 and middle_distance < 0.19
    return fingers_together and touching_forehead


def draw_aura(image, mask, progress, returning):
    """Draw an animated blue-white energy ring and streaks around the subject."""
    if progress <= 0.02:
        return image

    ys, xs = np.where(mask > 0.25)
    if len(xs) < 30:
        return image

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(45, int((x2 - x1) * (0.62 + 0.10 * np.sin(time.time() * 9))))
    ry = max(55, int((y2 - y1) * 0.68))

    glow = image.copy()
    tick = time.time()
    pulse = 0.5 + 0.5 * np.sin(tick * 11)
    color = (255, 220, 120)

    for layer in range(4):
        scale = 1.0 + layer * 0.12 + pulse * 0.025
        cv2.ellipse(
            glow, (cx, cy), (int(rx * scale), int(ry * scale)),
            int((tick * 95 + layer * 35) % 180), 205, 335,
            color, max(1, 5 - layer), cv2.LINE_AA
        )

    # Energy streaks move upward during disappearance and downward on return.
    direction = -1 if not returning else 1
    count = 24
    for i in range(count):
        phase = (tick * (0.65 + (i % 5) * 0.08) + i / count) % 1.0
        x = int(x1 + (x2 - x1) * ((i * 0.618) % 1.0))
        y = int(y2 - phase * (y2 - y1)) if direction < 0 else int(y1 + phase * (y2 - y1))
        length = int(14 + 36 * progress)
        cv2.line(glow, (x, y), (x + int(np.sin(i * 4) * 5), y + direction * length),
                 (255, 245, 205), 1 + (i % 2), cv2.LINE_AA)

    # Small sparks orbit the energy ring.
    for i in range(18):
        angle = tick * 2.2 + i * (2 * np.pi / 18)
        px = int(cx + np.cos(angle) * rx * 1.08)
        py = int(cy + np.sin(angle) * ry * 1.08)
        radius = 1 + (i % 3)
        cv2.circle(glow, (px, py), radius, (255, 255, 230), -1, cv2.LINE_AA)

    return cv2.addWeighted(glow, 0.72, image, 0.28, 0)


def main():
    args = parse_args()
    width, height, fps = args.width, args.height, args.fps

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}. Try --camera 1.")

    mp_selfie = mp.solutions.selfie_segmentation
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    virtual_cam = None
    if not args.no_virtual_camera:
        try:
            import pyvirtualcam
            virtual_cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            print(f"Virtual camera ready: {virtual_cam.device}")
        except Exception as exc:
            print("Virtual camera unavailable. Preview will still work.")
            print(f"Reason: {exc}")
            print("Install OBS Studio and enable OBS Virtual Camera, then restart.")

    background = None
    empty_frames = []
    empty_count = 0
    visible = True
    transition_start = None
    transition_from_visible = True
    gesture_start = None
    gesture_cooldown = 0.0
    last_status = time.time()

    print("\nGOKU INSTANT CAM")
    print("1. Move out of the camera view for a moment so the background can be captured.")
    print("2. Hold your index + middle fingertips together against your forehead.")
    print("3. Repeat the gesture to reappear.")
    print("Keys: B = recapture background | T = test teleport | Q/ESC = quit\n")

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter, \
         mp_hands.Hands(max_num_hands=1, model_complexity=0,
                        min_detection_confidence=0.55, min_tracking_confidence=0.5) as hands, \
         mp_face.FaceMesh(max_num_faces=1, refine_landmarks=False,
                          min_detection_confidence=0.55, min_tracking_confidence=0.5) as face_mesh:

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Webcam frame could not be read.")
                break

            frame = cv2.resize(frame, (width, height))
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            seg_result = segmenter.process(rgb)
            hand_result = hands.process(rgb)
            face_result = face_mesh.process(rgb)
            rgb.flags.writeable = True

            mask = make_mask(seg_result.segmentation_mask, frame.shape)
            person_ratio = float(np.mean(mask > 0.35))

            # Capture a clean room background automatically when the user steps away.
            if background is None:
                if person_ratio < 0.035:
                    empty_frames.append(frame.copy())
                    empty_count += 1
                    if len(empty_frames) > 18:
                        empty_frames.pop(0)
                    if empty_count >= 18 and len(empty_frames) >= 12:
                        background = np.median(np.stack(empty_frames), axis=0).astype(np.uint8)
                        print("Background captured. Come back into frame.")
                else:
                    empty_count = 0
                    empty_frames.clear()

            if background is None:
                background = np.zeros_like(frame)

            gesture = detect_forehead_gesture(hand_result, face_result)
            now = time.time()

            if gesture and now > gesture_cooldown and transition_start is None:
                if gesture_start is None:
                    gesture_start = now
                elif now - gesture_start >= GESTURE_HOLD_SECONDS:
                    transition_from_visible = visible
                    transition_start = now
                    gesture_start = None
                    gesture_cooldown = now + 1.4
                    print("Instant Transmission!")
            else:
                gesture_start = None

            progress = 0.0
            returning = not transition_from_visible
            if transition_start is not None:
                progress = min(1.0, (now - transition_start) / TRANSITION_SECONDS)
                if progress >= 1.0:
                    visible = not transition_from_visible
                    transition_start = None
                    progress = 0.0

            if transition_start is None:
                opacity = 1.0 if visible else 0.0
                aura_progress = 0.0
            else:
                opacity = (1.0 - progress) if transition_from_visible else progress
                aura_progress = float(np.sin(progress * np.pi))

            alpha = (mask * opacity)[:, :, None]
            output = (frame.astype(np.float32) * alpha +
                      background.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

            if transition_start is not None:
                output = draw_aura(output, mask, aura_progress, returning)

            # Gesture countdown is shown in the app preview only, never sent to the virtual camera.
            preview = output.copy()
            if background is None or empty_count < 18:
                cv2.putText(preview, "STEP OUT OF FRAME: CAPTURING BACKGROUND",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (80, 220, 255), 2, cv2.LINE_AA)
            elif gesture_start is not None:
                remaining = max(0.0, GESTURE_HOLD_SECONDS - (now - gesture_start))
                cv2.putText(preview, f"HOLD... {remaining:.1f}s",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 240, 180), 2, cv2.LINE_AA)
            else:
                state_text = "VISIBLE" if visible else "INSTANT TRANSMISSION"
                cv2.putText(preview, state_text, (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (160, 240, 255), 2, cv2.LINE_AA)

            if virtual_cam is not None:
                try:
                    virtual_cam.send(output)
                    virtual_cam.sleep_until_next_frame()
                except Exception as exc:
                    print(f"Virtual camera stopped: {exc}")
                    virtual_cam = None

            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                background = None
                empty_frames.clear()
                empty_count = 0
                print("Background reset. Step out of frame.")
            elif key == ord("t") and transition_start is None:
                transition_from_visible = visible
                transition_start = time.time()
                gesture_cooldown = time.time() + 1.4

    camera.release()
    if virtual_cam is not None:
        virtual_cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
