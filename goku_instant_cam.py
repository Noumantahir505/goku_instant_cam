"""
Goku Instant Cam
Lightweight Python-only webcam effect inspired by Goku's Instant Transmission.

Gesture: bring index + middle fingertips together to the forehead.
Visual: a very brief blue-white flash, a thin afterimage/speed smear, then
the subject is gone. No energy rings, orbiting sparks, or surrounding aura.
"""
import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

WINDOW = "Goku Instant Cam"
WIDTH, HEIGHT, FPS = 640, 480, 20
SMALL_W, SMALL_H = 320, 240
GESTURE_HOLD = 0.55
TRANSITION_TIME = 0.32


def arguments():
    p = argparse.ArgumentParser(description="Goku Instant Transmission webcam")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--no-virtual-camera", action="store_true")
    return p.parse_args()


def soft_mask(raw_mask, size):
    mask = cv2.resize(raw_mask, size, interpolation=cv2.INTER_LINEAR)
    mask = np.clip((mask - 0.16) / 0.68, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (3, 3), 0)


def forehead_gesture(hand_result, face_result):
    """Check the classic two-finger-to-forehead pose using light FaceDetection."""
    if not hand_result or not hand_result.multi_hand_landmarks:
        return False
    if not face_result or not face_result.detections:
        return False

    box = face_result.detections[0].location_data.relative_bounding_box
    forehead = np.array([box.xmin + box.width * 0.50,
                         box.ymin + box.height * 0.20], dtype=np.float32)
    hand = hand_result.multi_hand_landmarks[0].landmark
    index_tip = np.array([hand[8].x, hand[8].y], dtype=np.float32)
    middle_tip = np.array([hand[12].x, hand[12].y], dtype=np.float32)

    together = np.linalg.norm(index_tip - middle_tip) < 0.065
    near_forehead = (np.linalg.norm(index_tip - forehead) < 0.145 and
                     np.linalg.norm(middle_tip - forehead) < 0.16)
    return bool(together and near_forehead)


def instant_transmission(output, source_frame, mask, phase, direction):
    """
    Short Instant Transmission cue: a cold flash, bright silhouette edge and
    one directional afterimage. No rings, orbiting particles or long fade.
    """
    if phase >= 0.48:
        return output

    h, w = output.shape[:2]
    strength = max(0.0, 1.0 - phase / 0.48)
    out = output.astype(np.float32)

    # Localized cool-white flash around the subject, not a full-screen fade.
    soft = cv2.GaussianBlur(mask, (0, 0), 8)
    flash = max(0.0, 1.0 - phase / 0.12)
    glow_alpha = (soft * (0.24 * strength + 0.38 * flash))[:, :, None]
    cool_white = np.zeros_like(output, dtype=np.float32)
    cool_white[:] = (255, 248, 226)  # BGR: cool blue-white
    out = out * (1.0 - glow_alpha) + cool_white * glow_alpha

    # Fine bright edge around the silhouette, visible for only a few frames.
    binary = (mask > 0.20).astype(np.uint8)
    expanded = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1)
    edge = np.clip(expanded.astype(np.float32) - binary.astype(np.float32), 0, 1)
    edge_alpha = (edge * (0.72 * strength))[:, :, None]
    out = out * (1.0 - edge_alpha) + cool_white * edge_alpha

    # One displaced, tinted afterimage gives the characteristic instant snap.
    ys, xs = np.where(mask > 0.35)
    if len(xs) > 20:
        dx = int((-1 if direction < 0 else 1) * (8 + 22 * strength))
        matrix = np.float32([[1, 0, dx], [0, 1, 0]])
        ghost_frame = cv2.warpAffine(
            source_frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        ghost_mask = cv2.warpAffine(
            mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        ghost_tint = cv2.addWeighted(
            ghost_frame, 0.58,
            np.full_like(ghost_frame, (255, 235, 190)), 0.42, 0
        ).astype(np.float32)
        ghost_alpha = (ghost_mask * (0.38 * strength))[:, :, None]
        out = out * (1.0 - ghost_alpha) + ghost_tint * ghost_alpha

    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    args = arguments()
    width, height, fps = args.width, args.height, args.fps
    small_size = (SMALL_W, SMALL_H)

    camera = cv2.VideoCapture(
        args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open webcam {args.camera}. Try --camera 1.")

    virtual_cam = None
    if not args.no_virtual_camera:
        try:
            import pyvirtualcam
            virtual_cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            print("Virtual camera connected:", virtual_cam.device)
            print("IMPORTANT: Do NOT click 'Start Virtual Camera' inside OBS.")
            print("Python is sending the processed frames directly to OBS Virtual Camera.")
            print("Open Zoom/Meet/Teams and select OBS Virtual Camera.")
        except Exception as exc:
            print("Virtual camera is not ready:", exc)
            print("In OBS, make sure 'Start Virtual Camera' is STOPPED.")
            print("Close any app using the camera, then restart Goku Instant Cam.")
            print("Preview mode will continue.")

    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.55, min_tracking_confidence=0.50
    )
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.55
    )

    background = None
    empty_frames = []
    empty_count = 0
    visible = True
    transition_started = None
    transition_from_visible = True
    gesture_started = None
    cooldown_until = 0.0
    frame_number = 0
    cached_mask = np.zeros((height, width), dtype=np.float32)
    cached_hands = None
    cached_face = None
    last_gesture = False

    print("\\nGOKU INSTANT CAM")
    print("Step out of frame briefly to capture your room background.")
    print("Return, then touch your index + middle fingertips to your forehead.")
    print("Repeat the gesture to reappear.")
    print("OBS Virtual Camera: leave OBS's own Start Virtual Camera button STOPPED.")
    print("B = recapture background | T = test | Q/ESC = quit\\n")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False

            # Run heavy inference at 320x240 and reuse landmark results between frames.
            seg = segmenter.process(rgb_small)
            cached_mask = soft_mask(seg.segmentation_mask, (width, height))
            if frame_number % 2 == 0:
                cached_hands = hands.process(rgb_small)
            if frame_number % 3 == 0:
                cached_face = face_detector.process(rgb_small)
            frame_number += 1
            rgb_small.flags.writeable = True

            person_ratio = float(np.mean(cached_mask > 0.40))
            if background is None:
                if person_ratio < 0.025:
                    empty_frames.append(frame.copy())
                    if len(empty_frames) > 8:
                        empty_frames.pop(0)
                    empty_count += 1
                    if empty_count >= 18 and len(empty_frames) >= 6:
                        background = np.median(
                            np.stack(empty_frames), axis=0
                        ).astype(np.uint8)
                        print("Room background captured. Come back into frame.")
                else:
                    empty_count = 0
                    empty_frames.clear()

            now = time.time()
            gesture = forehead_gesture(cached_hands, cached_face)
            if gesture and now >= cooldown_until and transition_started is None:
                if gesture_started is None:
                    gesture_started = now
                elif now - gesture_started >= GESTURE_HOLD:
                    transition_from_visible = visible
                    transition_started = now
                    gesture_started = None
                    cooldown_until = now + 1.0
                    print("Instant Transmission!")
            else:
                gesture_started = None

            phase = 0.0
            if transition_started is not None:
                phase = min(1.0, (now - transition_started) / TRANSITION_TIME)
                if phase >= 1.0:
                    visible = not transition_from_visible
                    transition_started = None
                    phase = 0.0

            if transition_started is None:
                opacity = 1.0 if visible else 0.0
                effect_phase = 0.0
                direction = -1 if visible else 1
            else:
                # Instant Transmission is a snap, not a dissolve:
                # a short flash/afterimage, then a hard disappearance/appearance.
                effect_phase = phase
                direction = -1 if transition_from_visible else 1
                if transition_from_visible:
                    opacity = 1.0 if phase < 0.10 else 0.0
                else:
                    opacity = 0.0 if phase < 0.10 else 1.0

            base = background if background is not None else np.zeros_like(frame)
            alpha = (cached_mask * opacity)[:, :, None]
            output = (frame.astype(np.float32) * alpha +
                      base.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

            if transition_started is not None:
                output = instant_transmission(output, frame, cached_mask, effect_phase, direction)

            preview = output.copy()
            if background is None:
                message = "STEP OUT OF FRAME TO CAPTURE BACKGROUND"
                color = (0, 210, 255)
            elif gesture_started is not None:
                left = max(0.0, GESTURE_HOLD - (now - gesture_started))
                message = f"HOLD FINGERS TO FOREHEAD: {left:.1f}s"
                color = (220, 235, 255)
            else:
                message = "VISIBLE - INSTANT TRANSMISSION READY" if visible else "INSTANT TRANSMISSION"
                color = (180, 235, 255)
            cv2.putText(preview, message, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)

            if virtual_cam is not None:
                try:
                    virtual_cam.send(output)
                    virtual_cam.sleep_until_next_frame()
                except Exception as exc:
                    print("Virtual camera output stopped:", exc)
                    virtual_cam.close()
                    virtual_cam = None

            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                background = None
                empty_frames.clear()
                empty_count = 0
                print("Background reset. Step out of frame.")
            elif key == ord("t") and transition_started is None:
                transition_from_visible = visible
                transition_started = time.time()
                cooldown_until = time.time() + 1.0

    finally:
        camera.release()
        if virtual_cam is not None:
            virtual_cam.close()
        segmenter.close()
        hands.close()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
