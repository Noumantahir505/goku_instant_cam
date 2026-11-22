"""
Goku Instant Cam
Lightweight Python-only webcam effect inspired by Goku's Instant Transmission.

Gesture: bring index + middle fingertips together to the forehead.
Visual: vertical energy streaks, localized space-warp distortion, rapid silhouette flicker,
and a short vertical motion smear.
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
TRANSITION_TIME = 0.56


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



def transformation_hand_state(hand_result, face_result):
    """Return close/open two-hand states with forgiving head-and-shoulder bounds."""
    if not hand_result or not hand_result.multi_hand_landmarks or len(hand_result.multi_hand_landmarks) < 2:
        return False, False, None

    wrists = []
    for hand in hand_result.multi_hand_landmarks[:2]:
        # Use palm centre rather than wrist alone; this is steadier when hands overlap.
        palm = hand.landmark[9]
        wrists.append(np.array([palm.x, palm.y], dtype=np.float32))

    distance = float(np.linalg.norm(wrists[0] - wrists[1]))
    midpoint = (wrists[0] + wrists[1]) * 0.5

    if face_result and face_result.detections:
        box = face_result.detections[0].location_data.relative_bounding_box
        face_center_x = box.xmin + box.width * 0.5
        # Include forehead through upper chest: wrists naturally drop when arms open.
        y_min = max(0.0, box.ymin - 0.25)
        y_max = min(1.0, box.ymin + box.height * 2.8)
        centered = abs(float(midpoint[0]) - face_center_x) < 0.42
        in_zone = y_min <= float(midpoint[1]) <= y_max
    else:
        # Allow the gesture even if face detection briefly drops a frame.
        centered = 0.12 <= float(midpoint[0]) <= 0.88
        in_zone = 0.05 <= float(midpoint[1]) <= 0.78

    together = distance < 0.29 and centered and in_zone
    spread = distance > 0.48 and centered and in_zone
    return together, spread, distance


def apply_super_saiyan_aura(frame, mask, now):
    """High-energy golden Super Saiyan aura: layered glow, flame tongues and lightning."""
    h, w = frame.shape[:2]
    binary = (mask > 0.24).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) < 30:
        return frame

    x0, x1 = max(0, int(xs.min()) - 42), min(w - 1, int(xs.max()) + 42)
    y0, y1 = max(0, int(ys.min()) - 48), min(h - 1, int(ys.max()) + 12)
    roi_mask = binary[y0:y1 + 1, x0:x1 + 1]
    rh, rw = roi_mask.shape
    pulse = 0.78 + 0.16 * np.sin(now * 8.5) + 0.06 * np.sin(now * 17.0)

    # Build several expanding rings around the segmented silhouette.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (57, 57))
    ring1 = cv2.subtract(cv2.dilate(roi_mask, k1), roi_mask)
    ring2 = cv2.subtract(cv2.dilate(roi_mask, k2), cv2.dilate(roi_mask, k1))
    ring3 = cv2.subtract(cv2.dilate(roi_mask, k3), cv2.dilate(roi_mask, k2))

    # Wide amber halo, bright golden middle and near-white hot edge.
    aura_small = cv2.GaussianBlur(ring1.astype(np.float32), (0, 0), 4)
    aura_mid = cv2.GaussianBlur(ring2.astype(np.float32), (0, 0), 7)
    aura_wide = cv2.GaussianBlur(ring3.astype(np.float32), (0, 0), 11)
    out = frame.copy()
    patch = out[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    halo = np.zeros_like(patch)
    halo[:] = (0, 150, 255)       # amber
    mid = np.zeros_like(patch)
    mid[:] = (0, 215, 255)        # golden yellow
    hot = np.zeros_like(patch)
    hot[:] = (80, 245, 255)        # hot yellow-white
    a3 = (aura_wide * (0.42 * pulse))[:, :, None]
    a2 = (aura_mid * (0.58 * pulse))[:, :, None]
    a1 = (aura_small * (0.74 * pulse))[:, :, None]
    patch = patch * (1.0 - a3) + halo * a3
    patch = patch * (1.0 - a2) + mid * a2
    patch = patch * (1.0 - a1) + hot * a1
    out[y0:y1 + 1, x0:x1 + 1] = np.clip(patch, 0, 255).astype(np.uint8)

    # Flickering flame tongues rise from both sides and above the body.
    flame_layer = out.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        left_edge = contour[np.argsort(contour[:, 0])[:max(1, len(contour) // 5)]]
        right_edge = contour[np.argsort(contour[:, 0])[-max(1, len(contour) // 5):]]
        for side, edge_points in ((-1, left_edge), (1, right_edge)):
            if len(edge_points) == 0:
                continue
            for i in range(8):
                p = edge_points[(i * max(1, len(edge_points) // 8) + int(now * 13) % len(edge_points)) % len(edge_points)]
                px, py = int(p[0]) + x0, int(p[1]) + y0
                height = 18 + int((0.5 + 0.5 * np.sin(now * 11 + i * 2.1)) * 34)
                sway = int(np.sin(now * 9 + i * 1.8) * 10)
                base_x = px + side * (5 + (i % 3) * 3)
                tip_x = base_x + sway + side * int(height * 0.18)
                tip_y = max(0, py - height)
                mid_x = (base_x + tip_x) // 2 + side * 5
                mid_y = (py + tip_y) // 2
                cv2.polylines(flame_layer, [np.array([
                    [base_x, py], [mid_x, mid_y], [tip_x, tip_y]
                ], dtype=np.int32)], False, (0, 190, 255), 3, cv2.LINE_AA)
                cv2.line(flame_layer, (mid_x, mid_y), (tip_x, tip_y),
                         (90, 255, 255), 1, cv2.LINE_AA)

    # Tall flame spikes above the head, animated independently.
    head_x = int((x0 + x1) * 0.5)
    head_y = max(4, y0 + 12)
    for i in range(7):
        base_x = head_x + int((i - 3) * 11)
        height = 22 + int((0.5 + 0.5 * np.sin(now * 12 + i * 1.7)) * 42)
        lean = int(np.sin(now * 8 + i) * 9)
        cv2.polylines(flame_layer, [np.array([
            [base_x - 4, head_y + 18],
            [base_x + lean, head_y - height // 2],
            [base_x + lean * 2, max(0, head_y - height)]
        ], dtype=np.int32)], False, (0, 205, 255), 3, cv2.LINE_AA)

    out = cv2.addWeighted(out, 1.0, flame_layer, 0.72 * pulse, 0)

    # Fast jagged electric bolts around the aura, not a static oval ring.
    lightning = out.copy()
    for i in range(5):
        side = -1 if i % 2 == 0 else 1
        anchor_x = x0 - 12 if side < 0 else x1 + 12
        anchor_y = y0 + int((i + 1) * (y1 - y0) / 6)
        phase = now * 10 + i * 2.4
        points = [[anchor_x, anchor_y]]
        x, y = anchor_x, anchor_y
        for j in range(4):
            x += side * int(5 + 5 * abs(np.sin(phase + j)))
            y -= 7 + int(5 * abs(np.cos(phase + j * 1.3)))
            points.append([max(0, min(w - 1, x)), max(0, min(h - 1, y))])
        cv2.polylines(lightning, [np.array(points, dtype=np.int32)], False,
                      (0, 225, 255), 2, cv2.LINE_AA)
        cv2.polylines(lightning, [np.array(points, dtype=np.int32)], False,
                      (150, 255, 255), 1, cv2.LINE_AA)

    # Small hot sparks drift upward in the aura margin.
    for i in range(18):
        sx = x0 + ((i * 23 + int(now * (34 + i % 5))) % max(1, x1 - x0 + 1))
        sy = y0 + ((i * 41 - int(now * (28 + i % 7))) % max(1, y1 - y0 + 1))
        if binary[sy, sx] == 0:
            radius = 1 + (i % 2)
            cv2.circle(lightning, (sx, sy), radius, (40, 240, 255), -1, cv2.LINE_AA)

    out = cv2.addWeighted(out, 1.0, lightning, 0.82 * pulse, 0)
    return out

def instant_transmission(output, source_frame, mask, phase, direction):
    """
    Lightweight Goku-inspired Instant Transmission stack:
    vertical energy streaks, localized wave/noise displacement,
    flicker/dissolve silhouette, vertical blur, shockwave, electric sparks, afterimage echoes and RGB glitch.
    """
    if phase >= 0.62:
        return output

    h, w = output.shape[:2]
    out = output.copy()
    strength = max(0.0, 1.0 - phase / 0.62)
    active = (mask > 0.12).astype(np.uint8)
    ys, xs = np.where(mask > 0.30)

    if len(xs) > 20:
        x0, x1 = max(0, int(xs.min()) - 18), min(w - 1, int(xs.max()) + 18)
        y0, y1 = max(0, int(ys.min()) - 18), min(h - 1, int(ys.max()) + 18)

        # 1) Wave warp / deterministic noise displacement, localized to subject.
        roi = out[y0:y1 + 1, x0:x1 + 1]
        rh, rw = roi.shape[:2]
        yy, xx = np.indices((rh, rw), dtype=np.float32)
        time_phase = phase * 34.0
        wave_x = (np.sin(yy * 0.095 + time_phase) * 3.2 +
                  np.sin(yy * 0.037 - time_phase * 0.7) * 1.8) * strength
        wave_y = np.sin(xx * 0.045 + time_phase) * 1.2 * strength
        map_x = np.clip(xx + wave_x, 0, rw - 1).astype(np.float32)
        map_y = np.clip(yy + wave_y, 0, rh - 1).astype(np.float32)
        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        local_mask = cv2.GaussianBlur(mask[y0:y1 + 1, x0:x1 + 1],
                                      (0, 0), 5)
        warp_alpha = (local_mask * (0.72 * strength))[:, :, None]
        out[y0:y1 + 1, x0:x1 + 1] = np.clip(
            roi.astype(np.float32) * (1.0 - warp_alpha) +
            warped.astype(np.float32) * warp_alpha, 0, 255
        ).astype(np.uint8)

        # 2) Vertical directional blur at the actual snap moment.
        snap = max(0.0, 1.0 - abs(phase - 0.22) / 0.16)
        if snap > 0:
            blurred = cv2.blur(out, (1, 13))
            blur_mask = cv2.GaussianBlur(mask, (0, 0), 7)
            blur_alpha = (blur_mask * (0.78 * snap))[:, :, None]
            out = cv2.addWeighted(out, 1.0, blurred, 0.0, 0)
            out = np.clip(out.astype(np.float32) * (1.0 - blur_alpha) +
                          blurred.astype(np.float32) * blur_alpha, 0, 255).astype(np.uint8)

        # 3) Vertical blue-white energy streaks around the silhouette.
        streak_layer = out.copy()
        streak_strength = min(1.0, strength * 1.35)
        count = max(5, min(22, (x1 - x0) // 15))
        for i in range(count):
            # Deterministic positions avoid per-frame random-number overhead.
            x = x0 + ((i * 37 + int(phase * 190)) % max(1, x1 - x0 + 1))
            top = max(0, y0 - ((i * 19) % 34))
            bottom = min(h - 1, y1 + ((i * 13) % 30))
            offset = int(np.sin(phase * 45 + i * 1.7) * 5)
            x = max(0, min(w - 1, x + offset))
            cv2.line(streak_layer, (x, top), (x, bottom),
                     (255, 238, 205), 1 if i % 3 else 2, cv2.LINE_AA)
        streak_alpha = 0.42 * streak_strength
        out = cv2.addWeighted(out, 1.0 - streak_alpha,
                              streak_layer, streak_alpha, 0)

        # Brief cold flash and sharp silhouette rim.
        flash = max(0.0, 1.0 - phase / 0.16)
        soft = cv2.GaussianBlur(mask, (0, 0), 8)
        flash_alpha = (soft * (0.48 * flash))[:, :, None]
        cool = np.zeros_like(out)
        cool[:] = (255, 248, 226)
        out = np.clip(out.astype(np.float32) * (1.0 - flash_alpha) +
                      cool.astype(np.float32) * flash_alpha, 0, 255).astype(np.uint8)

        binary = (mask > 0.20).astype(np.uint8)
        edge = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1) - binary
        edge_alpha = (edge.astype(np.float32) * (0.70 * strength))[:, :, None]
        out = np.clip(out.astype(np.float32) * (1.0 - edge_alpha) +
                      cool.astype(np.float32) * edge_alpha, 0, 255).astype(np.uint8)

        # 4) Expanding shockwave ring around the subject's center.
        wave_strength = max(0.0, 1.0 - abs(phase - 0.25) / 0.22)
        if wave_strength > 0:
            center = (int((x0 + x1) * 0.5), int((y0 + y1) * 0.52))
            max_radius = int(max(x1 - x0, y1 - y0) * 0.72)
            radius = max(2, int(max_radius * min(1.0, phase / 0.42)))
            wave_layer = out.copy()
            cv2.circle(wave_layer, center, radius, (255, 245, 220),
                       max(1, int(2 * wave_strength)), cv2.LINE_AA)
            cv2.circle(wave_layer, center, max(1, radius - 5),
                       max(1, int(1 * wave_strength)), (255, 190, 120), 1, cv2.LINE_AA)
            out = cv2.addWeighted(out, 1.0, wave_layer,
                                  0.48 * wave_strength, 0)

        # 5) Tiny deterministic electric sparks along the silhouette edges.
        spark_strength = max(0.0, 1.0 - abs(phase - 0.19) / 0.20)
        if spark_strength > 0:
            spark_layer = out.copy()
            for i in range(9):
                sx = x0 + ((i * 29 + int(phase * 260)) % max(1, x1 - x0 + 1))
                sy = y0 + ((i * 47 + int(phase * 170)) % max(1, y1 - y0 + 1))
                if mask[sy, sx] < 0.18:
                    length = 5 + (i % 4) * 3
                    bend = -1 if i % 2 else 1
                    points = np.array([
                        [sx, sy],
                        [sx + bend * 3, sy - length // 2],
                        [sx - bend * 2, sy - length],
                    ], dtype=np.int32)
                    cv2.polylines(spark_layer, [points], False,
                                  (255, 250, 225), 1, cv2.LINE_AA)
            out = cv2.addWeighted(out, 1.0, spark_layer,
                                  0.82 * spark_strength, 0)

        # 6) Two quick displaced afterimage echoes, strongest during vanish.
        echo_strength = max(0.0, 1.0 - abs(phase - 0.16) / 0.18)
        if echo_strength > 0:
            for echo_index, shift in enumerate((7, 14), start=1):
                dx = int((1 if direction < 0 else -1) * shift)
                matrix = np.float32([[1, 0, dx], [0, 1, 0]])
                echo_frame = cv2.warpAffine(
                    source_frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                echo_mask = cv2.warpAffine(
                    mask, matrix, (w, h), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
                echo_tint = echo_frame.copy()
                echo_tint[:, :, 0] = np.clip(
                    echo_tint[:, :, 0].astype(np.float32) * 1.12 + 18, 0, 255
                ).astype(np.uint8)
                echo_alpha = (echo_mask * (0.17 * echo_strength / echo_index))[:, :, None]
                out = np.clip(out.astype(np.float32) * (1.0 - echo_alpha) +
                              echo_tint.astype(np.float32) * echo_alpha,
                              0, 255).astype(np.uint8)

        # 7) Brief RGB channel split for a spatial-glitch accent.
        glitch_strength = max(0.0, 1.0 - abs(phase - 0.23) / 0.12)
        if glitch_strength > 0:
            roi = out[y0:y1 + 1, x0:x1 + 1]
            shift = max(1, int(4 * glitch_strength))
            blue_shift = np.roll(roi[:, :, 0], shift, axis=1)
            red_shift = np.roll(roi[:, :, 2], -shift, axis=1)
            glitch_roi = roi.copy()
            glitch_roi[:, :, 0] = blue_shift
            glitch_roi[:, :, 2] = red_shift
            glitch_mask = cv2.GaussianBlur(mask[y0:y1 + 1, x0:x1 + 1],
                                           (0, 0), 6)
            glitch_alpha = (glitch_mask * (0.42 * glitch_strength))[:, :, None]
            out[y0:y1 + 1, x0:x1 + 1] = np.clip(
                roi.astype(np.float32) * (1.0 - glitch_alpha) +
                glitch_roi.astype(np.float32) * glitch_alpha,
                0, 255
            ).astype(np.uint8)

    return out

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
        static_image_mode=False, max_num_hands=2, model_complexity=0,
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
    transform_armed_at = None
    transform_cooldown_until = 0.0
    aura_active = False

    print("\\nGOKU INSTANT CAM")
    print("Step out of frame briefly to capture your room background.")
    print("Return, then touch your index + middle fingertips to your forehead.")
    print("Repeat the gesture to reappear.")
    print("TRANSFORM: cross both hands close in front of your head, then spread them apart quickly.")
    print("Repeat the transformation gesture to toggle the golden aura off.")
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
            hands_together, hands_spread, hand_distance = transformation_hand_state(cached_hands, cached_face)
            if hands_together and now >= transform_cooldown_until:
                if transform_armed_at is None:
                    transform_armed_at = now
            if (hands_spread and transform_armed_at is not None and
                    now - transform_armed_at <= 1.35 and now >= transform_cooldown_until):
                aura_active = not aura_active
                transform_armed_at = None
                transform_cooldown_until = now + 1.8
                print("SUPER SAIYAN TRANSFORMATION!" if aura_active else "AURA DEACTIVATED")
            elif transform_armed_at is not None and now - transform_armed_at > 1.35:
                transform_armed_at = None
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
                    # Hold, then rapidly flicker/dissolve before the hard vanish.
                    if phase < 0.10:
                        opacity = 1.0
                    elif phase < 0.34:
                        flicker = (int(phase * 115) % 2)
                        opacity = 0.12 if flicker else 0.88
                    else:
                        opacity = 0.0
                else:
                    # Rebuild the silhouette with the reverse flicker sequence.
                    if phase < 0.20:
                        opacity = 0.0
                    elif phase < 0.44:
                        flicker = (int(phase * 115) % 2)
                        opacity = 0.12 if flicker else 0.88
                    else:
                        opacity = 1.0

            base = background if background is not None else np.zeros_like(frame)
            alpha = (cached_mask * opacity)[:, :, None]
            output = (frame.astype(np.float32) * alpha +
                      base.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

            if transition_started is not None:
                output = instant_transmission(output, frame, cached_mask, effect_phase, direction)

            if aura_active and visible and transition_started is None:
                output = apply_super_saiyan_aura(output, cached_mask, now)

            preview = output.copy()
            if background is None:
                message = "STEP OUT OF FRAME TO CAPTURE BACKGROUND"
                color = (0, 210, 255)
            elif transform_armed_at is not None:
                message = "HANDS DETECTED - NOW SPREAD YOUR ARMS!"
                color = (40, 225, 255)
            elif hand_distance is not None:
                message = f"TRANSFORM HANDS DETECTED: {hand_distance:.2f}"
                color = (180, 220, 255)
            elif aura_active:
                message = "SUPER SAIYAN AURA ACTIVE"
                color = (40, 225, 255)
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
