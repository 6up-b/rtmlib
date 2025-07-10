import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

from rtmlib import Wholebody
from rtmlib.visualization.draw import draw_skeleton
from rtmlib.visualization.skeleton import (
    coco17,
    coco133,
    hand21,
    halpe26,
    openpose18,
    openpose134,
)

ANGLE_TRIPLETS = [
    (5, 7, 9),    # left elbow
    (6, 8, 10),   # right elbow
    (11, 13, 15), # left knee
    (12, 14, 16), # right knee
    (19, 15, 17), # left ankle dorsiflexion
    (22, 16, 20), # right ankle dorsiflexion
    # (13, 11, 5), # left hip
    # (14, 12, 6), # right hip
    # (7, 9, 5), # left shoulder
    # (8, 10, 6), # right hip
]


def _select_skeleton(num_keypoints):
    if num_keypoints == 17:
        return coco17
    if num_keypoints == 18:
        return openpose18
    if num_keypoints == 21:
        return hand21
    if num_keypoints == 26:
        return halpe26
    if num_keypoints == 133:
        return coco133
    if num_keypoints == 134:
        return openpose134
    return None


def _get_edges(num_keypoints):
    skeleton = _select_skeleton(num_keypoints)
    edges = []
    if skeleton is not None:
        name_to_idx = {
            info['name']: info['id']
            for info in skeleton['keypoint_info'].values()
        }
        for ske_info in skeleton['skeleton_info'].values():
            link = ske_info['link']
            if link[0] in name_to_idx and link[1] in name_to_idx:
                edges.append((name_to_idx[link[0]], name_to_idx[link[1]]))
    return edges


def compute_joint_angles(keypoints, scores, kpt_thr=0.3):
    keypoints = np.asarray(keypoints).squeeze()
    scores = np.asarray(scores).squeeze()
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
        scores = scores[0]
    if scores.ndim == 0:
        scores = np.full(len(keypoints), float(scores))

    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        if (
            a >= len(keypoints)
            or b >= len(keypoints)
            or c >= len(keypoints)
            or scores[a] < kpt_thr
            or scores[b] < kpt_thr
            or scores[c] < kpt_thr
        ):
            angles.append(float('nan'))
            continue
        v1 = keypoints[a] - keypoints[b]
        v2 = keypoints[c] - keypoints[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(float('nan'))
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(math.acos(cosang)))
    return np.array(angles)


def update_ema(current, ema, alpha):
    if np.isnan(current):
        return ema
    if np.isnan(ema):
        return current
    return alpha * current + (1 - alpha) * ema

def error_to_color(err, max_deg=100):
    if np.isnan(err):
        return (255, 255, 255)
    MAX_ERR = math.radians(max_deg)
    t = np.clip(err / MAX_ERR, 0.0, 1.0)
    r = int(255 * t)
    g = int(255 * (1 - t))
    return (0, g, r)

def make_error_overlay(
    keypoints,
    scores,
    colors,
    img_shape,
    kpt_thr=0.3,
    radius=80,
    blur=81,
):
    """Create a blurred overlay based on joint angle errors.

    ``colors`` is a mapping of keypoint indices to BGR tuples. Only those
    keypoints are drawn.
    """
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores).squeeze()

    if keypoints.ndim == 3:
        keypoints = keypoints[0]
        scores = scores[0]
    if scores.ndim == 0:
        scores = np.full(len(keypoints), float(scores))

    overlay = np.zeros(img_shape, dtype=np.uint8)

    if colors:
        for idx, color in colors.items():
            if (
                idx < len(keypoints)
                and idx < scores.size
                and scores[idx] >= kpt_thr
            ):
                cv2.circle(
                    overlay,
                    (int(keypoints[idx][0]), int(keypoints[idx][1])),
                    radius,
                    color,
                    -1,
                )

    k = blur if blur % 2 == 1 else blur + 1
    overlay = cv2.GaussianBlur(overlay, (k, k), 0)
    return overlay


def draw_pose_silhouette(
    img,
    keypoints,
    scores,
    kpt_thr=0.3,
    radius=15,
    fill_color=(64, 64, 64),
    thickness=2,
    error_overlay=None,
):
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if keypoints.ndim == 3:
        for kpts, scs in zip(keypoints, scores):
            img = draw_pose_silhouette(
                img,
                kpts,
                scs,
                kpt_thr=kpt_thr,
                radius=radius,
                fill_color=fill_color,
                thickness=thickness,
                error_overlay=error_overlay,
            )
        return img

    scores = np.squeeze(scores)
    edges = _get_edges(len(keypoints))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for kp, sc in zip(keypoints, scores):
        if float(sc) >= kpt_thr:
            cv2.circle(mask, (int(kp[0]), int(kp[1])), radius, 255, -1)
    for a, b in edges:
        if (
            a >= len(scores)
            or b >= len(scores)
            or scores[a] < kpt_thr
            or scores[b] < kpt_thr
        ):
            continue
        pt0 = keypoints[a]
        pt1 = keypoints[b]
        cv2.line(mask, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])), 255, radius * 2)

    if not np.any(mask):
        return img

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    hull = max(contours, key=cv2.contourArea)

    fill_mask = np.zeros_like(mask)
    cv2.drawContours(fill_mask, [hull], -1, 255, -1)

    border_mask = np.zeros_like(mask)
    cv2.drawContours(border_mask, [hull], -1, 255, thickness)

    # Draw an outer black outline so the hull stands out against the video
    cv2.drawContours(img, [hull], -1, (64, 64, 64), thickness + 3)

    alpha = 0.4
    idx = fill_mask > 0
    if np.any(idx):
        img[idx] = (img[idx] * (1 - alpha) + np.array(fill_color) * alpha).astype(np.uint8)

    if error_overlay is not None:
        color_mask = cv2.bitwise_and(error_overlay, error_overlay, mask=border_mask)
        color_mask = np.clip(color_mask * 1.2, 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, color_mask, 1.0, 0)
    else:
        cv2.drawContours(img, [hull], -1, (0, 160, 0), thickness)

    return img


def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side choreography and webcam silhouette demo'
    )
    parser.add_argument('--video', required=True, help='Path to choreography video')
    parser.add_argument('--debug-video', help='Optional prerecorded webcam footage')
    parser.add_argument('--log-file', default='angle_log.csv', help='CSV log path')
    parser.add_argument('--mode', default='lightweight',
                        choices=['performance', 'balanced', 'lightweight'])
    parser.add_argument('--backend', default='onnxruntime',
                        choices=['opencv', 'onnxruntime', 'openvino'])
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--radius', type=int, default=24,
                        help='Circle radius for silhouette')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='Keypoint score threshold')
    args = parser.parse_args()

    cap_video = cv2.VideoCapture(args.video)
    if args.debug_video:
        cap_webcam = cv2.VideoCapture(args.debug_video)
    else:
        cap_webcam = cv2.VideoCapture(0)

    body = Wholebody(mode=args.mode, backend=args.backend, device=args.device)

    fps = cap_video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    eval_frames = max(1, int(round(0.7 * fps)))
    overlay_duration = 0.5
    alpha = 1 - pow(0.05, 1 / 10)

    ema = np.full(len(ANGLE_TRIPLETS), np.nan)
    log_rows = []
    overlay_timer = 0.0
    colors = None
    frame_idx = 0

    while cap_video.isOpened() and cap_webcam.isOpened():
        ret_v, frame_v = cap_video.read()
        ret_w, frame_w = cap_webcam.read()
        if not ret_v or not ret_w:
            break

        kpts_v, scores_v = body(frame_v)
        kpts_w, scores_w = body(frame_w)

        angles_v = compute_joint_angles(kpts_v, scores_v, args.kpt_thr)
        angles_w = compute_joint_angles(kpts_w, scores_w, args.kpt_thr)
        diffs = np.abs(angles_w - angles_v)

        for i, d in enumerate(diffs):
            ema[i] = update_ema(d, ema[i], alpha)

        frame_idx += 1
        if frame_idx % eval_frames == 0:
            overlay_timer = overlay_duration
            colors = {}
            for idx, (_, b, _) in enumerate(ANGLE_TRIPLETS):
                colors[b] = error_to_color(ema[idx])
            log_rows.append([time.time()] + ema.tolist())
        elif overlay_timer > 0:
            overlay_timer -= 1 / fps
            if overlay_timer <= 0:
                colors = None

        err_v = make_error_overlay(
            kpts_v,
            scores_v,
            colors,
            frame_v.shape,
            kpt_thr=args.kpt_thr,
            radius=80,
            blur=91,
        )
        err_w = make_error_overlay(
            kpts_w,
            scores_w,
            colors,
            frame_w.shape,
            kpt_thr=args.kpt_thr,
            radius=80,
            blur=91,
        )

        disp_v = draw_pose_silhouette(
            frame_v.copy(),
            kpts_v,
            scores_v,
            kpt_thr=args.kpt_thr,
            radius=args.radius,
            fill_color=(0, 0, 0),
            thickness=4,
            error_overlay=err_v,
        )
        disp_w = draw_pose_silhouette(
            frame_w.copy(),
            kpts_w,
            scores_w,
            kpt_thr=args.kpt_thr,
            radius=args.radius,
            fill_color=(0, 0, 0),
            thickness=4,
            error_overlay=err_w,
        )

        skel_v = frame_v.copy()
        skel_v = draw_skeleton(
            skel_v,
            kpts_v,
            scores_v,
            openpose_skeleton=False,
            kpt_thr=args.kpt_thr,
            line_width=2,
        )
        disp_v = cv2.addWeighted(disp_v, 1.0, skel_v, 0.3, 0)

        skel_w = frame_w.copy()
        skel_w = draw_skeleton(
            skel_w,
            kpts_w,
            scores_w,
            openpose_skeleton=False,
            kpt_thr=args.kpt_thr,
            line_width=2,
        )
        disp_w = cv2.addWeighted(disp_w, 1.0, skel_w, 0.3, 0)

        disp_v = cv2.resize(disp_v, (640, 480))
        disp_w = cv2.resize(disp_w, (640, 480))
        combined = np.hstack([disp_v, disp_w])

        cv2.imshow('Choreo (left) vs Webcam (right)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_video.release()
    cap_webcam.release()
    cv2.destroyAllWindows()

    if log_rows:
        log_path = Path(args.log_file)
        with log_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + [f'angle_{i}' for i in range(len(ANGLE_TRIPLETS))])
            writer.writerows(log_rows)


if __name__ == '__main__':
    main()