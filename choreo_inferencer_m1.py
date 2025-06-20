import argparse
import os
from typing import List

import cv2
import numpy as np

from rtmlib import Body, draw_skeleton
from rtmlib.tools.solution.pose_tracker import pose_to_bbox


def run_inference(video_path: str, cache_path: str, model: Body) -> tuple:
    """Run pose inference on a video or load cached results."""
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return (data['keypoints'].tolist(), data['scores'].tolist(),
                data['centers'])

    cap = cv2.VideoCapture(video_path)
    keypoints_list: List[np.ndarray] = []
    scores_list: List[np.ndarray] = []
    centers_list: List[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        kpts, scores = model(frame)
        keypoints_list.append(kpts)
        scores_list.append(scores)
        bbox = pose_to_bbox(kpts)
        center = (bbox[:2] + bbox[2:]) / 2
        centers_list.append(center)
    cap.release()

    np.savez_compressed(cache_path,
                        keypoints=np.array(keypoints_list, dtype=object),
                        scores=np.array(scores_list, dtype=object),
                        centers=np.array(centers_list))
    return keypoints_list, scores_list, np.array(centers_list)


def calibrate_similarity_transform(cap_video, cap_webcam, body_model,
                                   video_kpts, video_scores):
    """
    One-time similarity transform calibration on left-click:
    User holds a neutral pose; left-click on the webcam window to capture.
    Displays webcam pose inference overlay and uses static choreography frame.
    """
    clicked = {'flag': False}
    calib_data = {'kpts_w0': None, 'scores_w0': None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked['flag'] = True

    # Read one static choreography frame for display during calibration
    ret_v0, frame_v0 = cap_video.read()
    if not ret_v0:
        raise RuntimeError("Failed to read from choreography video for calibration.")

    cv2.namedWindow('Calibration - Webcam')
    cv2.setMouseCallback('Calibration - Webcam', on_mouse)

    print("Calibration: Please hold a neutral pose and left-click the webcam window to capture.")
    # Loop until user clicks with live webcam and static video frame
    while True:
        _, frame_v = True, frame_v0  # static frame
        ret_w, frame_w = cap_webcam.read()
        if not ret_w:
            continue

        # Run pose inference on webcam frame
        kpts_w, scores_w = body_model(frame_w)
        # Draw skeleton overlay on webcam frame
        frame_w_vis = frame_w.copy()
        if kpts_w.size:
            frame_w_vis = draw_skeleton(
                frame_w_vis, kpts_w, scores_w,
                openpose_skeleton=False, kpt_thr=0.3)

        cv2.putText(frame_w_vis,
                    "Calibration: Left-click when in neutral pose",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow('Calibration - Webcam', frame_w_vis)
        cv2.imshow('Calibration - Video', frame_v)

        if clicked['flag'] and kpts_w.size:
            # Capture calibration keypoints once
            calib_data['kpts_w0'] = kpts_w
            calib_data['scores_w0'] = scores_w
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to cancel
            break

    # Use captured calibration keypoints
    kpts_w0 = calib_data['kpts_w0']
    scores_w0 = calib_data['scores_w0']
    kpts_v0 = video_kpts[0]
    scores_v0 = video_scores[0]

    # If no valid detection, fallback to identity
    if kpts_w0 is None or scores_w0 is None or scores_w0.size < 4:
        print("Calibration failed, using identity transform.")
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    # Choose anchor joints: shoulders and hips
    anchor_idxs = np.array([5, 6, 11, 12])  # COCO: L/R shoulder, L/R hip
    # Ensure we have full 17-length scores
    if scores_w0.ndim != 1 or scores_w0.shape[0] < anchor_idxs.max()+1:
        print("Unexpected calibration scores shape, using translation-only.")
        pts_w = kpts_w0[:, :2].mean(axis=0)
        pts_v = kpts_v0[:, :2].mean(axis=0)
        return np.array([[1, 0, pts_v[0]-pts_w[0]],
                         [0, 1, pts_v[1]-pts_w[1]]], dtype=np.float32)

    conf_w = scores_w0[anchor_idxs]
    conf_v = scores_v0[anchor_idxs]
    mask = (conf_w > 0.3) & (conf_v > 0.3)

    if mask.sum() < 3:
        print("Not enough valid anchors, falling back to translation-only calibration.")
        pts_w = kpts_w0[anchor_idxs, :2]
        pts_v = kpts_v0[anchor_idxs, :2]
        src_center = pts_w.mean(axis=0)
        dst_center = pts_v.mean(axis=0)
        M = np.array([[1, 0, dst_center[0] - src_center[0]],
                      [0, 1, dst_center[1] - src_center[1]]],
                     dtype=np.float32)
    else:
        pts_w = kpts_w0[anchor_idxs[mask], :2].astype(np.float32)
        pts_v = kpts_v0[anchor_idxs[mask], :2].astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts_w, pts_v)
        if M is None:
            print("Similarity estimation failed, falling back to translation-only.")
            src_center = pts_w.mean(axis=0)
            dst_center = pts_v.mean(axis=0)
            M = np.array([[1, 0, dst_center[0] - src_center[0]],
                          [0, 1, dst_center[1] - src_center[1]]],
                         dtype=np.float32)

    cv2.destroyWindow('Calibration - Webcam')
    cv2.destroyWindow('Calibration - Video')
    return M


def main():
    parser = argparse.ArgumentParser(
        description='Overlay live webcam pose onto a choreography video with one-time similarity calibration.')
    parser.add_argument('--video', required=True, help='Path to choreography video')
    parser.add_argument('--cache', help='Cache file (.npz) for pose results')
    parser.add_argument('--mode', default='lightweight',
                        choices=['performance', 'balanced', 'lightweight'],
                        help='Body model mode')
    parser.add_argument('--backend', default='onnxruntime',
                        choices=['opencv', 'onnxruntime', 'openvino'],
                        help='Inference backend')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device used for inference')
    parser.add_argument('--openpose', action='store_true',
                        help='Use OpenPose-style skeleton')

    args = parser.parse_args()

    cache_path = args.cache or os.path.splitext(args.video)[0] + '.npz'
    body_model = Body(to_openpose=args.openpose,
                      mode=args.mode,
                      backend=args.backend,
                      device=args.device)

    video_kpts, video_scores, video_centers = run_inference(
        args.video, cache_path, body_model)

    cap_video = cv2.VideoCapture(args.video)
    cap_webcam = cv2.VideoCapture(0)

    # Calibration step
    M = calibrate_similarity_transform(cap_video, cap_webcam,
                                       body_model,
                                       video_kpts, video_scores)
    # Reset video after calibration
    cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while cap_video.isOpened() and cap_webcam.isOpened():
        ret_v, frame_v = cap_video.read()
        ret_w, frame_w = cap_webcam.read()
        if not ret_v or not ret_w or frame_idx >= len(video_kpts):
            break

        kpts_w, scores_w = body_model(frame_w)
        if kpts_w.size == 0:
            frame_idx += 1
            continue

        pts = kpts_w[:, :2].astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts_hom = np.hstack([pts, ones])
        aligned_xy = (M @ pts_hom.T).T

        aligned_kpts = np.zeros_like(kpts_w)
        aligned_kpts[:, :2] = aligned_xy
        aligned_kpts[:, 2] = kpts_w[:, 2]

        img_show = frame_v.copy()
        img_show = draw_skeleton(img_show,
                                 aligned_kpts,
                                 scores_w,
                                 openpose_skeleton=args.openpose,
                                 kpt_thr=0.3)

        cv2.imshow('Choreography Alignment', img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap_video.release()
    cap_webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()