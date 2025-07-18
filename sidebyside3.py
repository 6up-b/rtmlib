import argparse
import cv2
import numpy as np

from rtmlib import Wholebody
from rtmlib.visualization.skeleton import (coco17, coco133, hand21, halpe26,
                                            openpose18, openpose134)


def _select_skeleton(num_keypoints):
    """Return skeleton metadata for a given number of keypoints."""
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


def _get_root(keypoints, scores, kpt_thr=0.3):
    """Compute a reference root point for a set of keypoints."""
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores).squeeze()
    if scores.ndim == 0:
        scores = np.full(len(keypoints), float(scores))

    hip_l, hip_r, nose = 11, 12, 0

    if (scores.size > max(hip_l, hip_r)
            and scores[hip_l] >= kpt_thr and scores[hip_r] >= kpt_thr):
        return (keypoints[hip_l] + keypoints[hip_r]) / 2

    for idx in (hip_l, hip_r, nose):
        if scores.size > idx and scores[idx] >= kpt_thr:
            return keypoints[idx]

    valid = (scores >= kpt_thr) if scores.ndim > 0 else np.ones(len(keypoints),
                                                                dtype=bool)
    if np.any(valid):
        return keypoints[valid].mean(axis=0)

    return keypoints.mean(axis=0)


def draw_joint_scores(img,
                      kpts_a,
                      scores_a,
                      kpts_b,
                      scores_b,
                      kpt_thr=0.3,
                      radius=6):
    """Draw keypoints with colors determined by pose similarity."""
    kpts_a = np.asarray(kpts_a)
    scores_a = np.asarray(scores_a).squeeze()
    kpts_b = np.asarray(kpts_b)
    scores_b = np.asarray(scores_b).squeeze()

    if kpts_a.ndim == 3:
        kpts_a = kpts_a[0]
        scores_a = scores_a[0]
    if kpts_b.ndim == 3:
        kpts_b = kpts_b[0]
        scores_b = scores_b[0]

    if scores_a.ndim == 0:
        scores_a = np.full(len(kpts_a), float(scores_a))
    if scores_b.ndim == 0:
        scores_b = np.full(len(kpts_b), float(scores_b))

    if kpts_a.size == 0 or kpts_b.size == 0:
        return img

    root_a = _get_root(kpts_a, scores_a, kpt_thr)
    root_b = _get_root(kpts_b, scores_b, kpt_thr)

    num = min(len(kpts_a), len(kpts_b))
    for i in range(num):
        if (i >= scores_a.size or i >= scores_b.size or scores_a[i] < kpt_thr
                or scores_b[i] < kpt_thr):
            continue

        vec_a = kpts_a[i] - root_a
        vec_b = kpts_b[i] - root_b
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            continue

        # cosine angle similarity
        cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        angle_score = (cos_sim + 1.0) / 2.0

        # distance similarity
        dist_ratio = abs(norm_a - norm_b) / max(norm_a, norm_b)
        dist_score = 1.0 - np.clip(dist_ratio, 0.0, 1.0)

        closeness = (angle_score + dist_score) / 2.0
        color = (0, int(255 * closeness), int(255 * (1 - closeness)))

        cv2.circle(img, (int(kpts_a[i][0]), int(kpts_a[i][1])), radius, color,
                   -1)

    return img


def make_error_overlay(kpts_a,
                       scores_a,
                       kpts_b,
                       scores_b,
                       img_shape,
                       kpt_thr=0.3,
                       radius=6,
                       blur=21):
    """Return a blurred color overlay representing joint errors.

    The ``radius`` parameter controls how large each keypoint circle is when
    drawing the error overlay. Using a larger radius increases the chroma of the
    final silhouette contour.
    """
    overlay = np.zeros(img_shape, dtype=np.uint8)
    overlay = draw_joint_scores(overlay, kpts_a, scores_a, kpts_b, scores_b,
                                kpt_thr=kpt_thr, radius=radius)
    k = blur if blur % 2 == 1 else blur + 1
    overlay = cv2.GaussianBlur(overlay, (k, k), 0)
    return overlay


def draw_pose_silhouette(img,
                         keypoints,
                         scores,
                         kpt_thr=0.3,
                         radius=15,
                         fill_color=(64, 64, 64),
                         thickness=2,
                         error_overlay=None):
    """Draw a convex hull silhouette from pose keypoints.

    Args:
        img (np.ndarray): Image to draw on.
        keypoints (np.ndarray): Keypoint coordinates with shape ``(K, 2)`` or
            ``(N, K, 2)``.
        scores (np.ndarray): Keypoint scores with shape ``(K,)`` or ``(N, K)``.
        fill_color (tuple): RGB fill color for the silhouette. The filled region
            is blended with the image at 20%% opacity.
    """
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if keypoints.ndim == 3:
        for kpts, scs in zip(keypoints, scores):
            img = draw_pose_silhouette(img,
                                      kpts,
                                      scs,
                                      kpt_thr=kpt_thr,
                                      radius=radius,
                                      fill_color=fill_color,
                                      thickness=thickness,
                                      error_overlay=error_overlay)
        return img

    scores = np.squeeze(scores)

    skeleton = _select_skeleton(len(keypoints))
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

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
        if float(sc) >= kpt_thr:
            cv2.circle(mask, (int(kp[0]), int(kp[1])), radius, 255, -1)

    for a, b in edges:
        if (a >= len(scores) or b >= len(scores) or scores[a] < kpt_thr
                or scores[b] < kpt_thr):
            continue
        pt0 = keypoints[a]
        pt1 = keypoints[b]
        cv2.line(mask, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1])),
                 255, radius * 2)

    if not np.any(mask):
        return img

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    hull = max(contours, key=cv2.contourArea)

    fill_mask = np.zeros_like(mask)
    cv2.drawContours(fill_mask, [hull], -1, 255, -1)

    border_mask = np.zeros_like(mask)
    cv2.drawContours(border_mask, [hull], -1, 255, thickness)

     # draw the semi-transparent fill first
    alpha = 0.4
    idx = fill_mask > 0
    if np.any(idx):
        img[idx] = (img[idx] * (1 - alpha) + np.array(fill_color) * alpha).astype(np.uint8)

    # overlay the gradient error colors along the silhouette edge. if no
    # error overlay is supplied, fall back to a solid green outline.
    if error_overlay is not None:
        color_mask = cv2.bitwise_and(error_overlay, error_overlay, mask=border_mask)
        color_mask = np.clip(color_mask * 1.2, 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, color_mask, 1.0, 0)
    else:
        cv2.drawContours(img, [hull], -1, (0, 160, 0), thickness+6)

    # ensure the contour outline sits on top of the fill

    return img

'''
point of scoring is to show that you did somnething wrong

within 8 second interval --> evaluation saying "perfect"

evaluation metric flashes every 4 beats to see if current pose matches choreography

immediate error evaluation (joint red) but at endd of 8 seocnd interval show overall interval score as green

discrete bpm basis for sum of evaluation metrics

neccessary to show live:
- know what you did wrong (joint) (doesnt need to show where it needs to go)
- knowing when you did something wrong live

nonneccessary:
- correction/what shouldve been done
- afterwards being told whqat to fix
- live scoring isnt entirely neccessary



'''

def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side choreography and webcam silhouette demo')
    parser.add_argument('--video', required=True, help='Path to choreography video')
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
    cap_webcam = cv2.VideoCapture(0)

    body = Wholebody(mode=args.mode, backend=args.backend, device=args.device)

    while cap_video.isOpened() and cap_webcam.isOpened():
        ret_v, frame_v = cap_video.read()
        ret_w, frame_w = cap_webcam.read()
        if not ret_v or not ret_w:
            break

        kpts_v, scores_v = body(frame_v)
        kpts_w, scores_w = body(frame_w)

        err_v = make_error_overlay(kpts_v,
                                   scores_v,
                                   kpts_w,
                                   scores_w,
                                   frame_v.shape,
                                   kpt_thr=args.kpt_thr,
                                   radius=max(3, int(args.radius * 1.5)),
                                   blur=args.radius * 4 + 1)

        err_w = make_error_overlay(kpts_w,
                                   scores_w,
                                   kpts_v,
                                   scores_v,
                                   frame_w.shape,
                                   kpt_thr=args.kpt_thr,
                                   radius=max(3, int(args.radius * 1.5)),
                                   blur=args.radius * 4 + 1)

        disp_v = draw_pose_silhouette(frame_v.copy(),
                                      kpts_v,
                                      scores_v,
                                      kpt_thr=args.kpt_thr,
                                      radius=args.radius,
                                      fill_color=(0, 0, 0),
                                      thickness=4,
                                      error_overlay=err_v)

        disp_w = draw_pose_silhouette(frame_w.copy(),
                                      kpts_w,
                                      scores_w,
                                      kpt_thr=args.kpt_thr,
                                      radius=args.radius,
                                      fill_color=(0, 0, 0),
                                      thickness=4,
                                      error_overlay=err_w)

        disp_v = draw_joint_scores(disp_v, kpts_v, scores_v, kpts_w, scores_w,
                                   kpt_thr=args.kpt_thr,
                                   radius=max(3, args.radius // 3))
        disp_w = draw_joint_scores(disp_w, kpts_w, scores_w, kpts_v, scores_v,
                                   kpt_thr=args.kpt_thr,
                                   radius=max(3, args.radius // 3))

        disp_v = cv2.resize(disp_v, (640, 480))
        disp_w = cv2.resize(disp_w, (640, 480))
        combined = np.hstack([disp_v, disp_w])

        cv2.imshow('Choreo (left) vs Webcam (right)', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_video.release()
    cap_webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()