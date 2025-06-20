import argparse
import cv2
import numpy as np

from rtmlib import Body, draw_skeleton, PoseTracker,  BodyWithFeet, Wholebody


def main():
    parser = argparse.ArgumentParser(description='Choreography video overlay with webcam pose estimation')
    parser.add_argument('--video-path', required=True, help='Path to choreography video')
    parser.add_argument('--backend', default='onnxruntime', help='Inference backend')
    parser.add_argument('--device', default='cuda', help='Runtime device')
    parser.add_argument('--openpose-skeleton', action='store_true', help='Use openpose style skeleton')
    args = parser.parse_args()

    video_cap = cv2.VideoCapture(args.video_path)
    webcam_cap = cv2.VideoCapture(0)


    # rtm pose m https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
    # rtm det nano 
    #body = Body(mode="lightweight", to_openpose=args.openpose_skeleton, backend=args.backend, device=args.device)
    #body_with_feet = BodyWithFeet(to_openpose=args.openpose_skeleton, backend=args.backend, device=args.device)
    body = Wholebody(mode="lightweight", to_openpose=args.openpose_skeleton, backend=args.backend, device=args.device)

    while video_cap.isOpened() and webcam_cap.isOpened():
        ret_video, video_frame = video_cap.read()
        ret_cam, cam_frame = webcam_cap.read()

        if not ret_video or not ret_cam:
            break

        keypoints, scores = body(cam_frame)

        frame_show = video_frame.copy()
        frame_show = draw_skeleton(frame_show,
                                   keypoints,
                                   scores,
                                   openpose_skeleton=args.openpose_skeleton,
                                   kpt_thr=0.3,
                                   line_width=5)

        cv2.imshow('Choreography Pose', frame_show)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_cap.release()
    webcam_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()