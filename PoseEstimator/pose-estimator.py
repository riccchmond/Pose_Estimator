import cv2
import mediapipe as mp
import time
import numpy as np
import os
import pandas as pd
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    static_image_mode=False,
    enable_segmentation=False,  # Explicitly disable unused features
    min_detection_confidence=0.5
)


# Context manager for video capture
@contextmanager
def video_capture_context(path):
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calculate_angle(a, b, c):
    """Calculate angle between three points with robust error handling"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        return min(360 - angle, angle) if angle > 180 else angle
    except Exception as e:
        logging.error(f"Angle calculation error: {str(e)}")
        return None


def process_video(video_path, pose_estimator):
    """Process a single video file and return structured data"""
    data = {
        'frame_number': [],
        'timestamp': [],
        'right_knee': [],
        'right_hip': [],
        'left_hip': [],
        'left_knee': [],
        'right_elbow': [],
        'right_trunk': [],
        'left_trunk': [],
        'right_shoulder': [],
        'landmarks_detected': []
    }

    try:
        with video_capture_context(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            start_time = time.time()

            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Process frame
                frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                results = pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get required landmarks
                        landmark_indices = {
                            'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
                            'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
                            'right_hip' : mp_pose.PoseLandmark.RIGHT_HIP,
                            'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            'left_ankle' : mp_pose.PoseLandmark.LEFT_ANKLE,
                            'left_knee' : mp_pose.PoseLandmark.LEFT_KNEE,
                            'left_hip' : mp_pose.PoseLandmark.LEFT_HIP,
                            'left_shoulder' : mp_pose.PoseLandmark.LEFT_SHOULDER,
                            'left_wrist' : mp_pose.PoseLandmark.LEFT_WRIST,
                            'right_wrist' : mp_pose.PoseLandmark.RIGHT_WRIST
                        }

                        # Calculate angles
                        angles = {
                            'right_knee': calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y], #landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].value.y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                # ... other angle calculations
                            ),

                            'left_knee' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            ),

                            'left_elbow' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            ),

                            'right_elbow' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            ),

                            'right_hip' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                            ),

                            'left_hip' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                            ),

                            'right_trunk' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                            ),

                            'left_trunk' : calculate_angle(
                                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            )
                            # ... other angles
                        }

                        # Store data
                        for key in data:
                            if key in angles:
                                data[key].append(angles[key])
                        data['landmarks_detected'].append(True)

                    except (AttributeError, IndexError) as e:
                        logging.warning(f"Landmark detection failed in frame {frame_count}: {str(e)}")
                        data['landmarks_detected'].append(False)
                else:
                    data['landmarks_detected'].append(False)

                # Store frame metadata
                data['frame_number'].append(frame_count)
                data['timestamp'].append(frame_time)
                frame_count += 1

                # Handle missing data
                for key in data:
                    if key not in ['frame_number', 'timestamp', 'landmarks_detected']:
                        if len(data[key]) != frame_count:
                            data[key].append(np.nan)

                # Visualization (optional)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Processing Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            logging.info(f"Processed {frame_count} frames from {os.path.basename(video_path)}")
            return data, fps

    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return None, None


def main():
    """Main processing pipeline"""
    # Configuration
    config = {
        'model_complexity': 1,
        'min_detection_confidence': 0.8,
        'min_tracking_confidence': 0.8,
        'output_dir': 'processed_data',
        'video_extensions': ['.mp4', '.avi', '.mov']
    }

    # Get input from user
    video_dir = input("Enter video directory: ")
    output_dir = input("Enter output directory: ") or config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pose estimator
    with mp_pose.Pose(
            model_complexity=config['model_complexity'],
            static_image_mode=False,
            min_detection_confidence=config['min_detection_confidence'],
            min_tracking_confidence=config['min_tracking_confidence']
    ) as pose_estimator:

        # Process videos
        for filename in os.listdir(video_dir):
            if os.path.splitext(filename)[1].lower() not in config['video_extensions']:
                continue

            video_path = os.path.join(video_dir, filename)
            logging.info(f"Processing {filename}...")

            # Process video
            data, fps = process_video(video_path, pose_estimator)

            if data and fps:
                # Create DataFrame with proper typing
                df = pd.DataFrame(data)
                df['source_video'] = filename
                df['fps'] = fps

                # Save to CSV
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_angles.csv")
                df.to_csv(output_path, index=False)
                logging.info(f"Saved data to {output_path}")

                # Generate summary report
                detection_rate = df['landmarks_detected'].mean() * 100
                logging.info(f"Landmark detection rate: {detection_rate:.2f}%")


if __name__ == "__main__":
    main()