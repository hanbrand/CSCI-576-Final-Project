import cv2
import numpy as np
import os
import sys

# Constants for video frame dimensions
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Function to compute motion vectors using optical flow
def compute_motion_vectors(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray, next=curr_gray, flow=None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    return flow

# Function to segment foreground and background using motion magnitude
def segment_foreground_background(flow, threshold_scale=2.0):
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Adaptive threshold: Use a scale of the mean motion magnitude
    mean_magnitude = np.mean(magnitude)
    threshold = mean_magnitude * threshold_scale
    foreground_mask = magnitude > threshold

    return foreground_mask

# Function to apply segmentation masks
def apply_segmentation_mask(frame, foreground_mask):
    segmented_frame = np.zeros_like(frame)
    segmented_frame[foreground_mask] = frame[foreground_mask]
    return segmented_frame

# Function to process raw RGB file
def process_video(input_file, output_file):
    # Read the raw RGB video file
    with open(input_file, "rb") as f:
        video_data = np.frombuffer(f.read(), dtype=np.uint8)
        frame_count = len(video_data) // (FRAME_WIDTH * FRAME_HEIGHT * 3)
        frames = video_data.reshape((frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3))[..., ::-1]  # Convert to BGR for OpenCV

    output_frames = []

    # Initialize variables for optical flow
    prev_frame = frames[0]
    output_frames.append(prev_frame)  # Add the first frame as it is

    for i in range(1, len(frames)):
        curr_frame = frames[i]

        # Compute motion vectors
        flow = compute_motion_vectors(prev_frame, curr_frame)

        # Segment the frame into foreground and background
        foreground_mask = segment_foreground_background(flow)

        # Apply segmentation mask to create a segmented frame
        segmented_frame = apply_segmentation_mask(curr_frame, foreground_mask)

        output_frames.append(segmented_frame)

        # Display the original, motion vectors, and segmented frames side-by-side
        combined_frame = np.hstack((curr_frame, segmented_frame))
        cv2.imshow("Original vs Motion Vectors vs Segmented", combined_frame)
        
        # Add a delay for frame display (30ms per frame approximates ~30fps)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

        prev_frame = curr_frame

    # Close the OpenCV window after processing
    cv2.destroyAllWindows()

    # Write the segmented video to the output file
    with open(output_file, "wb") as f:
        for frame in output_frames:
            f.write(frame[..., ::-1].tobytes())  # Convert back to RGB before saving

    return output_frames

# Main function
if __name__ == "__main__":
    input_file = sys.argv[1]  # Replace with your input file
    output_file = "segmented_video.rgb"  # Replace with your output file

    # Process the video
    process_video(input_file, output_file)
