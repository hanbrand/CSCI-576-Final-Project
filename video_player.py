import cv2
import sys

def display_legend(frame):
    """Overlay control legend slightly above the bottom of the video frame."""
    legend = [
        "Controls:",
        "  'Space' - Play/Pause",
        "  'Z' - Step backward",
        "  'X' - Step forward",
        "  'ESC' - Exit"
    ]
    # Calculate the starting Y position, with a margin to avoid clipping
    y_margin = 80  # Adjust margin from the bottom
    y_start = frame.shape[0] - y_margin
    for i, line in enumerate(legend):
        y_position = y_start + (i * 20)
        cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def video_player(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Frame rate and total frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps)  # Convert to milliseconds
    current_frame_index = 0

    is_paused = False

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            current_frame_index += 1

            # Display the legend on the frame
            display_legend(frame)

            # Show the frame
            cv2.imshow("Video Player", frame)

        # Wait for user input
        key = cv2.waitKey(delay if not is_paused else 0) & 0xFF

        if key == 27:  # ESC key to exit
            break
        elif key == ord(' '):  # Spacebar to toggle play/pause
            is_paused = not is_paused
        elif key == ord('x'):  # Step forward (X key)
            is_paused = True
            if current_frame_index < total_frames - 1:
                current_frame_index += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if ret:
                    display_legend(frame)
                    cv2.imshow("Video Player", frame)
        elif key == ord('z'):  # Step backward (Z key)
            is_paused = True
            if current_frame_index > 1:
                current_frame_index -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index - 1)
                ret, frame = cap.read()
                if ret:
                    display_legend(frame)
                    cv2.imshow("Video Player", frame)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python video_player.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    video_player(video_file)