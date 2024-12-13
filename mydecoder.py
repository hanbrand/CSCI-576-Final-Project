import cv2
import sys
import numpy as np
import wave
import pyaudio
import time
import threading

# Constants
width = 960
height = 544
fps = 30
frame_duration = 1.0 / fps  # seconds per frame

# Control variables
playback_state = {"play": True, "stop": False, "step": False, "reset": False}

def dequantize(block, n):
    q = 2 ** n
    return block * q

def process_frame(compressed_data, n1, n2):
    # Initialize frame in BGR color space
    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    
    block_idx = 0
    # Process each 8x8 block
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if block_idx >= len(compressed_data):
                break
                
            block_flag, coeffs = compressed_data[block_idx]
            
            # Choose quantization parameter based on block type
            n = n1 if block_flag == 1 else n2
            
            # Process each channel
            for c in range(3):
                # Get coefficients for this channel
                channel_coeffs = coeffs[c * 64:(c + 1) * 64]
                block = np.array(channel_coeffs).reshape(8, 8)
                
                # Dequantize
                block = dequantize(block, n)
                
                # Apply inverse DCT
                block = cv2.idct(np.float32(block))
                
                # Add back DC offset and clip to valid range
                block = block + 128
                block = np.clip(block, 0, 255)
                
                # Store in frame
                frame_bgr[y:y+8, x:x+8, c] = block.astype(np.uint8)
                
            block_idx += 1
            
    return frame_bgr

def preload_frames(compressed_file):
    frames = []
    with open(compressed_file, 'rb') as cmp_file:
        # Read quantization parameters
        n1, n2 = np.frombuffer(cmp_file.read(2), dtype=np.uint8)
        
        while True:
            compressed_data = []
            blocks_per_frame = (width // 8) * (height // 8)
            
            # Read all blocks for current frame
            for _ in range(blocks_per_frame):
                try:
                    block_flag = np.frombuffer(cmp_file.read(1), dtype=np.uint8)[0]
                    coeffs = np.frombuffer(cmp_file.read(64*3*2), dtype=np.int16)  # 64 coeffs * 3 channels * 2 bytes
                    compressed_data.append((block_flag, coeffs))
                except:
                    break
            
            if not compressed_data:
                break
            
            # Decompress frame
            frame_bgr = process_frame(compressed_data, n1, n2)
            frames.append(frame_bgr)
    
    return frames

def play_audio(audio_file, stop_event, audio_offset=0):
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()

    # Seek to the correct position based on audio_offset
    if audio_offset > 0:
        frame_offset = int(audio_offset * wf.getframerate())
        wf.setpos(frame_offset)

    # Open audio stream
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    # Play audio
    while not stop_event.is_set():
        if not playback_state["play"]:
            time.sleep(0.1)
            continue

        data = wf.readframes(1024)
        if data == b'':
            break
        stream.write(data)

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    global playback_state

    if len(sys.argv) < 3:
        print("Usage: python mydecoder.py <input_video.cmp> <input_audio.wav>")
        return

    compressed_file = sys.argv[1]
    audio_file = sys.argv[2]

    print("Decompressing frames")
    frames = preload_frames(compressed_file)
    print("Done with decompression")

    # Start audio playback
    stop_event = threading.Event()
    audio_thread = threading.Thread(target=play_audio, args=(audio_file, stop_event))
    audio_thread.start()

    start_time = time.time()
    current_frame = 0

    while not playback_state["stop"]:
        if playback_state["reset"]:
            current_frame = 0
            start_time = time.time()
            stop_event.set()  # Stop the current audio thread
            audio_thread.join()
            stop_event.clear()  # Clear stop flag
            audio_thread = threading.Thread(target=play_audio, args=(audio_file, stop_event))
            audio_thread.start()
            playback_state["reset"] = False
            continue

        if current_frame >= len(frames):
            break

        # Handle play, pause, and stepping logic
        if playback_state["play"]:
            elapsed_time = time.time() - start_time
            expected_frame = int(elapsed_time * fps)

            # Skip or wait to synchronize
            if current_frame < expected_frame:
                current_frame += 1
                continue
            elif current_frame > expected_frame:
                time.sleep(frame_duration * (current_frame - expected_frame))

            # Display current frame
            cv2.imshow("A/V Player", frames[current_frame])
            current_frame += 1
        elif playback_state["step"]:
            # Stop the audio thread
            stop_event.set()
            audio_thread.join()
            stop_event.clear()

            # Step the video forward
            if current_frame < len(frames):
                cv2.imshow("A/V Player", frames[current_frame])
                current_frame += 1
            playback_state["step"] = False

            # Calculate new audio offset (move forward by one frame duration)
            audio_offset = current_frame / fps
            start_time = time.time() - audio_offset

            # Restart audio thread synced to the new offset
            audio_thread = threading.Thread(target=play_audio, args=(audio_file, stop_event, audio_offset))
            audio_thread.start()
        else:
            # Pause logic
            pause_start_time = time.time()
            while not playback_state["play"]:  # Wait until unpause
                key = cv2.waitKey(1)
                if key == ord('p'):
                    playback_state["play"] = True
                elif key == ord('q'):
                    playback_state["stop"] = True
                    break
                elif key == ord('r'):
                    playback_state["reset"] = True
                    break
                elif key == ord('s'):  # Allow stepping while paused
                    playback_state["step"] = True
                    break
            # Adjust start_time to account for pause duration
            start_time += time.time() - pause_start_time

        # Handle user input
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit
            playback_state["stop"] = True
        elif key == ord('p'):  # Pause/Play toggle
            playback_state["play"] = not playback_state["play"]
        elif key == ord('s'):  # Step forward
            if not playback_state["play"] and current_frame < len(frames):
                playback_state["step"] = True
        elif key == ord('r'):  # Reset
            playback_state["reset"] = True

    # Stop audio thread
    stop_event.set()
    audio_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()