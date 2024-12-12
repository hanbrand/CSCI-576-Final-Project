import cv2
import numpy as np

# Constants
width = 960
height = 540
fps = 30

def dequantize(block, n):
    q = 2 ** n
    return block * q

def process_frame(compressed_data, n1, n2):
    # Initialize frame in YCrCb color space
    frame_ycrcb = np.zeros((height, width, 3), dtype=np.uint8)
    
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
                channel_coeffs = coeffs[c*64:(c+1)*64].reshape(8,8)
                
                # Dequantize
                dequant_block = dequantize(channel_coeffs, n)
                
                # Inverse DCT
                idct_block = cv2.idct(np.float32(dequant_block))
                block = idct_block + 128
                
                # Clip values to valid range
                block = np.clip(block, 0, 255).astype(np.uint8)
                
                # Place block in frame
                frame_ycrcb[y:y+8, x:x+8, c] = block
                
            block_idx += 1

    # Convert back to BGR
    frame_bgr = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
    return frame_bgr

# Read compressed file
with open('output.cmp', 'rb') as cmp_file:
    # Read quantization parameters
    n1, n2 = np.frombuffer(cmp_file.read(2), dtype=np.uint8)
    
    frame_count = 0
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
        
        # Display frame
        cv2.imshow('Decoded Frame', frame_bgr)
        frame_count += 1
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()