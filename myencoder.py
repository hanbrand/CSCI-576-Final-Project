import cv2
import numpy as np
import sys
from scipy.ndimage import gaussian_filter

width = 960
height = 540
fps = 30
block_size = 16
frame_size = width * height * 3
n1 = int(sys.argv[2])  # Quantization exponent for foreground
n2 = int(sys.argv[3])  # Quantization exponent for background

def pad_frame(frame):
    padded_frame = cv2.copyMakeBorder(frame, 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return padded_frame

def estimate_global_transform(prev_frame_gray, curr_frame_gray):
    # Use ORB to find keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(prev_frame_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_frame_gray, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use a subset of the best matches
    good_matches = matches[:200] if len(matches) > 200 else matches

    if len(good_matches) < 4:
        # Not enough matches to compute a transform
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Estimate a partial affine transform: translation, rotation, scale
    # Use RANSAC to be robust to outliers
    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M

def warp_frame(prev_frame, M, width, height):
    # Warp the prev_frame to align it with the current frame according to M
    if M is None:
        # If no transform, just return the prev_frame
        return prev_frame
    warped = cv2.warpAffine(prev_frame, M, (width, height))
    return warped

def segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, diff_threshold):
    # Convert to gray for simpler difference measurement
    curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_warped = cv2.cvtColor(warped_prev_bgr, cv2.COLOR_BGR2GRAY)

    # Apply smoothing to reduce noise
    curr_gray = gaussian_filter(curr_gray, sigma=1)
    prev_gray_warped = gaussian_filter(prev_gray_warped, sigma=1)

    h, w = curr_gray.shape
    mask = np.ones((h, w), dtype=bool)  # Initialize mask

    rows = h // block_size
    cols = w // block_size

    for r in range(rows):
        for c in range(cols):
            by = r * block_size
            bx = c * block_size

            # Border blocks are background
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                mask[by:by+block_size, bx:bx+block_size] = False
                continue

            curr_block = curr_gray[by:by+block_size, bx:bx+block_size]
            prev_block = prev_gray_warped[by:by+block_size, bx:bx+block_size]

            # Compute MAD
            mad = np.mean(np.abs(curr_block.astype(np.float32) - prev_block.astype(np.float32)))

            if mad < diff_threshold:
                # Background
                mask[by:by+block_size, bx:bx+block_size] = False

    return mask

def quantize(block, n):
    q = 2 ** n
    quant_block = (block / q).round()
    return quant_block.astype(np.int16)

def process_frame(frame, segmentation):
    height, width, _ = frame.shape
    compressed_data = []

    # Process each 8x8 block in the BGR channels
    channels = cv2.split(frame)

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block_type = segmentation[y:y+8, x:x+8]
            if np.any(block_type > 0):
                n = n1  # Foreground
                block_flag = 1
            else:
                n = n2  # Background
                block_flag = 0

            coeffs = []
            for channel in channels:
                block = channel[y:y+8, x:x+8]
                if block.shape != (8,8):
                    block = cv2.copyMakeBorder(block, 0, 8 - block.shape[0], 0, 8 - block.shape[1], cv2.BORDER_CONSTANT, value=0)
                dct_block = cv2.dct(np.float32(block) - 128)
                quant_block = quantize(dct_block, n)
                coeffs.extend(quant_block.flatten())

            # Store block_type and quantized coefficients
            compressed_data.append((block_flag, coeffs))

    return compressed_data

print('Encoding video:', sys.argv[1])
# Open compressed file for writing
with open('input_video.cmp', 'wb') as cmp_file, open(sys.argv[1], 'rb') as f:
    prev_frame_bgr = None

    # Write quantization exponents n1 and n2 at the beginning
    cmp_file.write(np.array([n1, n2], dtype=np.uint8).tobytes())

    while True:
        raw_frame = f.read(frame_size)
        if not raw_frame:
            print('Done encoding')
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        curr_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Pad the frame
        curr_frame_bgr = pad_frame(curr_frame_bgr)

        if prev_frame_bgr is not None:
            # Estimate global transform from prev to curr
            prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
            M = estimate_global_transform(prev_gray, curr_gray)

            # Warp previous frame according to global transform
            warped_prev_bgr = warp_frame(prev_frame_bgr, M, width, height)

            # Segment foreground/background using residual differences
            mask = segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, 15)

            # Create masked display frame
            masked_frame = curr_frame_bgr.copy()
            masked_frame[~mask] = 0  # Set background pixels to black

            # # DEBUGGING: Show side-by-side
            # combined = np.hstack((curr_frame_bgr, masked_frame))
            # cv2.imshow('Original (Left) vs Segmented (Right)', combined)

            # Process and compress the current frame
            compressed_frame = process_frame(curr_frame_bgr, mask)

            # Write compressed data
            for block_flag, coeffs in compressed_frame:
                cmp_file.write(np.array([block_flag], dtype=np.uint8).tobytes())
                cmp_file.write(np.array(coeffs, dtype=np.int16).tobytes())

        # DEBUGGING:
        # else:
        #     # First frame, no segmentation, just display original twice
        #     cv2.imshow('Original (Left) vs Segmented (Right)', np.hstack((curr_frame_bgr, curr_frame_bgr)))

        # if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        #     break

        prev_frame_bgr = curr_frame_bgr.copy()

cv2.destroyAllWindows()