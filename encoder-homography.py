import cv2
import numpy as np
import sys
from scipy.ndimage import gaussian_filter

width = 960
height = 540
fps = 30
block_size = 15
frame_size = width * height * 3

def estimate_global_homography(prev_gray, curr_gray):
    # Use ORB (or SIFT if available) for feature detection and description
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    if des1 is None or des2 is None:
        return None
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches[:300] if len(matches) > 300 else matches
    if len(good_matches) < 4:
        return None
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
    # Find homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    return H

def warp_frame_with_homography(prev_frame, H, width, height):
    if H is None:
        # If no homography found, return prev_frame as is
        return prev_frame
    warped = cv2.warpPerspective(prev_frame, H, (width, height))
    return warped

def segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, diff_threshold):
    curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_warped = cv2.cvtColor(warped_prev_bgr, cv2.COLOR_BGR2GRAY)
    
    # Smooth to reduce noise
    curr_gray = gaussian_filter(curr_gray, sigma=1)
    prev_gray_warped = gaussian_filter(prev_gray_warped, sigma=1)
    
    h, w = curr_gray.shape
    segmented_frame = curr_frame_bgr.copy()
    
    rows = h // block_size
    cols = w // block_size
    
    for r in range(rows):
        for c in range(cols):
            by = r * block_size
            bx = c * block_size
            
            # If this block is within the edge margin, mark as background
            if (r < 1 or r >= rows - 1 or 
                c < 1 or c >= cols - 1):
                segmented_frame[by:by+block_size, bx:bx+block_size] = (0, 255, 0)
                continue
            
            # For non-edge blocks, proceed with difference check
            curr_block = curr_gray[by:by+block_size, bx:bx+block_size]
            prev_block = prev_gray_warped[by:by+block_size, bx:bx+block_size]
            
            mad = np.mean(np.abs(curr_block.astype(np.float32) - prev_block.astype(np.float32)))
            
            if mad < diff_threshold:
                # Background: paint green
                segmented_frame[by:by+block_size, bx:bx+block_size] = (0, 255, 0)
            # else: it's considered foreground and left as is
    
    return segmented_frame

with open(sys.argv[1], 'rb') as f:
    prev_frame_bgr = None

    while True:
        raw_frame = f.read(frame_size)
        if not raw_frame:
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        curr_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if prev_frame_bgr is not None:
            prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Estimate a global homography to handle perspective changes
            H = estimate_global_homography(prev_gray, curr_gray)
            
            # Warp previous frame to align with current frame perspective
            warped_prev_bgr = warp_frame_with_homography(prev_frame_bgr, H, width, height)
            
            # Segment foreground using residual differences
            segmented_frame = segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, diff_threshold=10)
            
            combined = np.hstack((curr_frame_bgr, segmented_frame))
            cv2.imshow('Original (Left) vs Segmented (Right)', combined)
        else:
            # First frame, no segmentation
            cv2.imshow('Original (Left) vs Segmented (Right)', np.hstack((curr_frame_bgr, curr_frame_bgr)))
        
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
        
        prev_frame_bgr = curr_frame_bgr.copy()

cv2.destroyAllWindows()
