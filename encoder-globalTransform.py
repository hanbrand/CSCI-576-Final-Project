import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

width = 960
height = 540
fps = 30
block_size = 15
frame_size = width * height * 3

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

def segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, diff_threshold=15):
    # Convert to gray for simpler difference measurement
    curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray_warped = cv2.cvtColor(warped_prev_bgr, cv2.COLOR_BGR2GRAY)

    # You could also apply some smoothing to reduce noise
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
            curr_block = curr_gray[by:by+block_size, bx:bx+block_size]
            prev_block = prev_gray_warped[by:by+block_size, bx:bx+block_size]

            # Compute difference
            mad = np.mean(np.abs(curr_block.astype(np.float32) - prev_block.astype(np.float32)))

            if mad < diff_threshold:
                # Background: paint green block
                segmented_frame[by:by+block_size, bx:bx+block_size] = (0, 255, 0)
            # else foreground remains original

    return segmented_frame

with open('./rgbs/WalkingMovingBackground.rgb', 'rb') as f:
    prev_frame_bgr = None

    while True:
        raw_frame = f.read(frame_size)
        if not raw_frame:
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        curr_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if prev_frame_bgr is not None:
            # Estimate global transform from prev to curr
            prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
            M = estimate_global_transform(prev_gray, curr_gray)

            # Warp previous frame according to global transform
            warped_prev_bgr = warp_frame(prev_frame_bgr, M, width, height)

            # Segment foreground/background using residual differences
            segmented_frame = segment_foreground_background(curr_frame_bgr, warped_prev_bgr, block_size, diff_threshold=15)

            # Show side-by-side
            combined = np.hstack((curr_frame_bgr, segmented_frame))
            cv2.imshow('Original (Left) vs Segmented (Right)', combined)
        else:
            # First frame, no segmentation, just display original twice
            cv2.imshow('Original (Left) vs Segmented (Right)', np.hstack((curr_frame_bgr, curr_frame_bgr)))

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        prev_frame_bgr = curr_frame_bgr.copy()

cv2.destroyAllWindows()
