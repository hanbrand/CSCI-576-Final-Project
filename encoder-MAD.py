import cv2
import sys
import numpy as np
from collections import defaultdict

# Parameters
width = 960
height = 540
block_size = 16
frame_size = width * height * 3
search_range = block_size  # Search window size
vector_similarity_threshold = 2.0
mad_threshold = 1000

def compute_mad(block1, block2):
    return np.sum(np.abs(block1 - block2))

def get_neighbors(y, x, height, width):
    neighbors = []
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height//block_size and 0 <= nx < width//block_size:
            neighbors.append((ny, nx))
    return neighbors

def cluster_motion_vectors(motion_vectors, height, width):
    visited = np.zeros((height//block_size, width//block_size), dtype=bool)
    clusters = []
    
    for y in range(height//block_size):
        for x in range(width//block_size):
            if not visited[y,x]:
                cluster = []
                stack = [(y,x)]
                vector = motion_vectors[y,x]
                
                while stack:
                    cy, cx = stack.pop()
                    if not visited[cy,cx]:
                        visited[cy,cx] = True
                        curr_vector = motion_vectors[cy,cx]
                        
                        if np.linalg.norm(curr_vector - vector) < vector_similarity_threshold:
                            cluster.append((cy,cx))
                            for ny, nx in get_neighbors(cy, cx, height, width):
                                if not visited[ny,nx]:
                                    stack.append((ny,nx))
                
                if cluster:
                    clusters.append(cluster)
    
    return clusters

# Main processing loop
with open(sys.argv[1], 'rb') as f:
    raw_data = f.read(frame_size)
    prev_frame_rgb = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
    prev_frame_bgr = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2BGR)
    green_block = np.full((block_size, block_size, 3), [0, 255, 0], dtype=np.uint8)

    while True:
        raw_data = f.read(frame_size)
        if len(raw_data) != frame_size:
            break

        curr_frame_rgb = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        curr_frame_bgr = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2BGR)
        segmented_frame = np.zeros_like(curr_frame_bgr)
        
        # Store motion vectors
        motion_vectors = np.zeros((height//block_size, width//block_size, 2))

        # Compute motion vectors
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                min_mad = float('inf')
                best_match = (0, 0)
                curr_block = curr_frame_bgr[y:y+block_size, x:x+block_size]

                for j in range(max(0, y-search_range), min(height-block_size, y+search_range)+1, block_size):
                    for i in range(max(0, x-search_range), min(width-block_size, x+search_range)+1, block_size):
                        prev_block = prev_frame_bgr[j:j+block_size, i:i+block_size]
                        mad = compute_mad(curr_block, prev_block)
                        if mad < min_mad:
                            min_mad = mad
                            best_match = (i, j)

                motion_vector = np.array([best_match[0] - x, best_match[1] - y])
                motion_vectors[y//block_size, x//block_size] = motion_vector

        # Cluster motion vectors
        clusters = cluster_motion_vectors(motion_vectors, height, width)

        # Identify background clusters and segment
        for cluster in clusters:
            vectors = [motion_vectors[y,x] for y,x in cluster]
            mean_vector = np.mean(vectors, axis=0)
            
            # Check if background (near zero motion or consistent motion)
            is_background = np.linalg.norm(mean_vector) < 2 or \
                          np.all([np.linalg.norm(v - mean_vector) < vector_similarity_threshold for v in vectors])
            
            for y, x in cluster:
                if is_background:
                    segmented_frame[y*block_size:(y+1)*block_size, 
                                  x*block_size:(x+1)*block_size] = green_block
                else:
                    segmented_frame[y*block_size:(y+1)*block_size, 
                                  x*block_size:(x+1)*block_size] = curr_frame_bgr[y*block_size:(y+1)*block_size, 
                                                                                x*block_size:(x+1)*block_size]

        combined_frame = np.hstack((curr_frame_bgr, segmented_frame))
        cv2.imshow('Original and Segmented', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame_bgr = curr_frame_bgr

cv2.destroyAllWindows()