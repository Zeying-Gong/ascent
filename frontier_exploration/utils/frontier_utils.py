from typing import Tuple

import numpy as np


def closest_line_segment(
    coord: np.ndarray, segments: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    closest_points = closest_point_on_segment(coord, segments[:, 0], segments[:, 1])
    # Identify the segment that yielded the closest point
    min_idx = np.argmin(np.linalg.norm(closest_points - coord, axis=1))
    closest_segment, closest_point = segments[min_idx], closest_points[min_idx]

    return closest_segment, closest_point


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    t = np.einsum("ij,ij->i", p - a, segment) / np.einsum("ij,ij->i", segment, segment)
    t = np.clip(t, 0, 1)
    return a + t[:, np.newaxis] * segment


if __name__ == "__main__":
    import time

    import cv2

    # Test data
    coord = np.array(np.random.rand(2) * 500, dtype=int)

    # Create a list of line segments using random sampling
    segments = (np.random.rand(10, 2, 2) * 500).astype(int)

    # Time the function
    start = time.perf_counter()
    for i in range(10000):
        closest = closest_line_segment(coord, segments)
    end = time.perf_counter()
    elapsed = end - start

    # Print the average execution time
    print(f"Average execution time: {elapsed / 10000:.6f} seconds")

    # Use OpenCV to draw the segments, highlighting the closest one
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw all segments in red
    for s in segments:
        cv2.line(img, tuple(s[0]), tuple(s[1]), (0, 0, 255), 1)
    # Draw the closest segment in green
    cv2.line(img, tuple(closest[0]), tuple(closest[1]), (0, 255, 0), 3)
    cv2.circle(img, coord, 10, (255, 0, 0), -1)
    # Display
    cv2.imshow("Closest segment", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
