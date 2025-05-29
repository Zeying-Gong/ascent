import os
from typing import List, Optional

import cv2
import numpy as np
from numba import njit

from frontier_exploration.utils.bresenham_line import bresenhamline
from frontier_exploration.utils.frontier_utils import closest_line_segment

VISUALIZE = os.environ.get("MAP_VISUALIZE", "False").lower() == "true"
DEBUG = os.environ.get("MAP_DEBUG", "False").lower() == "true"


def detect_frontier_waypoints(
    full_map: np.ndarray,
    explored_mask: np.ndarray,
    area_thresh: Optional[int] = -1,
    xy: Optional[np.ndarray] = None,
):
    DEBUG = False
    if DEBUG:
        import time

        os.makedirs("debug/20241129/map_debug", exist_ok=True)
        # 创建一个彩色图像来清楚标记 full_map 和 explored_mask 的值
        visualization = np.zeros((*full_map.shape, 3), dtype=np.uint8)

        # 标记 full_map 的不同区域
        visualization[full_map == 1] = (0, 255, 0)  # Navigable (绿色)
        visualization[full_map == 0] = (255, 0, 0)  # Non-navigable (红色)

        # 将 explored_mask 的已探索部分叠加为灰色
        visualization[explored_mask > 0] = (127, 127, 127)  # Explored (灰色)

        # 保存清晰的可视化图像
        cv2.imwrite(
            f"debug/20241129/map_debug/{int(time.time())}_debug_visualization_{area_thresh}.png",
            visualization,
        )
        print("Saved detailed visualization for full_map and explored_mask.")
        
        # cv2.imwrite(
        #     f"debug/20241129/map_debug/{int(time.time())}_debug_full_map_{area_thresh}.png", full_map
        # )
        # cv2.imwrite(
        #     f"debug/20241129/map_debug/{int(time.time())}_debug_explored_mask_{area_thresh}.png",
        #     explored_mask,
        # )

    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)

        cv2.imshow("inputs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    explored_mask[full_map == 0] = 0
    frontiers = detect_frontiers(full_map, explored_mask, area_thresh)
    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        # Draw a dot at each point on each frontier
        for idx, frontier in enumerate(frontiers):
            # Uniformly sample colors from the COLORMAP_RAINBOW
            color = cv2.applyColorMap(
                np.uint8([255 * (idx + 1) / len(frontiers)]), cv2.COLORMAP_RAINBOW
            )[0][0]
            color = tuple(int(i) for i in color)
            for idx2, p in enumerate(frontier):
                if idx2 < len(frontier) - 1:
                    cv2.line(img, p[0], frontier[idx2 + 1][0], color, 3)
        cv2.imshow("frontiers", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    waypoints = frontier_waypoints(frontiers, xy)
    return waypoints


def detect_frontiers(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: Optional[int] = -1
) -> List[np.ndarray]:
    """Detects frontiers in a map.

    Args:
        full_map (np.ndarray): White polygon on black image, where white is navigable.
        Mono-channel mask.
        explored_mask (np.ndarray): Portion of white polygon that has been seen already.
        This is also a mono-channel mask.
        area_thresh (int, optional): Minimum unexplored area (in pixels) needed adjacent
        to a frontier for that frontier to be valid. Defaults to -1.

    Returns:
        np.ndarray: A mono-channel mask where white contours represent each frontier.
    """
    # Find the contour of the explored area
    filtered_explored_mask = filter_out_small_unexplored(
        full_map, explored_mask, area_thresh
    )
    contours, _ = cv2.findContours(
        filtered_explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.imshow("contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
    unexplored_mask = cv2.blur(  # blurring for some leeway
        np.where(unexplored_mask > 0, 255, unexplored_mask), (3, 3)
    )
    frontiers = []
    # TODO: There shouldn't be more than one contour (only one explored area on map)
    for contour in contours:
        frontiers.extend(
            contour_to_frontiers(interpolate_contour(contour), unexplored_mask)
        )
    return frontiers


def filter_out_small_unexplored(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int
):
    """Edit the explored map to add small unexplored areas, which ignores their
    frontiers."""
    if area_thresh == -1:
        return explored_mask

    unexplored_mask = full_map.copy()
    unexplored_mask[explored_mask > 0] = 0

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("unexplored mask", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours in the unexplored mask
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        # Draw the contours in red
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imshow("unexplored mask with contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    DEBUG = False
    if DEBUG:
        import time
        os.makedirs("debug/20241130", exist_ok=True)
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        # Draw the contours in red
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        filename = f"debug/20241130/{int(time.time())}_unexplored_contours.png"
        cv2.imwrite(filename, img)
        print(f"Saved unexplored mask with contours: {filename}")

    # Add small unexplored areas to the explored map
    small_contours = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < area_thresh:
            mask = np.zeros_like(explored_mask)
            mask = cv2.drawContours(mask, [contour], 0, 1, -1)
            masked_values = unexplored_mask[mask.astype(bool)]
            values = set(masked_values.tolist())
            if 1 in values and len(values) == 1:
                small_contours.append(contour)
    new_explored_mask = explored_mask.copy()
    cv2.drawContours(new_explored_mask, small_contours, -1, 255, -1)

    if VISUALIZE and len(small_contours) > 0:
        # Draw the full map and the new explored mask, then outline the contours that
        # were added to the explored mask
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[new_explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, small_contours, -1, (0, 0, 255), 3)
        cv2.imshow("small unexplored areas", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if DEBUG and len(small_contours) > 0:
        os.makedirs("debug/small_unexplored_areas", exist_ok=True)
        # Draw the full map and the new explored mask, then outline the contours that
        # were added to the explored mask
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[new_explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, small_contours, -1, (0, 0, 255), 3)

        # Add labels and log areas
        for idx, contour in enumerate(small_contours):
            # Calculate the centroid of the contour for labeling
            moments = cv2.moments(contour)
            if moments["m00"] != 0:  # Avoid division by zero
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
            else:
                cX, cY = contour[0][0]  # Fallback to the first point

            # Calculate the area
            area = cv2.contourArea(contour)

            # Log the area to the console
            print(f"Contour {idx + 1}: Area = {area}")

            # Add text label to the image
            cv2.putText(
                img,
                f"{idx + 1}",
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (255, 255, 255),  # Font color (white)
                2,  # Thickness
            )

        filename = f"debug/small_unexplored_areas/{int(time.time())}_small_unexplored_areas.png"
        cv2.imwrite(filename, img)
        print(f"Saved small unexplored areas visualization: {filename}")

    return new_explored_mask


def interpolate_contour(contour):
    """Given a cv2 contour, this function will add points in between each pair of
    points in the contour using the bresenham algorithm to make the contour more
    continuous.
    :param contour: A cv2 contour of shape (N, 1, 2)
    :return:
    """
    # First, reshape and expand the frontier to be a 2D array of shape (N-1, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((contour[:-1], contour[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Also add a segment connecting the last point to the first point
    line_segments = np.concatenate(
        (line_segments, np.array([contour[-1], contour[0]]).reshape((1, 2, 2)))
    )
    pts = []
    for (x0, y0), (x1, y1) in line_segments:
        pts.append(
            bresenhamline(np.array([[x0, y0]]), np.array([[x1, y1]]), max_iter=-1)
        )
    pts = np.concatenate(pts).reshape((-1, 1, 2))
    return pts


@njit
def contour_to_frontiers(contour, unexplored_mask):
    """Given a contour from OpenCV, return a list of numpy arrays. Each array contains
    contiguous points forming a single frontier. The contour is assumed to be a set of
    contiguous points, but some of these points are not on any frontier, indicated by
    having a value of 0 in the unexplored mask. This function will split the contour
    into multiple arrays that exclude such points."""
    bad_inds = []
    num_contour_points = len(contour)
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    # np.split is fast but does NOT remove the element at the split index
    filtered_frontiers = []
    front_last_split = (
        0 not in bad_inds
        and len(bad_inds) > 0
        and max(bad_inds) < num_contour_points - 2
    )
    for idx, f in enumerate(frontiers):
        # a frontier must have at least 2 points (3 with bad ind)
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])
    # Combine the first and last frontier if the first point of the first frontier and
    # the last point of the last frontier are the first and last points of the original
    # contour. Only check if there are at least 2 frontiers.
    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
    return filtered_frontiers


def frontier_waypoints(
    frontiers: List[np.ndarray], xy: Optional[np.ndarray] = None
) -> np.ndarray:
    """For each given frontier, returns the point on the frontier closest (euclidean
    distance) to the given coordinate. If coordinate is not given, will just return
    the midpoints of each frontier.

    Args:
        frontiers (List[np.ndarray]): list of arrays of shape (X, 1, 2), where each
        array is a frontier and X is NOT the same across arrays
        xy (np.ndarray): the given coordinate

    Returns:
        np.ndarray: array of waypoints, one for each frontier
    """
    if xy is None:
        return np.array([get_frontier_midpoint(i) for i in frontiers])
    return np.array([get_closest_frontier_point(xy, i) for i in frontiers])


@njit
def get_frontier_midpoint(frontier) -> np.ndarray:
    """Given a list of contiguous points (numpy arrays) representing a frontier, first
    calculate the total length of the frontier, then find the midpoint of the
    frontier"""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate((frontier[:-1], frontier[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # Calculate the length of each line segment
    line_lengths = np.sqrt(
        np.square(line_segments[:, 0, 0] - line_segments[:, 1, 0])
        + np.square(line_segments[:, 0, 1] - line_segments[:, 1, 1])
    )
    cum_sum = np.cumsum(line_lengths)
    total_length = cum_sum[-1]
    # Find the midpoint of the frontier
    half_length = total_length / 2
    # Find the line segment that contains the midpoint
    line_segment_idx = np.argmax(cum_sum > half_length)
    # Calculate the coordinates of the midpoint
    line_segment = line_segments[line_segment_idx]
    line_length = line_lengths[line_segment_idx]
    # Use the difference between the midpoint length and cumsum
    # to find the proportion of the line segment that the midpoint is at
    length_up_to = cum_sum[line_segment_idx - 1] if line_segment_idx > 0 else 0
    proportion = (half_length - length_up_to) / line_length
    # Calculate the midpoint coordinates
    midpoint = line_segment[0] + proportion * (line_segment[1] - line_segment[0])
    return midpoint


def get_closest_frontier_point(xy, frontier):
    """Returns the point on the frontier closest to the given coordinate."""
    # First, reshape and expand the frontier to be a 2D array of shape (X, 2)
    # representing line segments between adjacent points
    line_segments = np.concatenate([frontier[:-1], frontier[1:]], axis=1).reshape(
        (-1, 2, 2)
    )
    closest_segment, closest_point = closest_line_segment(xy, line_segments)
    return closest_point


if __name__ == "__main__":
    import argparse
    import time

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--full_map", help="Path to full map image", default="full_map.png"
    )
    parser.add_argument(
        "-e",
        "--explored_mask",
        help="Path to explored map image",
        default="explored_mask.png",
    )
    parser.add_argument(
        "-a",
        "--area_thresh",
        help="Minimum unexplored area (in pixels) needed adjacent to a frontier for"
        "that frontier to be valid",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        help="Number of iterations to run the algorithm for timing purposes. Set to "
        "0 for no timing",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    if VISUALIZE:
        args.num_iterations = 0

    # Read in the map
    full_map = cv2.imread(args.full_map, 0)
    # Read in the explored map
    explored_mask = cv2.imread(args.explored_mask, 0)
    times = []
    for _ in range(args.num_iterations + 1):
        start_time = time.time()
        waypoints = detect_frontier_waypoints(full_map, explored_mask, args.area_thresh)
        times.append(time.time() - start_time)
    if args.num_iterations > 0:
        # Skip first run as it's slower due to JIT compilation
        print(
            f"Avg. time taken for algorithm over {args.num_iterations} runs:",
            np.mean(times[1:]),
        )

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(full_map, cmap="gray")
    plt.imshow(explored_mask, cmap="gray", alpha=0.5)
    for waypoint in waypoints:
        plt.scatter(waypoint[0], waypoint[1], c="red", s=50)
    plt.show()
