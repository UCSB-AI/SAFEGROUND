"""
Region Extraction Module

Extracts connected regions from a probability heatmap using Breadth-First Search (BFS).
This identifies spatially coherent clusters of high-probability patches.

Key Steps:
1. Apply activation threshold relative to max probability
2. Find connected components using 4-connectivity BFS
3. Compute region statistics (average score, center, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional

DEFAULT_ACTIVATION_THRESHOLD = 0.0


def extract_regions_from_heatmap(
    heatmap_prob: np.ndarray,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """
    Extract connected regions from heatmap using BFS.

    Identifies spatially contiguous regions where patch probabilities exceed
    a threshold relative to the maximum probability. Each region is then
    characterized by its average probability and center of mass.

    Args:
        heatmap_prob: Normalized probability heatmap (n_height, n_width)
        activation_threshold: Fraction of max probability to consider as
                             activated (default: 0.0). With the default,
                             every patch with positive probability is active.

    Returns:
        region_scores: List of region scores (average probability per region)
        region_centers: List of region centers as (x, y) in normalized coordinates

    Example:
        >>> import numpy as np
        >>> heatmap = np.array([[0.1, 0.8, 0.1],
        ...                     [0.1, 0.7, 0.1]])
        >>> scores, centers = extract_regions_from_heatmap(heatmap)
        >>> len(scores)  # Number of detected regions
    """
    n_height, n_width = heatmap_prob.shape
    max_prob = heatmap_prob.max()

    if max_prob == 0:
        return [], []

    threshold = max_prob * activation_threshold
    mask = heatmap_prob > threshold

    activated_patches = []
    for y in range(n_height):
        for x in range(n_width):
            if mask[y, x]:
                activated_patches.append((y, x, heatmap_prob[y, x]))

    if len(activated_patches) == 0:
        return [], []

    regions = []
    visited = set()

    for y, x, prob in activated_patches:
        if (y, x) in visited:
            continue

        region = [(y, x, prob)]
        visited.add((y, x))
        queue = [(y, x)]

        while queue:
            cy, cx = queue.pop(0)

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx

                if (0 <= ny < n_height and 0 <= nx < n_width and
                    (ny, nx) not in visited and mask[ny, nx]):
                    visited.add((ny, nx))
                    region.append((ny, nx, heatmap_prob[ny, nx]))
                    queue.append((ny, nx))

        regions.append(region)

    region_scores = []
    region_centers = []

    for region in regions:
        avg_score = sum(prob for _, _, prob in region) / len(region)
        region_scores.append(avg_score)

        total_weight = sum(prob for _, _, prob in region)
        weighted_x = sum((x + 0.5) / n_width * prob for _, x, prob in region) / total_weight
        weighted_y = sum((y + 0.5) / n_height * prob for y, _, prob in region) / total_weight
        region_centers.append((weighted_x, weighted_y))

    sorted_indices = sorted(
        range(len(region_scores)),
        key=lambda i: region_scores[i],
        reverse=True
    )
    sorted_scores = [region_scores[i] for i in sorted_indices]
    sorted_centers = [region_centers[i] for i in sorted_indices]

    return sorted_scores, sorted_centers


def bfs_connected_components(
    mask: np.ndarray
) -> List[List[Tuple[int, int]]]:
    """
    Extract all connected components from a binary mask.

    Uses BFS with 4-connectivity (up, down, left, right).

    Args:
        mask: Binary mask where True indicates activated pixels

    Returns:
        List of connected components, each as list of (y, x) coordinates
    """
    h, w = mask.shape
    visited = set()
    components = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] and (y, x) not in visited:
                component = []
                queue = [(y, x)]
                visited.add((y, x))

                while queue:
                    cy, cx = queue.pop(0)
                    component.append((cy, cx))

                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx

                        if (0 <= ny < h and 0 <= nx < w and
                            mask[ny, nx] and (ny, nx) not in visited):
                            visited.add((ny, nx))
                            queue.append((ny, nx))

                components.append(component)

    return components


def compute_region_statistics(
    component: List[Tuple[int, int]],
    heatmap_prob: np.ndarray,
    n_width: int,
    n_height: int
) -> dict:
    """
    Compute statistics for a single region.

    Args:
        component: List of (y, x) patch coordinates
        heatmap_prob: Probability heatmap
        n_width: Number of patches horizontally
        n_height: Number of patches vertically

    Returns:
        Dictionary with region statistics
    """
    probs = [heatmap_prob[y, x] for y, x in component]

    avg_score = sum(probs) / len(probs)

    total_weight = sum(probs)
    weighted_x = sum((x + 0.5) / n_width * p for (y, x), p in zip(component, probs)) / total_weight
    weighted_y = sum((y + 0.5) / n_height * p for (y, x), p in zip(component, probs)) / total_weight

    center_x = sum(x for _, x in component) / len(component) / n_width
    center_y = sum(y for y, _ in component) / len(component) / n_height

    return {
        'size': len(component),
        'average_score': avg_score,
        'center': (weighted_x, weighted_y),
        'simple_center': (center_x, center_y),
        'max_prob': max(probs),
        'sum_prob': sum(probs)
    }


def sort_regions_by_score(
    region_scores: List[float],
    region_centers: List[Tuple[float, float]]
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """
    Sort regions by score in descending order.

    Args:
        region_scores: List of region scores
        region_centers: List of region centers

    Returns:
        Tuple of (sorted_scores, sorted_centers) both sorted by score
    """
    sorted_indices = sorted(
        range(len(region_scores)),
        key=lambda i: region_scores[i],
        reverse=True
    )
    return (
        [region_scores[i] for i in sorted_indices],
        [region_centers[i] for i in sorted_indices]
    )
