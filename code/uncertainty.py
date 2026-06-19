"""
Unified Uncertainty Quantification API

This module provides a high-level interface for computing uncertainty from
spatial predictions. It combines all uncertainty methods into a single API.

Pipeline:
    1. Sample multiple predictions from model
    2. Create spatial heatmap from samples
    3. Extract connected regions
    4. Compute uncertainty scores

Usage:
    from uncertainty import compute_all_uncertainties

    uncertainties = compute_all_uncertainties(
        sampled_coords=[(100, 200), (105, 195), ...],
        resized_width=560,
        resized_height=840
    )
    # Returns: {'margin': 0.3, 'entropy': 0.5, 'concentration': 0.4, 'combined': 0.41}
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from heatmap import create_heatmap_from_samples
from regions import extract_regions_from_heatmap
from margin import compute_region_margin
from entropy import compute_region_entropy
from concentration import compute_region_concentration
from combined import compute_combined_uncertainty, DEFAULT_WEIGHTS

DEFAULT_PATCH_SIZE = 28
DEFAULT_ACTIVATION_THRESHOLD = 0.0


@dataclass
class UncertaintyResult:
    """Complete uncertainty computation result."""
    margin: float
    entropy: float
    concentration: float
    combined: float
    n_regions: int
    region_scores: List[float]
    heatmap_shape: Tuple[int, int]


def compute_all_uncertainties(
    sampled_coords: List[Tuple[float, float]],
    resized_width: int,
    resized_height: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
) -> Dict[str, float]:
    """
    Compute all uncertainty measures from sampled coordinates.

    This is the main entry point for uncertainty quantification.

    Args:
        sampled_coords: List of (x, y) coordinates from model predictions
        resized_width: Width of resized image
        resized_height: Height of resized image
        patch_size: Patch size for heatmap (default: 28)
        activation_threshold: Threshold for region activation (default: 0.0)

    Returns:
        Dictionary with all uncertainty values:
            - margin: Margin-based uncertainty
            - entropy: Entropy-based uncertainty
            - concentration: Concentration-based uncertainty
            - combined: Weighted combination

    Examples:
        >>> coords = [(100, 200), (105, 195), (98, 205), (102, 198)]
        >>> unc = compute_all_uncertainties(coords, 560, 840)
        >>> unc['margin']
        0.15
        >>> unc['combined']
        0.32
    """
    heatmap, heatmap_prob, n_height, n_width = create_heatmap_from_samples(
        sampled_coords, resized_width, resized_height, patch_size
    )

    region_scores, _ = extract_regions_from_heatmap(
        heatmap_prob, activation_threshold
    )

    margin_u = compute_region_margin(region_scores)
    entropy_u = compute_region_entropy(region_scores)
    concentration_u = compute_region_concentration(region_scores)
    combined_u = compute_combined_uncertainty(region_scores)

    return {
        'margin': margin_u,
        'entropy': entropy_u,
        'concentration': concentration_u,
        'combined': combined_u
    }


def compute_uncertainty(
    sampled_coords: List[Tuple[float, float]],
    resized_width: int,
    resized_height: int,
    method: str = 'combined',
    patch_size: int = DEFAULT_PATCH_SIZE,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
) -> float:
    """
    Compute uncertainty using a specific method.

    Args:
        sampled_coords: List of (x, y) coordinates
        resized_width: Width of resized image
        resized_height: Height of resized image
        method: Uncertainty method ('margin', 'entropy', 'concentration', 'combined')
        patch_size: Patch size for heatmap
        activation_threshold: Threshold for region activation (default: 0.0)

    Returns:
        Uncertainty value in [0, 1]
    """
    heatmap, heatmap_prob, n_height, n_width = create_heatmap_from_samples(
        sampled_coords, resized_width, resized_height, patch_size
    )

    region_scores, _ = extract_regions_from_heatmap(
        heatmap_prob, activation_threshold
    )

    if method == 'margin':
        return compute_region_margin(region_scores)
    elif method == 'entropy':
        return compute_region_entropy(region_scores)
    elif method == 'concentration':
        return compute_region_concentration(region_scores)
    elif method == 'combined':
        return compute_combined_uncertainty(region_scores)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_uncertainty_detailed(
    sampled_coords: List[Tuple[float, float]],
    resized_width: int,
    resized_height: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
) -> UncertaintyResult:
    """
    Compute all uncertainties with detailed information.

    Args:
        sampled_coords: List of (x, y) coordinates
        resized_width: Width of resized image
        resized_height: Height of resized image
        patch_size: Patch size for heatmap
        activation_threshold: Threshold for region activation (default: 0.0)

    Returns:
        UncertaintyResult with all values and metadata
    """
    heatmap, heatmap_prob, n_height, n_width = create_heatmap_from_samples(
        sampled_coords, resized_width, resized_height, patch_size
    )

    region_scores, _ = extract_regions_from_heatmap(
        heatmap_prob, activation_threshold
    )

    margin_u = compute_region_margin(region_scores)
    entropy_u = compute_region_entropy(region_scores)
    concentration_u = compute_region_concentration(region_scores)
    combined_u = compute_combined_uncertainty(region_scores)

    return UncertaintyResult(
        margin=margin_u,
        entropy=entropy_u,
        concentration=concentration_u,
        combined=combined_u,
        n_regions=len(region_scores),
        region_scores=region_scores,
        heatmap_shape=(n_height, n_width)
    )


def check_hit(
    predicted_point: Tuple[float, float],
    ground_truth_bbox: List[float]
) -> bool:
    """
    Check if predicted point is inside ground truth bounding box.

    Args:
        predicted_point: (x, y) predicted coordinates
        ground_truth_bbox: [x1, y1, x2, y2] bounding box

    Returns:
        True if point is inside bbox
    """
    if predicted_point is None:
        return False

    x, y = predicted_point
    x1, y1, x2, y2 = ground_truth_bbox

    inside = (x1 <= x <= x2) and (y1 <= y <= y2)
    return inside


def get_available_methods() -> List[str]:
    """
    Get list of available uncertainty methods.

    Returns:
        List of method names
    """
    return ['margin', 'entropy', 'concentration', 'combined']


def get_method_description(method: str) -> str:
    """
    Get description of an uncertainty method.

    Args:
        method: Method name

    Returns:
        Description string
    """
    descriptions = {
        'margin': 'Margin-based uncertainty - measures gap between top-2 region scores',
        'entropy': 'Entropy-based uncertainty - measures distribution entropy',
        'concentration': 'Concentration-based uncertainty - measures HHI complement',
        'combined': 'Combined uncertainty - weighted combination of all methods'
    }
    return descriptions.get(method, 'Unknown method')
