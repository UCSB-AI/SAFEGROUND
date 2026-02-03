"""
Combined Uncertainty


Default Weights:
- Margin: 0.2 (20%)
- Entropy: 0.2 (20%)
- Concentration: 0.6 (60%)

"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from SAFEGROUND.code.margin import compute_region_margin
from SAFEGROUND.code.entropy import compute_region_entropy
from SAFEGROUND.code.concentration import compute_region_concentration


DEFAULT_WEIGHTS = {
    'margin': 0.2,
    'entropy': 0.2,
    'concentration': 0.6
}


@dataclass
class CombinedUncertaintyResult:
    """Result of combined uncertainty computation."""
    uncertainty: float
    margin: float
    entropy: float
    concentration: float
    weights: Dict[str, float]


def compute_combined_uncertainty(
    sorted_scores: List[float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute combined uncertainty from region scores.

    Args:
        sorted_scores: Region scores sorted in descending order
        weights: Dictionary with weights for each method.
                 Default: {'margin': 0.2, 'entropy': 0.2, 'concentration': 0.6}

    Returns:
        Combined uncertainty value in [0, 1]

    Examples:
        >>> scores = [0.5, 0.3, 0.2]
        >>> compute_combined_uncertainty(scores)
        0.65

        >>> # Custom weights
        >>> compute_combined_uncertainty(scores, {'margin': 0.5, 'entropy': 0.3, 'concentration': 0.2})
        0.42
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    margin_u = compute_region_margin(sorted_scores)
    entropy_u = compute_region_entropy(sorted_scores)
    concentration_u = compute_region_concentration(sorted_scores)

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    combined = (
        normalized_weights['margin'] * margin_u +
        normalized_weights['entropy'] * entropy_u +
        normalized_weights['concentration'] * concentration_u
    )

    return combined


def compute_combined_uncertainty_detailed(
    sorted_scores: List[float],
    weights: Optional[Dict[str, float]] = None
) -> CombinedUncertaintyResult:
    """
    Compute combined uncertainty with detailed breakdown.

    Args:
        sorted_scores: Region scores sorted in descending order
        weights: Dictionary with weights for each method

    Returns:
        CombinedUncertaintyResult with all intermediate values
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    margin_u = compute_region_margin(sorted_scores)
    entropy_u = compute_region_entropy(sorted_scores)
    concentration_u = compute_region_concentration(sorted_scores)

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    combined = (
        normalized_weights['margin'] * margin_u +
        normalized_weights['entropy'] * entropy_u +
        normalized_weights['concentration'] * concentration_u
    )

    return CombinedUncertaintyResult(
        uncertainty=combined,
        margin=margin_u,
        entropy=entropy_u,
        concentration=concentration_u,
        weights=normalized_weights
    )


def get_default_weights() -> Dict[str, float]:
    """
    Get the default weights for combined uncertainty.

    Returns:
        Dictionary with default weights
    """
    return DEFAULT_WEIGHTS.copy()


def set_weights(margin: float = 0.2, entropy: float = 0.2, concentration: float = 0.6) -> Dict[str, float]:
    """
    Create custom weights dictionary.

    Args:
        margin: Weight for margin uncertainty
        entropy: Weight for entropy uncertainty
        concentration: Weight for concentration uncertainty

    Returns:
        Dictionary with custom weights (normalized to sum to 1)
    """
    total = margin + entropy + concentration
    return {
        'margin': margin / total,
        'entropy': entropy / total,
        'concentration': concentration / total
    }
