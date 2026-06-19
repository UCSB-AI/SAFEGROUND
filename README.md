# SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration

This repository contains the **official implementation of SafeGround**, an uncertainty-aware framework for reliable and risk-controlled GUI grounding under limited model access.

SafeGround estimates *spatial uncertainty* by aggregating multiple stochastic grounding predictions into a patch-level probability distribution, and calibrates uncertainty thresholds with finite-sample guarantees. This enables risk-aware deployment of GUI agents through selective prediction and safe deferral.

<p align="center">
  <img src="images/safeground.png" alt="Framework Overview" width="85%">
</p>

**Paper:**
📄 *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration*

🔗 [https://arxiv.org/abs/2602.02419](https://arxiv.org/abs/2602.02419)

**Project Page:**
🔗 [https://safeground-ericlab.github.io](https://safeground-ericlab.github.io)

---

## Repository Structure

```
SAFEGROUND/
├── heatmap.py          # Heatmap construction from sampled coordinates
├── regions.py          # Connected region extraction (4-connectivity BFS)
├── margin.py           # Margin-based uncertainty (top-2 ambiguity)
├── entropy.py          # Entropy-based uncertainty (distributional dispersion)
├── concentration.py    # Concentration-based uncertainty (HHI complement)
├── combined.py         # Weighted combination of uncertainty measures
├── uncertainty.py      # Unified uncertainty computation API
├── selective_prediction.py # Accepted error control (Clopper–Pearson)
└── README.md           # Project documentation
```

---

## Uncertainty Quantification Pipeline

**Pipeline:**
**Stochastic Coordinates → Patch Heatmap → Spatial Regions → Uncertainty Score**

Given multiple stochastic grounding samples, SafeGround constructs a spatial probability distribution over a patch grid, identifies coherent high-probability regions, and computes region-level uncertainty measures that capture different failure modes of GUI grounding.


### Implemented Uncertainty Measures

| Method          | Description                     |
| --------------- | ------------------------------- |
| `margin`        | Ambiguity between top-2 regions |
| `entropy`       | Distributional dispersion       |
| `concentration` | Lack of spatial concentration   |
| `combined`      | Composite uncertainty           |

---

## Selective Prediction with Accepted Error Control

SafeGround calibrates an uncertainty threshold on a held-out calibration set to control the **error rate among accepted predictions**. A prediction is accepted when its uncertainty is no greater than the calibrated threshold and is otherwise abstained from or deferred.

The region activation threshold is fixed to **0** by default. Consequently, every heatmap patch with positive probability is included during connected-region extraction.

### Clopper–Pearson Upper Confidence Bound

Given `w` observed errors among `m` accepted calibration samples, the one-sided upper confidence bound with calibration failure probability `δ` is:

```
r_upper = Beta.ppf(1 - δ, w + 1, m - w)
```

When all accepted calibration samples are errors (`w = m`), the upper bound is defined as `1`.

Following Algorithm 1, candidate thresholds are tested in ascending uncertainty order. SafeGround retains each certified threshold while its upper bound does not exceed the target accepted error rate, and stops at the first uncertified threshold. If the first candidate cannot be certified, no prediction is accepted.

### Reported Metrics

| Metric            | Description                                |
| ----------------- | ------------------------------------------ |
| `threshold`       | Calibrated uncertainty threshold           |
| `power`           | Fraction of correct predictions retained   |
| `coverage`        | Fraction of all predictions accepted       |
| `abstention_rate` | Fraction of predictions rejected           |
| `accepted_error_rate` | Error fraction among accepted predictions |
| `calibration_upper_bound` | Clopper–Pearson accepted-error upper bound |

---
More code coming soon.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026safegroundknowtrustgui,
  title={SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration},
  author={Qingni Wang and Yue Fan and Xin Eric Wang},
  year={2026},
  eprint={2602.02419},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.02419}
}
```
