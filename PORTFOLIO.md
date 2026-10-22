# Technical case study

This repository demonstrates a reproducible synthetic cost-modeling workflow and two interfaces sharing one fitted inference service. The central engineering issue was target leakage: billed components and insurance allocations reconstructed the outcome and invalidated apparent near-perfect scores.

The correction uses an explicit input allowlist, training-only preprocessing, disjoint patient splits, validation-only model selection, a baseline, deterministic inference and regression tests. See [feature audit](reports/FEATURE_AUDIT.md), [measured results](reports/evaluation.json) and [execution evidence](EXECUTION_REPORT.md).

What this work does not establish: clinical validity, prospective accuracy, fairness across real patient groups, production reliability, deployment, financial impact or treatment benefit. The synthetic generator encodes the relationships the model learns. More realistic temporal data, external validation, subgroup evaluation, calibrated uncertainty and security review would be necessary before considering real-world use.
