"""decision_calib: reusable harness for calibration studies of zero-token decision models.

Modules:
    models     - model registry with a uniform predict interface.
    datasets   - dataset loaders returning standardized cases.
    metrics    - pure-numpy calibration metrics (ECE, Brier, NLL, reliability data).
    calibrate  - temperature scaling (fit global / per-type / per-task).
"""
