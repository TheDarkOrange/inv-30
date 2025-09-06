import numpy as np


def calibrate_conformal(preds_val: np.ndarray, y_val: np.ndarray, alpha: float = 0.1):
    # symmetric absolute residual quantile
    resid = np.abs(preds_val - y_val)
    qhat = float(np.quantile(resid, 1 - alpha))
    return {"alpha": alpha, "qhat": qhat}
