"""Link quality from arrival times, for when the telegram has no countable heartbeat."""

import numpy as np

MIN_SAMPLES = 20  # fewer packets say nothing about the rhythm


def estimate_loss_percent(times) -> float | None:
    """Estimate packet loss from the arrival times (epoch seconds, increasing).

    The median interval is the sending period; whatever is missing from the gaps counts as lost, so a Starlink
    outage too. This works without a counter in the telegram, but remains an estimate: None for too few
    packets or an unusable rhythm."""
    times = np.asarray(times, dtype=float)
    if len(times) < MIN_SAMPLES or not np.isfinite(times).all():
        return None
    period = float(np.median(np.diff(times)))
    if period <= 0:
        return None
    expected = (times[-1] - times[0]) / period + 1
    return max(0.0, 100 * (1 - len(times) / expected))
