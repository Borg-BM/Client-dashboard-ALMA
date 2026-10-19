"""Verbindingskwaliteit uit aankomsttijden, voor als het telegram geen telbare hartslag heeft."""

import numpy as np

MIN_SAMPLES = 20  # minder pakketten zeggen niets over het ritme


def estimate_loss_percent(times) -> float | None:
    """Schat het pakketverlies uit de aankomsttijden (epoch-seconden, oplopend).

    Het mediane interval is de zendperiode; wat daar in de gaten bij ontbreekt telt als verloren, dus ook een
    uitval van Starlink. Dit werkt zonder teller in het telegram, maar blijft een schatting: None bij te weinig
    pakketten of een onbruikbaar ritme."""
    times = np.asarray(times, dtype=float)
    if len(times) < MIN_SAMPLES or not np.isfinite(times).all():
        return None
    period = float(np.median(np.diff(times)))
    if period <= 0:
        return None
    expected = (times[-1] - times[0]) / period + 1
    return max(0.0, 100 * (1 - len(times) / expected))
