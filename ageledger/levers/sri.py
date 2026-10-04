"""Sleep Regularity Index (Phillips 2017; as used in Windred 2024, Sleep, DOI 10.1093/sleep/zsad253).

SRI = 200 x P(same sleep/wake state at t and t+24h) - 100, on a 1-minute grid.
+100 = identical timing every day; 0 = random; negative = systematically shifted.

Grid: each night (daily row, `date` = the day the night ENDED on) owns the 24-h block
[date-1 12:00, date 12:00). A block is "covered" when that row has both a valid onset and offset.
Every valid sleep interval is painted onto the grid (it may spill past noon into the next block —
that is real sleep in that block). Minute t is scored only when t's block AND the block 24h later are
both covered; gaps are skipped, never imputed.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

MIN_PER_DAY = 1440


def _naive_local(value) -> Optional[pd.Timestamp]:
    """Parse an ISO string to a naive local-wall-clock Timestamp; None if unparseable/missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)       # keep local wall time, drop offset
    return ts


def sleep_regularity_index(daily: pd.DataFrame) -> tuple[Optional[float], int]:
    """Return (SRI, n_nights). n_nights = rows with a valid onset/offset pair.

    SRI is None when there is no pair of consecutive covered nights to compare.
    The caller decides the window and the min-nights threshold.
    """
    if daily is None or daily.empty or not {"date", "sleep_onset", "sleep_offset"} <= set(daily.columns):
        return None, 0

    nights: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []   # (day, onset, offset)
    for _, row in daily.iterrows():
        on, off = _naive_local(row["sleep_onset"]), _naive_local(row["sleep_offset"])
        day = _naive_local(row["date"])
        if on is None or off is None or day is None:
            continue
        if not (pd.Timedelta(0) < off - on < pd.Timedelta(hours=24)):
            continue
        nights.append((day.normalize(), on, off))
    n = len(nights)
    if n == 0:
        return None, 0

    days = sorted({d for d, _, _ in nights})
    origin = days[0] - pd.Timedelta(hours=12)            # start of first block
    n_blocks = (days[-1] - days[0]).days + 2             # +1 spare block for spill-over
    asleep = np.zeros(n_blocks * MIN_PER_DAY, dtype=bool)
    covered = np.zeros(n_blocks, dtype=bool)

    for day, on, off in nights:
        covered[(day - days[0]).days] = True
        a = int((on - origin) / pd.Timedelta(minutes=1))
        b = int((off - origin) / pd.Timedelta(minutes=1))
        a, b = max(a, 0), min(b, asleep.size)
        if b > a:
            asleep[a:b] = True

    pair_blocks = np.flatnonzero(covered[:-1] & covered[1:])
    if pair_blocks.size == 0:
        return None, n

    same = 0
    total = 0
    for k in pair_blocks:
        s = k * MIN_PER_DAY
        x = asleep[s:s + MIN_PER_DAY]
        y = asleep[s + MIN_PER_DAY:s + 2 * MIN_PER_DAY]
        same += int(np.count_nonzero(x == y))
        total += MIN_PER_DAY
    return 200.0 * same / total - 100.0, n
