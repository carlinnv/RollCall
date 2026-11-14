"""Revenue math. Dollars justify the program; they never set which student gets help first."""
from collections import Counter
from datetime import date
from typing import Iterable, List

from ..models import Absence, Barrier, RevenueSettings, RevenueSummary, RootCauseCost

STATE_DEFAULTS = {
    # Midpoint of the $35-$100 per-day range cited in PRD 1 (vendor estimate, unverified).
    "CA": RevenueSettings(state="CA", daily_rate=65.0, recovery_enabled=True),
}


def dollars(days: int, settings: RevenueSettings) -> float:
    return round(days * settings.daily_rate, 2)


def in_week(a: Absence, start: date, end: date) -> bool:
    return start <= a.date <= end


def summarize(
    absences: Iterable[Absence],
    settings: RevenueSettings,
    week_start: date,
    week_end: date,
    days_recovered: int,
) -> RevenueSummary:
    week = [a for a in absences if in_week(a, week_start, week_end)]
    counts = Counter(a.classification.barrier if a.classification else Barrier.UNKNOWN for a in week)
    by_cause: List[RootCauseCost] = sorted(
        (RootCauseCost(barrier=b, absences=n, dollars=dollars(n, settings)) for b, n in counts.items()),
        # Known causes by dollars; "unknown" (no reason captured yet) always last.
        key=lambda r: (r.barrier == Barrier.UNKNOWN, -r.dollars),
    )
    recovered = days_recovered if settings.recovery_enabled else 0
    return RevenueSummary(
        settings=settings,
        week_start=week_start,
        week_end=week_end,
        absences_this_week=len(week),
        dollars_at_risk_week=dollars(len(week), settings),
        days_recovered_to_date=recovered,
        dollars_recovered_to_date=dollars(recovered, settings),
        by_root_cause=by_cause,
    )
