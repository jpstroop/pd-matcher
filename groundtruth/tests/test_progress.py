"""Unit tests for the pure progress math and the cadence-gated reporter.

No pool, no real matching: rate/ETA/percent/formatting are exercised on
fabricated counters, and the reporter is driven with a controllable clock
and a fabricated outcome stream.
"""

from logging import getLogger

from pd_groundtruth.progress import ProgressReporter
from pd_groundtruth.progress import ProgressSnapshot
from pd_groundtruth.progress import format_progress
from pd_groundtruth.sampling import BudgetModel

_LOGGER = getLogger("test.progress")


def _budget() -> BudgetModel:
    return BudgetModel(
        caps={
            ("eng", "ge90"): 500,
            ("eng", "b80_90"): 200,
            ("eng", "b70_80"): 200,
            ("eng", "below"): 300,
            ("fre", "ge90"): 60,
            ("fre", "below"): 80,
        }
    )


def _snapshot(
    done: int,
    total: int,
    elapsed: float,
    kept: dict[tuple[str, str], int] | None = None,
) -> ProgressSnapshot:
    return ProgressSnapshot(
        done=done,
        total=total,
        elapsed_seconds=elapsed,
        kept_by_stratum={} if kept is None else kept,
    )


def test_records_per_second_and_minute() -> None:
    snap = _snapshot(6000, 14350, 740.0)
    assert snap.records_per_second == 6000 / 740.0
    assert snap.records_per_minute == snap.records_per_second * 60


def test_rate_is_zero_when_no_elapsed_time() -> None:
    snap = _snapshot(10, 100, 0.0)
    assert snap.records_per_second == 0.0
    assert snap.records_per_minute == 0.0


def test_rate_is_zero_when_negative_elapsed() -> None:
    snap = _snapshot(10, 100, -1.0)
    assert snap.records_per_second == 0.0


def test_percent_floors() -> None:
    assert _snapshot(6000, 14350, 1.0).percent == 41
    assert _snapshot(0, 100, 1.0).percent == 0
    assert _snapshot(100, 100, 1.0).percent == 100


def test_percent_zero_total() -> None:
    assert _snapshot(0, 0, 1.0).percent == 0


def test_eta_seconds_computed() -> None:
    snap = _snapshot(100, 300, 10.0)
    assert snap.records_per_second == 10.0
    assert snap.eta_seconds == 20.0


def test_eta_none_when_rate_zero() -> None:
    assert _snapshot(0, 100, 0.0).eta_seconds is None


def test_eta_none_when_complete() -> None:
    assert _snapshot(100, 100, 5.0).eta_seconds is None
    assert _snapshot(120, 100, 5.0).eta_seconds is None


def test_render_matches_expected_shape() -> None:
    kept = {
        ("eng", "ge90"): 412,
        ("eng", "b80_90"): 90,
        ("eng", "b70_80"): 88,
        ("eng", "below"): 300,
        ("fre", "ge90"): 100,
        ("fre", "below"): 40,
    }
    snap = _snapshot(6000, 14350, 740.0, kept)
    line = snap.render(_budget())
    assert line.startswith("progress  6,000/14,350 (41%)  agg ")
    assert "rec/s (" in line
    assert "/min)  elapsed 12:20  ETA " in line
    assert "eng[ge90 412/500 b80_90 90/200 b70_80 88/200 below 300/300]" in line
    assert "fre 140" in line


def test_render_eta_dashes_when_rate_zero() -> None:
    line = _snapshot(0, 100, 0.0, {}).render(_budget())
    assert "ETA --" in line


def test_render_empty_budget_has_empty_kept() -> None:
    line = _snapshot(1, 2, 1.0, {}).render(BudgetModel(caps={}))
    assert line.endswith("kept: ")


def test_format_progress_round_trips() -> None:
    direct = format_progress(100, 300, 10.0, {}, _budget())
    via_struct = _snapshot(100, 300, 10.0, {}).render(_budget())
    assert direct == via_struct


def test_mmss_does_not_wrap_minutes() -> None:
    line = _snapshot(1, 100, 3725.0, {}).render(BudgetModel(caps={}))
    assert "elapsed 62:05" in line


class _FakeClock:
    """A monotonic clock whose value is advanced explicitly in tests."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_reporter_emits_every_n_records() -> None:
    clock = _FakeClock()
    logger = getLogger("test.progress.n")
    reporter = ProgressReporter(
        logger=logger,
        total=10,
        budget=BudgetModel(caps={}),
        clock=clock,
        every_n=3,
        every_seconds=1000.0,
    )
    results = [reporter.update(done, {}) for done in range(1, 10)]
    assert results == [False, False, True, False, False, True, False, False, True]


def test_reporter_emits_on_time_when_count_low() -> None:
    clock = _FakeClock()
    reporter = ProgressReporter(
        logger=_LOGGER,
        total=1000,
        budget=BudgetModel(caps={}),
        clock=clock,
        every_n=500,
        every_seconds=5.0,
    )
    assert reporter.update(1, {}) is False
    clock.advance(5.0)
    assert reporter.update(2, {}) is True


def test_reporter_resets_cadence_after_emit() -> None:
    clock = _FakeClock()
    reporter = ProgressReporter(
        logger=_LOGGER,
        total=100,
        budget=BudgetModel(caps={}),
        clock=clock,
        every_n=2,
        every_seconds=1000.0,
    )
    assert reporter.update(2, {}) is True
    assert reporter.update(3, {}) is False
    assert reporter.update(4, {}) is True


def test_reporter_over_fabricated_stream_logs_expected_count(caplog: object) -> None:
    from logging import INFO

    from _pytest.logging import LogCaptureFixture

    assert isinstance(caplog, LogCaptureFixture)
    clock = _FakeClock()
    logger = getLogger("test.progress.stream")
    reporter = ProgressReporter(
        logger=logger,
        total=10,
        budget=BudgetModel(caps={("eng", "ge90"): 5}),
        clock=clock,
        every_n=4,
        every_seconds=1000.0,
    )
    kept: dict[tuple[str, str], int] = {}
    with caplog.at_level(INFO, logger="test.progress.stream"):
        for done in range(1, 11):
            kept[("eng", "ge90")] = min(5, done)
            reporter.update(done, kept)
    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2
    assert all(message.startswith("progress  ") for message in lines)
