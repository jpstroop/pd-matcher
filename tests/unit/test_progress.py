"""Unit tests for the generic progress math and cadence-gated reporter.

No workload: rate/ETA/percent/formatting are exercised on fabricated
counters, and the reporter is driven with a controllable clock and a
fabricated stream.
"""

from logging import INFO
from logging import getLogger

from _pytest.logging import LogCaptureFixture

from pd_matcher.progress import ProgressReporter
from pd_matcher.progress import ProgressSnapshot

_LOGGER = getLogger("test.progress")


def _snapshot(done: int, total: int, elapsed: float) -> ProgressSnapshot:
    return ProgressSnapshot(done=done, total=total, elapsed_seconds=elapsed)


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
    line = _snapshot(6000, 14350, 740.0).render()
    assert line.startswith("progress  6,000/14,350 (41%)  agg ")
    assert "rec/s (" in line
    assert "/min)  elapsed 12:20  ETA " in line


def test_render_eta_dashes_when_rate_zero() -> None:
    assert "ETA --" in _snapshot(0, 100, 0.0).render()


def test_render_has_no_trailing_detail() -> None:
    assert _snapshot(2, 2, 1.0).render().endswith("ETA --")


def test_mmss_does_not_wrap_minutes() -> None:
    assert "elapsed 62:05" in _snapshot(1, 100, 3725.0).render()


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
    reporter = ProgressReporter(
        logger=getLogger("test.progress.n"),
        total=10,
        clock=clock,
        every_n=3,
        every_seconds=1000.0,
    )
    results = [reporter.update(done) for done in range(1, 10)]
    assert results == [False, False, True, False, False, True, False, False, True]


def test_reporter_emits_on_time_when_count_low() -> None:
    clock = _FakeClock()
    reporter = ProgressReporter(
        logger=_LOGGER,
        total=1000,
        clock=clock,
        every_n=500,
        every_seconds=5.0,
    )
    assert reporter.update(1) is False
    clock.advance(5.0)
    assert reporter.update(2) is True


def test_reporter_resets_cadence_after_emit() -> None:
    clock = _FakeClock()
    reporter = ProgressReporter(
        logger=_LOGGER,
        total=100,
        clock=clock,
        every_n=2,
        every_seconds=1000.0,
    )
    assert reporter.update(2) is True
    assert reporter.update(3) is False
    assert reporter.update(4) is True


def test_reporter_appends_detail_when_present(caplog: LogCaptureFixture) -> None:
    clock = _FakeClock()
    logger = getLogger("test.progress.detail")
    reporter = ProgressReporter(
        logger=logger,
        total=10,
        clock=clock,
        every_n=1,
        every_seconds=1000.0,
    )
    with caplog.at_level(INFO, logger="test.progress.detail"):
        assert reporter.update(1, detail="kept: eng 5") is True
    message = caplog.records[0].getMessage()
    assert message.startswith("progress  ")
    assert message.endswith(" kept: eng 5")


def test_reporter_omits_detail_when_empty(caplog: LogCaptureFixture) -> None:
    clock = _FakeClock()
    logger = getLogger("test.progress.nodetail")
    reporter = ProgressReporter(
        logger=logger,
        total=10,
        clock=clock,
        every_n=1,
        every_seconds=1000.0,
    )
    with caplog.at_level(INFO, logger="test.progress.nodetail"):
        assert reporter.update(1) is True
    message = caplog.records[0].getMessage()
    assert message.startswith("progress  ")
    assert "ETA" in message
    assert "kept" not in message
