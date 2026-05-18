"""Tests for :class:`pd_matcher.workers.shutdown.ShutdownCoordinator`."""

from multiprocessing import Event
from os import getpid
from os import kill
from signal import SIGINT
from signal import getsignal
from time import sleep

from pd_matcher.workers.shutdown import ShutdownCoordinator


def test_default_event_is_unset() -> None:
    coord = ShutdownCoordinator()
    assert coord.event.is_set() is False
    assert coord.is_set is False


def test_explicit_event_is_reused() -> None:
    event = Event()
    coord = ShutdownCoordinator(event)
    assert coord.event is event


def test_request_shutdown_sets_event() -> None:
    coord = ShutdownCoordinator()
    coord.request_shutdown()
    assert coord.is_set is True


def test_context_manager_installs_and_restores_handler() -> None:
    previous = getsignal(SIGINT)
    with ShutdownCoordinator() as coord:
        assert getsignal(SIGINT) is not previous
        assert coord.is_set is False
    assert getsignal(SIGINT) is previous


def test_sigint_flips_event_without_raising() -> None:
    """A single SIGINT must flip the event without raising KeyboardInterrupt."""
    with ShutdownCoordinator() as coord:
        kill(getpid(), SIGINT)
        # Signal handlers run on the main thread; give the OS a beat.
        for _ in range(20):
            if coord.is_set:
                break
            sleep(0.01)
        assert coord.is_set is True
