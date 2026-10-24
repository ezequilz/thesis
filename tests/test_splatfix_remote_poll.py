from splat_explorer.splatfix import remote_poll


def test_idle_and_stopping_workers_wait_ten_seconds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(remote_poll.time, 'sleep', sleeps.append)
    for stopping in (False, True):
        sleeps.clear()
        remote_poll.wait_for_remote_poll(lambda: stopping)
        assert sum(sleeps) == 10


def test_new_cancellation_wakes_without_waiting_for_next_remote_poll(monkeypatch):
    sleeps = []
    monkeypatch.setattr(remote_poll.time, 'sleep', sleeps.append)
    remote_poll.wait_for_remote_poll(lambda: len(sleeps) >= 2)
    assert sum(sleeps) == 2
