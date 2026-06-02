# tests/unit/test_diag_single_batch.py
from src.diag.single_batch import BatchReplay


def test_replay_caches_first_and_repeats():
    replay = BatchReplay()
    calls = iter([("batch-A",), ("batch-B",), ("batch-C",)])

    def fake_get_batch():
        return next(calls)

    first = replay(fake_get_batch)
    second = replay(fake_get_batch)
    third = replay(fake_get_batch)

    assert first == ("batch-A",)
    # subsequent calls return the cached first batch; the producer is NOT advanced
    assert second == ("batch-A",)
    assert third == ("batch-A",)
    assert replay.calls == 3
    assert replay.producer_calls == 1
