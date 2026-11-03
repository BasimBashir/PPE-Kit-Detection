import json
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.alerts.spool import EventSpool
from app.config.settings import EvidenceConfig
from app.evidence.store import EvidenceStore


def store(tmp_path, free=10**12):
    return EvidenceStore(tmp_path / "ev", EvidenceConfig(min_free_bytes=1000),
                         disk_usage=lambda _p: SimpleNamespace(free=free))


def test_save_open_and_traversal_rejected(tmp_path):
    st = store(tmp_path)
    eid = uuid.uuid4()
    rel, err = st.save(eid, np.zeros((10, 10, 3), np.uint8), time.time())
    assert err is None and rel.endswith(f"{eid}.jpg") and st.open(rel).is_file()
    (tmp_path / "secret.txt").write_text("x")
    for bad in ("../secret.txt", "..\\secret.txt", str(tmp_path / "secret.txt")):
        with pytest.raises(FileNotFoundError):
            st.open(bad)


def test_disk_low_and_write_failure_never_raise(tmp_path, monkeypatch):
    assert store(tmp_path, free=10).save(uuid.uuid4(), np.zeros((4, 4, 3), np.uint8), time.time()) == (None, "disk_low")
    st = store(tmp_path)

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert st.save(uuid.uuid4(), np.zeros((4, 4, 3), np.uint8), time.time()) == (None, "write_failed")


def test_retention_deletes_old_files(tmp_path):
    st = store(tmp_path)
    rel_old, _ = st.save(uuid.uuid4(), np.zeros((4, 4, 3), np.uint8), time.time())
    rel_new, _ = st.save(uuid.uuid4(), np.zeros((4, 4, 3), np.uint8), time.time())
    old = st.open(rel_old)
    t = time.time() - 40 * 86400
    os.utime(old, (t, t))
    files, freed = st.delete_older_than(30)
    assert files == 1 and freed > 0 and not old.exists() and st.open(rel_new).exists()


def test_spool_replay_keeps_failures_and_rewrites_atomically(tmp_path):
    sp = EventSpool(tmp_path)
    for i in range(3):
        sp.append({"event": {"id": i}})
    assert [p["event"]["id"] for p in sp.pending()] == [0, 1, 2]
    seen = []

    def insert(p):
        if p["event"]["id"] == 1:
            raise ConnectionError("db down")
        seen.append(p["event"]["id"])

    assert sp.replay(insert) == 1 and seen == [0]
    assert [p["event"]["id"] for p in sp.pending()] == [1, 2]
    assert sp.replay(lambda p: None) == 2 and sp.pending() == []
    # torn last line (power loss mid-write) is skipped, not fatal
    sp.path.write_text(json.dumps({"event": {"id": 9}}) + "\n{\"event\": {\"id\"")
    assert [p["event"]["id"] for p in sp.pending()] == [9]
