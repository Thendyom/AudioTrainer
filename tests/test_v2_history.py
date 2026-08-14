from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sqlite3

from audiotrainer.history import SessionRepository


def test_history_migrates_persists_exports_and_deletes_audio(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFF-test")
    repository = SessionRepository(tmp_path / "data")
    session = repository.save(
        mode="pitch",
        duration=1.2,
        source="upload",
        backend="yin",
        settings={"target": "A4"},
        result={"accuracy": 0.9},
        feedback=[],
        audio_source=source,
        retain_audio=True,
    )
    assert repository.get(session.session_id) == session
    assert repository.list(mode="pitch")[0].result["accuracy"] == 0.9
    assert session.audio_path is not None and Path(session.audio_path).exists()
    assert "session_id" in repository.export_json()
    assert repository.export_csv(tmp_path / "history.csv").exists()
    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        connection.close()
    assert repository.delete(session.session_id)
    assert not Path(session.audio_path).exists()


def test_history_settings_and_clear(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "data")
    repository.set_setting("retain_audio_default", True)
    assert repository.get_setting("retain_audio_default") is True
    for mode in ["pitch", "speech"]:
        repository.save(
            mode=mode, duration=1.0, source="upload", backend="baseline", settings={}, result={}, feedback=[]
        )
    assert repository.clear() == 2
    assert repository.list() == []


def test_history_supports_concurrent_reads(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "data")
    for index in range(3):
        repository.save(
            mode="pitch",
            duration=float(index + 1),
            source="generated",
            backend="yin",
            settings={},
            result={"accuracy": index / 3},
            feedback=[],
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        lengths = list(pool.map(lambda _: len(repository.list()), range(12)))
    assert lengths == [3] * 12
