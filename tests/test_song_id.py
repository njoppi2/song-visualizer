from __future__ import annotations

from pathlib import Path

from songviz.ingest import song_id_for_path


def test_song_id_stable_for_same_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world\n")

    a = song_id_for_path(p)
    b = song_id_for_path(p)
    assert a == b

