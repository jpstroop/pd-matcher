"""Unit tests for the MARCXML shard writer."""

from pathlib import Path

from lxml.etree import _Element
from lxml.etree import fromstring
from lxml.etree import parse
from pytest import raises

from pd_groundtruth.writer import MarcxmlShardWriter

_MARC_NS = "http://www.loc.gov/MARC21/slim"


def _record(index: int) -> _Element:
    """Build a small namespaced record carrying its index in 245$a."""
    xml = (
        f'<record xmlns="{_MARC_NS}">'
        f'<datafield tag="245"><subfield code="a">Title {index}</subfield></datafield>'
        f"</record>"
    )
    return fromstring(xml.encode("utf-8"))


def _count_records(path: Path) -> int:
    tree = parse(str(path))
    return len(tree.getroot())


def test_shards_roll_at_cap(tmp_path: Path) -> None:
    with MarcxmlShardWriter(tmp_path, shard_size=5) as writer:
        for index in range(12):
            writer.write(_record(index))
        assert writer.total_written == 12

    shards = sorted(tmp_path.glob("candidates_*.xml"))
    assert [p.name for p in shards] == [
        "candidates_00001.xml",
        "candidates_00002.xml",
        "candidates_00003.xml",
    ]
    assert [_count_records(p) for p in shards] == [5, 5, 2]


def test_shards_written_count(tmp_path: Path) -> None:
    writer = MarcxmlShardWriter(tmp_path, shard_size=5)
    for index in range(12):
        writer.write(_record(index))
    writer.close()
    assert writer.shards_written == 3


def test_records_round_trip_losslessly(tmp_path: Path) -> None:
    with MarcxmlShardWriter(tmp_path, shard_size=10) as writer:
        writer.write(_record(0))

    tree = parse(str(tmp_path / "candidates_00001.xml"))
    subfield = tree.getroot()[0][0][0]
    assert subfield.text == "Title 0"
    assert tree.getroot().tag == f"{{{_MARC_NS}}}collection"


def test_close_is_idempotent(tmp_path: Path) -> None:
    writer = MarcxmlShardWriter(tmp_path, shard_size=5)
    writer.write(_record(0))
    writer.close()
    writer.close()
    assert writer.shards_written == 1


def test_no_records_writes_no_shard(tmp_path: Path) -> None:
    with MarcxmlShardWriter(tmp_path):
        pass
    assert list(tmp_path.glob("candidates_*.xml")) == []


def test_invalid_shard_size_rejected(tmp_path: Path) -> None:
    with raises(ValueError, match="positive integer"):
        MarcxmlShardWriter(tmp_path, shard_size=0)
