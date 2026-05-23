"""Route smoke tests for the FastAPI review UI (``webui`` marker).

These exercise the thin route + template layer end-to-end against a temporary
``review.db`` via :class:`fastapi.testclient.TestClient`. They are deselected
from the default ``pdm run pytest`` (the web layer is excluded from coverage)
and run via ``pdm run webui``.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
from msgspec.json import encode as json_encode
from pd_matcher.models import MarcRecord
from pytest import fixture
from pytest import mark

from pd_groundtruth.review.app import create_app
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb

pytestmark = mark.webui


def _pair(*, language: str, control_id: str, nypl_uuid: str) -> PairInsert:
    marc = MarcRecord(
        control_id=control_id,
        title="A Studied Title",
        title_main="A Studied Title",
        main_author="Doe, Jane",
        publisher="Acme Press",
        publication_year=1953,
        language_code=language,
        country_code="nyu",
    )
    return PairInsert(
        language=language,
        decade=1950,
        score=0.93,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json=json_encode(marc).decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid=nypl_uuid,
        cce_title="A Studied Title",
        cce_author="Jane Doe",
        cce_publishers="Acme Press",
        cce_claimants="Jane Doe",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R12345",
        evidence_json='{"title.token_set": 1.0, "name.author": 0.8}',
    )


@fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(language="eng", control_id="eng-1", nypl_uuid="u-eng-1"))
        db.insert_pair(_pair(language="fre", control_id="fre-1", nypl_uuid="u-fre-1"))
    app = create_app(db_path)
    with TestClient(app) as test_client:
        yield test_client


def test_index_renders_a_card(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "A Studied Title" in response.text
    assert "title.token_set" in response.text
    assert "Renewed" in response.text


def test_index_language_filter_selects_french(client: TestClient) -> None:
    response = client.get("/", params={"language": "fre"})
    assert response.status_code == 200
    assert "fre-1" in response.text


def test_label_writes_row_and_redirects(client: TestClient) -> None:
    card = client.get("/")
    assert "pair #1" in card.text
    response = client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "language": "eng"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?language=eng"
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert ">1<" in stats.text


def test_label_advances_to_next_pair(client: TestClient) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "no_match"}, follow_redirects=False)
    nxt = client.get("/")
    assert nxt.status_code == 200
    assert "fre-1" in nxt.text


def test_pair_route_renders_specific_pair(client: TestClient) -> None:
    response = client.get("/pair/2")
    assert response.status_code == 200
    assert "fre-1" in response.text


def test_pair_route_404_for_missing(client: TestClient) -> None:
    response = client.get("/pair/9999")
    assert response.status_code == 404
    assert "not found" in response.text


def test_stats_route_renders_progress(client: TestClient) -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    assert "labeled" in response.text
    assert "By language" in response.text


def test_index_hides_back_link_before_any_label(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="back-link"' not in response.text


def test_back_link_targets_last_labeled_pair(client: TestClient) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    nxt = client.get("/")
    assert 'id="back-link"' in nxt.text
    assert "/pair/1" in nxt.text


def test_back_link_preserves_filter(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "2", "verdict": "match", "language": "fre"},
        follow_redirects=False,
    )
    nxt = client.get("/", params={"language": "fre"})
    assert "/pair/2?language=fre" in nxt.text


def test_card_links_marc_control_id_to_princeton_catalog(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="https://catalog.princeton.edu/catalog/eng-1"' in response.text


def test_card_renders_reason_chips(client: TestClient) -> None:
    response = client.get("/")
    assert "Different work / title collision" in response.text
    assert "Insufficient data on one side" in response.text
    assert 'name="note"' in response.text


def test_reason_chip_records_reason(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "reason": "diff_work"},
        follow_redirects=False,
    )
    stats = client.get("/stats")
    assert "Reasons (current" in stats.text
    assert "Different work / title collision" in stats.text


def test_multiple_reasons_are_recorded(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "reason": ["diff_work", "garbled"]},
        follow_redirects=False,
    )
    stats = client.get("/stats")
    assert "Different work / title collision" in stats.text
    assert "Garbled transcription" in stats.text


def test_cross_verdict_and_invalid_reasons_are_dropped(client: TestClient) -> None:
    client.post(
        "/label",
        data={
            "pair_id": "1",
            "verdict": "no_match",
            "reason": ["diff_work", "edition_unsure", "nonsense"],
        },
        follow_redirects=False,
    )
    stats = client.get("/stats")
    assert "Different work / title collision" in stats.text
    assert "Unsure about edition" not in stats.text


def test_invalid_reason_is_dropped(client: TestClient) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match", "reason": "diff_work"},
        follow_redirects=False,
    )
    stats = client.get("/stats")
    assert "Reasons (current" not in stats.text


def test_empty_queue_page_when_all_labeled(client: TestClient) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.post("/label", data={"pair_id": "2", "verdict": "match"}, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "All done" in response.text
