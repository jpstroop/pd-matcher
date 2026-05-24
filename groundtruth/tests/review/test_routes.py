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

from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.review.app import create_app
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb

pytestmark = mark.webui


def _pair(
    *,
    language: str,
    control_id: str,
    nypl_uuid: str,
    extent: str | None = None,
    predicted_status: str | None = "PD_REGISTERED_NOT_RENEWED",
    renewal_id: str | None = "R200001",
    renewal_oreg: str | None = "A111111",
    renewal_rdat: str | None = "1968-05-15",
    renewal_author: str | None = "Smith, John",
    renewal_title: str | None = "A study of widgets",
    renewal_claimants: str | None = "Estate of Jane Doe|PWH",
    renewal_new_matter: str | None = "added chapter 7",
) -> PairInsert:
    marc = MarcRecord(
        control_id=control_id,
        title="A Studied Title",
        title_main="A Studied Title",
        lccn="40012345",
        oclc="0001",
        isbns=("9780000000000",),
        title_part_number="Pt. 2",
        title_part_name="The empire of Sebastopol",
        main_author="Doe, Jane",
        edition="2nd ed.",
        publication_place="New York",
        publisher="Acme Press",
        publication_date_raw="c1953.",
        publication_year=1953,
        extent=extent if extent is not None else "xxiv, 841 p.",
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
        cce_edition="2nd ed.",
        cce_publication_places="New York; London",
        cce_author_place="Cambridge, Mass.",
        cce_author_is_claimant=True,
        cce_copies="2c.",
        cce_aff_date="1953-06-01",
        cce_desc="vi, 200 p.",
        cce_notes="first note\nsecond note",
        cce_new_matter_claimed="added chapter 5",
        cce_copy_date="1953-04-01",
        cce_notice_date="1953-04-02",
        cce_lccn="28000854",
        cce_prev_regnums="A100000; A200000",
        cce_predicted_status=predicted_status,
        cce_renewal_id=renewal_id,
        cce_renewal_oreg=renewal_oreg,
        cce_renewal_rdat=renewal_rdat,
        cce_renewal_author=renewal_author,
        cce_renewal_title=renewal_title,
        cce_renewal_claimants=renewal_claimants,
        cce_renewal_new_matter=renewal_new_matter,
    )


@fixture
def vault_path(tmp_path: Path) -> Path:
    return tmp_path / "vault.jsonl"


@fixture
def client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(language="eng", control_id="eng-1", nypl_uuid="u-eng-1"))
        db.insert_pair(_pair(language="fre", control_id="fre-1", nypl_uuid="u-fre-1"))
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def skip_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(_pair(language="eng", control_id="s-1", nypl_uuid="u-s-1"))
        db.insert_pair(_pair(language="eng", control_id="s-2", nypl_uuid="u-s-2"))
        db.insert_pair(_pair(language="eng", control_id="s-3", nypl_uuid="u-s-3"))
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def ebook_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="eng-ebook-1",
                nypl_uuid="u-eng-ebook-1",
                extent="1 online resource (xxi, 406 p.)",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def in_copyright_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="ic-1",
                nypl_uuid="u-ic-1",
                predicted_status="IN_COPYRIGHT_REGISTERED_AND_RENEWED",
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def no_predicted_status_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        db.insert_pair(
            _pair(
                language="eng",
                control_id="np-1",
                nypl_uuid="u-np-1",
                predicted_status=None,
                renewal_id=None,
                renewal_oreg=None,
                renewal_rdat=None,
                renewal_author=None,
                renewal_title=None,
                renewal_claimants=None,
                renewal_new_matter=None,
            )
        )
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def empty_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def labels_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        eng_match = db.insert_pair(_pair(language="eng", control_id="eng-m", nypl_uuid="u-eng-m"))
        eng_no = db.insert_pair(_pair(language="eng", control_id="eng-n", nypl_uuid="u-eng-n"))
        fre_match = db.insert_pair(_pair(language="fre", control_id="fre-m", nypl_uuid="u-fre-m"))
        db.add_label(eng_match, "match")
        db.add_label(eng_no, "no_match", reasons=("diff_work",))
        db.add_label(fre_match, "match")
    app = create_app(db_path, vault_path)
    with TestClient(app) as test_client:
        yield test_client


@fixture
def pagination_client(tmp_path: Path, vault_path: Path) -> Iterator[TestClient]:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path) as db:
        for i in range(110):
            pair_id = db.insert_pair(_pair(language="eng", control_id=f"c-{i}", nypl_uuid=f"u-{i}"))
            db.add_label(pair_id, "match")
    app = create_app(db_path, vault_path)
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


def test_ebook_badge_renders_for_online_resource(ebook_client: TestClient) -> None:
    response = ebook_client.get("/")
    assert response.status_code == 200
    assert '<span class="badge-ebook">E-book reprint</span>' in response.text


def test_ebook_badge_absent_for_physical_pairs(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "E-book reprint" not in response.text
    assert '<span class="badge-ebook">' not in response.text


def test_card_renders_new_reason_chips(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Generic title" in response.text
    assert "Looks right but publisher differs" in response.text
    assert "Possibly a translation vs. original" in response.text
    assert "Reprint / different physical format" in response.text
    assert "Possibly whole vs. part / volume" in response.text
    assert "Looks like one issue of a periodical" in response.text


def test_label_appends_to_vault(client: TestClient, vault_path: Path) -> None:
    assert not vault_path.exists()
    response = client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert vault_path.exists()
    entries = list(iter_entries(vault_path))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.marc_control_id == "eng-1"
    assert entry.nypl_uuid == "u-eng-1"
    assert entry.verdict == "match"
    assert entry.reasons == ()
    assert entry.labeler == "jpstroop"
    assert entry.schema == 1
    assert entry.marc_identifiers.lccn == "40012345"
    assert entry.marc_identifiers.oclc == "0001"
    assert entry.marc_identifiers.isbns == ("9780000000000",)


def test_label_appends_one_line_per_post(client: TestClient, vault_path: Path) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    after_first = vault_path.read_text(encoding="utf-8")
    client.post("/label", data={"pair_id": "2", "verdict": "no_match"}, follow_redirects=False)
    after_second = vault_path.read_text(encoding="utf-8")
    assert after_second.startswith(after_first)
    assert len(after_second.splitlines()) == 2


def test_label_relabel_preserves_history(client: TestClient, vault_path: Path) -> None:
    client.post("/label", data={"pair_id": "1", "verdict": "match"}, follow_redirects=False)
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "reason": "diff_work"},
        follow_redirects=False,
    )
    history = list(iter_entries(vault_path))
    assert [event.verdict for event in history] == ["match", "no_match"]
    latest = current_entries(vault_path)
    assert latest[("eng-1", "u-eng-1")].verdict == "no_match"
    assert latest[("eng-1", "u-eng-1")].reasons == ("diff_work",)


def test_label_db_timestamp_matches_vault_timestamp(
    client: TestClient, vault_path: Path, tmp_path: Path
) -> None:
    client.post(
        "/label",
        data={"pair_id": "1", "verdict": "no_match", "reason": "diff_work", "note": "hmm"},
        follow_redirects=False,
    )
    [vault_entry] = list(iter_entries(vault_path))
    with ReviewDb.connect(tmp_path / "review.db") as db:
        [label] = list(db.iter_current_labels())
    assert label.labeled_at == vault_entry.labeled_at
    assert label.note == vault_entry.note == "hmm"


def test_skip_query_excludes_single_pair(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"skip": 1})
    assert response.status_code == 200
    assert "s-2" in response.text
    assert "s-1" not in response.text


def test_skip_query_excludes_multiple_pairs(skip_client: TestClient) -> None:
    response = skip_client.get("/", params=[("skip", 1), ("skip", 2)])
    assert response.status_code == 200
    assert "s-3" in response.text
    assert "s-1" not in response.text
    assert "s-2" not in response.text


def test_skip_query_empty_returns_all_done(skip_client: TestClient) -> None:
    response = skip_client.get("/", params=[("skip", 1), ("skip", 2), ("skip", 3)])
    assert response.status_code == 200
    assert "All done" in response.text


def test_label_redirect_does_not_carry_skip_state(skip_client: TestClient) -> None:
    response = skip_client.post(
        "/label",
        data={"pair_id": "1", "verdict": "match"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    nxt = skip_client.get("/")
    assert "s-2" in nxt.text


def test_card_renders_skip_url_with_current_pair_id_appended(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"skip": 1})
    assert response.status_code == 200
    assert 'skipUrl = "/?skip=1&skip=2"' in response.text


def test_card_skip_url_includes_filters(skip_client: TestClient) -> None:
    response = skip_client.get("/", params={"language": "eng"})
    assert response.status_code == 200
    assert 'skipUrl = "/?language=eng&skip=1"' in response.text


def test_labels_route_with_no_labels_shows_empty_message(empty_client: TestClient) -> None:
    response = empty_client.get("/labels")
    assert response.status_code == 200
    assert "No labels yet" in response.text
    assert "<table" not in response.text


def test_labels_route_renders_table_for_each_label(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/3"' in response.text
    assert "verdict-match" in response.text
    assert "verdict-no_match" in response.text
    assert "Different work / title collision" in response.text


def test_labels_route_filter_by_verdict_narrows_rows(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "no_match"})
    assert response.status_code == 200
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/3"' not in response.text


def test_labels_route_filter_by_language_narrows_rows(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"language": "fre"})
    assert response.status_code == 200
    assert 'href="/pair/3"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/2"' not in response.text


def test_labels_route_filter_by_reason_excludes_labels_without_reason(
    labels_client: TestClient,
) -> None:
    response = labels_client.get("/labels", params={"reason": "diff_work"})
    assert response.status_code == 200
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/3"' not in response.text


def test_labels_route_substring_search_matches_marc_title(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"q": "studied"})
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text
    response = labels_client.get("/labels", params={"q": "no-such-thing"})
    assert response.status_code == 200
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/2"' not in response.text


def test_labels_route_combined_filters_and_together(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "no_match", "language": "eng"})
    assert response.status_code == 200
    assert 'href="/pair/2"' in response.text
    assert 'href="/pair/1"' not in response.text
    assert 'href="/pair/3"' not in response.text


def test_labels_route_pair_id_links_to_pair_detail(labels_client: TestClient) -> None:
    response = labels_client.get("/labels")
    assert response.status_code == 200
    assert 'href="/pair/1"' in response.text


def test_labels_route_pagination_caps_rows_and_disables_next_on_last_page(
    pagination_client: TestClient,
) -> None:
    page_two = pagination_client.get("/labels", params={"page": 2})
    assert page_two.status_code == 200
    body_rows = page_two.text.split("<tbody>")[1].split("</tbody>")[0]
    assert body_rows.count("<tr>") == 10
    assert "page 2 of 2" in page_two.text
    assert '<span class="disabled">Next' in page_two.text


def test_labels_route_pagination_disables_prev_on_first_page(
    pagination_client: TestClient,
) -> None:
    page_one = pagination_client.get("/labels")
    assert page_one.status_code == 200
    body_rows = page_one.text.split("<tbody>")[1].split("</tbody>")[0]
    assert body_rows.count("<tr>") == 100
    assert '<span class="disabled">' in page_one.text
    assert "Prev" in page_one.text


def test_labels_route_shows_filter_summary_when_active(labels_client: TestClient) -> None:
    response = labels_client.get("/labels", params={"verdict": "match"})
    assert response.status_code == 200
    assert "Showing 2 of 3 labels" in response.text
    assert "verdict=match" in response.text
    assert "Clear filters" in response.text


def test_labels_route_nav_link_present_on_other_pages(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/labels"' in response.text


def test_card_renders_extended_cce_fields(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Cambridge, Mass." in response.text
    assert "Author is claimant" in response.text
    assert "2nd ed." in response.text
    assert "New York; London" in response.text
    assert "vi, 200 p." in response.text
    assert "added chapter 5" in response.text
    assert "2c." in response.text
    assert "first note" in response.text
    assert "second note" in response.text
    assert "1953-04-01" in response.text
    assert "1953-06-01" in response.text
    assert "1953-04-02" in response.text


def test_card_omits_extended_cce_rows_when_absent(empty_client: TestClient) -> None:
    response = empty_client.get("/")
    assert response.status_code == 200
    assert "author place" not in response.text
    assert "Author is claimant" not in response.text
    assert "new matter claimed" not in response.text
    assert "affidavit date" not in response.text
    assert "lccn.loc.gov" not in response.text
    assert "previous registrations" not in response.text


def test_card_renders_lccn_as_lccn_loc_gov_link(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="https://lccn.loc.gov/28000854"' in response.text
    assert ">28000854<" in response.text


def test_card_renders_prev_regnums_row(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "previous registrations" in response.text
    assert "A100000; A200000" in response.text


def test_card_renders_extended_marc_fields(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "publication place" in response.text
    assert ">New York<" in response.text
    assert "date (raw)" in response.text
    assert "c1953." in response.text
    assert ">extent<" in response.text
    assert "xxiv, 841 p." in response.text
    assert ">isbns<" in response.text
    assert "9780000000000" in response.text
    assert ">oclc<" in response.text
    assert 'href="https://www.worldcat.org/oclc/0001"' in response.text
    assert "title parts" in response.text
    assert "Pt. 2: The empire of Sebastopol" in response.text


def test_card_omits_extended_marc_rows_when_absent(empty_client: TestClient) -> None:
    response = empty_client.get("/")
    assert response.status_code == 200
    assert "publication place" not in response.text
    assert "date (raw)" not in response.text
    assert "title parts" not in response.text
    assert ">isbns<" not in response.text
    assert ">oclc<" not in response.text
    assert "worldcat.org" not in response.text


def test_card_renders_predicted_status_pd_chip(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert 'class="status status-pd"' in response.text
    assert "PD_REGISTERED_NOT_RENEWED" in response.text


def test_card_renders_predicted_status_in_copyright_chip(
    in_copyright_client: TestClient,
) -> None:
    response = in_copyright_client.get("/")
    assert response.status_code == 200
    assert 'class="status status-in_copyright"' in response.text
    assert "IN_COPYRIGHT_REGISTERED_AND_RENEWED" in response.text


def test_card_omits_predicted_status_chip_when_absent(
    no_predicted_status_client: TestClient,
) -> None:
    response = no_predicted_status_client.get("/")
    assert response.status_code == 200
    assert 'class="status status-' not in response.text
    assert "predicted status" not in response.text


def test_card_renders_renewal_details_block_when_populated(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Renewal details" in response.text
    assert "1968-05-15" in response.text
    assert "Estate of Jane Doe|PWH" in response.text
    assert "added chapter 7" in response.text
    assert "R200001" in response.text
    assert "A111111" in response.text


def test_card_renders_renewal_claimants_diff_marker_when_registration_disagrees(
    client: TestClient,
) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "claimants-diff" in response.text


def test_card_omits_renewal_details_block_when_no_renewal_fields(
    no_predicted_status_client: TestClient,
) -> None:
    response = no_predicted_status_client.get("/")
    assert response.status_code == 200
    assert "Renewal details" not in response.text
    assert "renewal date" not in response.text
