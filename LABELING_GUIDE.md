# Labeling guide

A decision guide for the human reviewer in the pd-groundtruth review UI. See
`README.md` for pipeline mechanics; this doc is purely about how to label.

## Verdicts

- **match** — same intellectual work, same edition, same language, same
  publisher (and edition where stated). The CCE entry is the U.S. copyright
  registration for the artifact the MARC record describes.
- **no_match** — the pair is not the same registration. Use a reason chip to
  record *why* (different work, different author, translation, garbled
  transcription, etc.).
- **unsure** — there is genuine ambiguity even after looking. Prefer this to
  guessing. Use a reason chip to record what the doubt is.
- **skip** — keyboard `s`/`space`. Use sparingly; it leaves the pair in the
  queue. Prefer `unsure` so the doubt is recorded.

## Translation rules

Translations are the single most common ambiguity. The rule is grounded in
17 USC §103(b): a translation is a derivative work with its own, independent
copyright that does not affect the underlying original's copyright. The two
registrations therefore have independent clocks.

- Same intellectual work, **same language**, same edition, same publisher →
  `match`.
- Same intellectual work, **different language** (MARC describes the original;
  CCE describes a translation, or vice versa) → `no_match` with reason
  `translation`. These are legally distinct registrations.
- Same intellectual work, ambiguous whether the CCE entry is for the original
  or for a translation → `unsure` with reason `translation`.

CCE tells that the registration is for a translation (not the original):
- Phrases in the CCE title/note like "translated by", "translation of",
  "[Translation]".
- Author field on the CCE side names the *translator*, not the original
  author.
- Claimant is the translator or translating publisher.

## E-book reprint handling

When the card shows the yellow **E-book reprint** badge, the MARC record's
`extent` field contains "online resource" — meaning Princeton's record
describes a digital reissue, not the original publication. Year and publisher
in the MARC side typically belong to the digital reissue (e.g., 2010s reprint
by a digital aggregator), not to the original artifact the CCE entry was
written for.

For now: label normally, but treat year/publisher mismatches as expected
artifacts of the reissue, not as scoring failures. Use `reprint_or_format` on
the unsure side if the underlying work looks right but the digital reissue
muddies the comparison.

E-book reprints are filtered out at acquire time
([#30](https://github.com/jpstroop/pd-matcher/issues/30)). If you ever see
the yellow **E-book reprint** badge on a card, it means the filter missed an
indicator — please note the `pair_id` so we can extend the filter. The
[label vault](README.md#the-label-vault) (implemented in
[#28](https://github.com/jpstroop/pd-matcher/issues/28)) is what makes that
rebuild safe — every adjudicated verdict is persisted to
`data/label_vault.jsonl` and re-applied automatically the next time
`build-queue` runs.

## When to use each chip

### no_match reasons

- **Different work / title collision** — same title string, different works.
- **Same title, different author** — author disagreement clinches it.
- **Wrong year or edition** — distinct edition, not the registered one.
- **Translation / different language** — different language; treat as
  independent registrations (see above).
- **Garbled transcription** — the CCE side is too corrupted to be the same
  record.
- **Publisher-only overlap** — only the publisher matches; everything else
  diverges.
- **Generic title — likely a collision** — title is so generic ("Collected
  Works", "Annual Report") that overlap is almost certainly coincidental.

### unsure reasons

- **Insufficient data on one side** — one side is missing fields needed to
  decide.
- **Plausible but unverified** — looks right but cannot confirm.
- **Unsure about edition** — likely the same work but uncertain which
  edition.
- **Possibly a translation vs. original** — language ambiguity; cannot tell
  if the CCE entry registers the original or a translation.
- **Looks right but publisher differs** — work + author + year align, but
  publisher does not — possibly a co-publication or a reissue.
- **Reprint / different physical format (e-book, microform)** — same work
  but artifacts are in different physical formats and that's affecting the
  comparison.
- **Possibly whole vs. part / volume** — one side may describe the whole
  set, the other a single volume.
- **Looks like one issue of a periodical** — the CCE entry appears to be a
  single issue of a serial rather than a monograph.

## Field annotations

Below the reason chips, the card surfaces a small annotation grid: five
fields (`title`, `author`, `publisher`, `year`, `edition`) crossed with four
judgments. Use it to flag *which fields the scorer got wrong*, so the future
learned scorer has structured per-field signal to train on. Annotations are
**optional** — annotate only what you actively noticed; leave the rest blank.

Click a cell to flag a field; click again to clear. Only one judgment per
field (it's a row-radio, not a chip).

The four judgments:

- **correct** — the scorer's per-field reading agrees with reality. Use this
  sparingly to highlight fields where the scorer did well *despite* a tricky
  surface form (e.g. it correctly matched "Doe, J." to "Jane Doe").
- **overscored** — the scorer said the field matches, but on inspection it
  doesn't. This is a per-field false positive: the scorer overweighted a
  surface similarity that turned out to be coincidental. Common example:
  publisher "Macmillan" appears on both sides but they're different
  Macmillan imprints in different countries.
- **underscored** — the scorer said the field doesn't match, but actually it
  does once you account for transcription differences, abbreviations, or
  formatting. This is a per-field false negative: the scorer was too strict.
  Common example: author "Doe, John" vs. "John Doe" — same person, the
  scorer didn't know that.
- **n/a** — the field is missing on one or both sides, or otherwise not
  assessable. Use this when there's no signal to evaluate (e.g. the CCE side
  has no edition statement at all, so neither "correct" nor "overscored"
  makes sense for `edition`).

You don't have to annotate every field on every label. The intent is to
capture the *occasional* per-field surprises — the cases where you can see
the scorer making a systematic mistake. The aggregate of these annotations
on `…/stats` becomes the diagnostic for which per-field scorers need
re-tuning.

## Status assumptions catalog

The matcher computes a Cornell-categories PD status for every pair and
displays it as the `(estimate)` chip on the card. Each status leans on
**directly-observed facts** (publication year, country, presence/absence of
matching records in CCE) and on **documented assumptions** that bridge what
MARC + CCE can tell us to what U.S. copyright law actually requires.

This section catalogs every status's assumptions so you can sanity-check
a chip against the leaps the rule engine took. The matcher surfaces these
on each assessment internally; #47 will render them inline on the card.

### PD statuses

**`PD_BY_AGE_PRE_95_YEARS`** — work is older than the moving wall.
- *Observed:* `pub_year ≤ today.year − 95` (1931 as of 2026).
- *Assumptions:* none. Short-circuits before the rule engine runs.
- *Effectively cannot appear in our labeling queue* — acquire-time filter
  excludes anything below the wall.

**`PD_US_PUB_NO_NOTICE_1931_1977`** — US-published 1931–1977 with no
matching CCE registration found.
- *Observed:* country US, 1931 ≤ pub_year ≤ 1977, no matching registration.
- *Assumptions:*
  - **No notice inferred from absence of registration.** The 1909 Act
    required notice on published copies for a registration to be valid.
    A non-registration is treated as evidence the work also lacked
    notice (or that the absence has the same legal effect).
  - **Registration coverage is reliable for this pub year.** If outside
    the index's reliable registration window, the rule short-circuits
    to `UNKNOWN_INSUFFICIENT_COVERAGE` instead.

**`PD_REGISTERED_NOT_RENEWED`** — registered 1931–1963, no matching renewal.
- *Observed:* 1931 ≤ pub_year ≤ 1963, CCE registration found, no matching
  renewal record found.
- *Assumptions:*
  - **Notice was present**, inferred from the registration's existence
    (1909 Act required notice for registration validity).
  - **Renewal coverage is reliable for `pub_year + 27` AND `pub_year + 28`.**
    The 1909 Act §24 renewal window straddled two calendar years; both
    must be inside the renewal corpus for the absence to be trusted.
    Otherwise the rule short-circuits to `UNKNOWN_INSUFFICIENT_COVERAGE`.

**`PD_US_PUB_NO_REGISTRATION_1978_1989`** — US-published 1978–28 Feb 1989
with no CCE registration within the 5-year cure window.
- *Observed:* country US, 1978 ≤ pub_year ≤ 1989, no matching registration.
- *Assumptions:*
  - **No notice / no cure inferred from absence of registration** within
    the 5-year window the 1976 Act provided.
  - **Registration coverage is reliable for this pub year.** Otherwise
    returns `UNKNOWN_INSUFFICIENT_COVERAGE`. *In our pipeline this
    typically yields UIC because NYPL's CCE corpus ends Dec 31, 1977 —
    the legal regime boundary.*

**`PD_US_GOVERNMENT_WORK`** — work prepared by a US federal employee.
- *Observed:* country US, MARC publisher text matches a US-government
  regex pattern (GPO, `Department of …`, `Bureau of …`, National Park
  Service, Smithsonian, Library of Congress, National Archives, etc.).
- *Assumptions:*
  - **Publisher text alone is sufficient evidence of government
    authorship.** A private commercial reprint of a government work
    bearing a non-government publisher line could be misclassified.

**`PD_FOREIGN_IN_HOME_COUNTRY_PD_1996`** — pre-1923 foreign work without
US registration, presumed PD in source country by URAA baseline.
- *Observed:* foreign country, pub_year < 1923, no US registration.
- *Assumptions:*
  - **Author likely died before 1946**, so the work was in source-country
    PD by 1 Jan 1996 under life+50. Most pre-1923 foreign works satisfy
    this, but a work by an author who lived past 1946 would not.

**`PD_FOREIGN_NO_TREATY_COUNTRY`** — published in a country with no US
copyright relations (Eritrea, Ethiopia, Iran, Iraq, Marshall Islands,
San Marino).
- *Observed:* country in the no-treaty list.
- *Assumptions:* none. Country list is observed.

### IN_COPYRIGHT statuses

**`IN_COPYRIGHT_REGISTERED_AND_RENEWED`** — registered 1931–1963, renewal
found.
- *Observed:* 1931 ≤ pub_year ≤ 1963, registration found, renewal found.
- *Assumptions:*
  - **Notice was present**, inferred from registration. (Same caveat as
    `PD_REGISTERED_NOT_RENEWED`.)

**`IN_COPYRIGHT_1964_1977_WITH_NOTICE`** — registered 1964–1977 (no
renewal needed; the 1992 Copyright Renewal Act made renewal automatic
for this cohort).
- *Observed:* 1964 ≤ pub_year ≤ 1977, registration found.
- *Assumptions:*
  - **Notice was present**, inferred from registration.
  - **Automatic renewal applies** — 1992 Copyright Renewal Act made
    renewal automatic for 1964–77 registrations. The status name
    `WITH_NOTICE` is somewhat misleading; the actual load-bearing fact
    is the automatic-renewal regime, not notice (#48 will rename it).

**`IN_COPYRIGHT_1978_1989_CURED`** — registered 1978–28 Feb 1989; the
5-year registration cure preserved copyright even if notice was omitted
at publication.
- *Observed:* 1978 ≤ pub_year ≤ 1989, registration found.
- *Assumptions:*
  - **No notice → cure mechanism applies.** The rule doesn't try to
    verify whether notice was actually present at publication; the
    registration's existence within the cure window is sufficient.
  - **Registration coverage is reliable** for this pub year. *In our
    pipeline this typically does not fire because NYPL's CCE corpus
    ends Dec 31, 1977.*

**`IN_COPYRIGHT_US_PUB_POST_1989`** — US-published 1 March 1989 or later
under Berne (no notice required).
- *Observed:* country US, pub_year ≥ 1990.
- *Assumptions:* none. *In our pipeline this rule never fires* — acquire
  excludes everything past 1977.

**`IN_COPYRIGHT_PRE_1978_PUBLISHED_1978_2002_FLOOR`** — foreign work
published 1978–2002 with presumed pre-1978 creation date (31 Dec 2047
floor).
- *Observed:* foreign country, 1978 ≤ pub_year ≤ 2002, no US registration,
  not a no-treaty or delayed-URAA country.
- *Assumptions:*
  - **Pre-1978 creation date.** Posthumous or delayed publication of an
    earlier-written manuscript is assumed; that's the case the floor
    rule was written for.

**`IN_COPYRIGHT_FOREIGN_URAA_RESTORED`** — foreign 1931–1977 work without
US registration, URAA restoration date Jan 1, 1996.
- *Observed:* foreign country, 1931 ≤ pub_year ≤ 1977, no US registration,
  not a delayed-URAA country.
- *Assumptions:*
  - **Registration coverage is reliable** for this pub year (otherwise
    UIC). The absence of US registration is what triggers URAA's
    formality-failure restoration.

**`IN_COPYRIGHT_FOREIGN_POST_1989`** — foreign work published 2003 or
later under Berne (no formalities).
- *Observed:* foreign country with treaty relations, pub_year ≥ 2003.
- *Assumptions:* none. *Cannot fire in our pipeline* — acquire excludes
  anything past 1977.

### UNKNOWN statuses

**`UNKNOWN_INSUFFICIENT_DATA`** — country has a delayed Berne/WTO
accession (Afghanistan, Andorra, Bhutan, etc.) and we'd need its
specific restoration date to decide.
- *Observed:* country in the delayed-URAA list.
- *Assumptions:* none — this is an honest "we don't know" with a
  specific reason.

**`UNKNOWN_INSUFFICIENT_COVERAGE`** — the rule that would apply depends
on absence-of-evidence (e.g., "no registration → PD") but the pub year
or renewal year is outside our index's reliable coverage, so absence
isn't trustworthy evidence.
- *Observed:* relevant year falls outside the index's coverage window.
- *Assumptions:* none — also an honest "we don't know."
- This is the safety net for boundary years. In practice rare with the
  current corpus + #44's coverage fix; would dominate any extension to
  post-1977 records without corresponding NYPL data.

**`UNKNOWN_NO_RULE_MATCHED`** — none of the configured rules matched.
- *Observed:* fall-through. Should be rare; usually indicates an
  unanticipated combination (e.g., foreign work in our queue that the
  category-3 rules didn't anticipate).
- *Assumptions:* none.

### A general note on assumptions

Most of the assumptions above are forms of *"absence of evidence is
evidence of absence,"* which is sound only when we have the data to
look in. The coverage-awareness work (#39 + #44) systematically protects
against false-negative-driven over-claiming PD. Statuses that don't
depend on absence (e.g., `PD_BY_AGE_PRE_95_YEARS`, `IN_COPYRIGHT_REGISTERED_AND_RENEWED`)
are conclusions from positive observations and don't carry that risk.

The biggest remaining assumption is **notice inferred from registration**.
Books almost universally bore notice in the pre-1978 era because
registration without notice was procedurally invalid, but rare
exceptions exist (registration errors, etc.). The matcher cannot
detect those edge cases from MARC + CCE alone; a labeler reviewing
the card has more context than the engine.
