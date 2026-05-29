# Lower-band cross-field diagnostic — 2026-05-28

## Source

`scripts/dump_lowband_noted_pairs.py` joins `data/label_vault.jsonl` against
`data/review.db` and prints the 25 most-recent vault entries that carry a
non-null `note`. All 25 slots were filled (vault holds 111 noted entries; the
25 most recent span 2026-05-28 14:00 to 23:29 UTC). The diagnostic is fixed
in time against the working vault; subsequent labels will not change the
analysis below.

## 1. Verdict distribution

| Verdict   | Count |
|-----------|------:|
| `match`   |    11 |
| `no_match`|     9 |
| `unsure`  |     5 |
| **Total** |    25 |

The labeler's `no_match` notes overwhelmingly describe **false positives** the
matcher should not have surfaced in the lower band — the band itself is
working as intended; the scoring is misallocating credit. The `match` notes
describe **false negatives** where the matcher under-scored a real pair
because the supporting evidence lived in unexpected fields. `unsure` notes
flag genuinely sparse CCE records.

## 2. Cross-field leakage patterns

Five recurring shapes, with verbatim quotes:

### 2a. CCE title carries MARC statement-of-responsibility / publisher / city

The CCE side frequently bakes responsibility, publisher, and even publication
place into the title string. The MARC side keeps those in dedicated subfields.
Result: a real match scores poorly on `title.token_set` because the CCE title
has tokens MARC title does not.

- **pair 419** (verdict `match`): MARC title `"Maverick town"`; CCE title
  `"Maverick town, the story of old Tascosa by John L. McCarty, with chapter decorations by Harold D. Bugbee."` —
  `title.token_set = 0.3794`. Labeler note: "Probably only scoring low because of the
  MARC responsibility being included in the CCE title."
- **pair 161** (verdict `match`): MARC title `"McGillivray"`; CCE title
  `"McGillivray, lord of the Northwest. Toronto, Clarke, Irwin."` — publication place
  ("Toronto") and publisher ("Clarke, Irwin") inlined into the CCE title.
  `title.token_set = 0.5079`. Labeler note: "MARC Publication place and publisher in CCE title."
- **pair 392** (verdict `match`): MARC publisher `"The Pomegranate Press"`; CCE title
  `"Tear gas rag. Pomegranate Press"`. `title.token_set = 0.5792`, `name.publisher = 0.3226`.
  Labeler note: "MARC publisher in CCE title."

Count in sample: at least 4 (pair_ids 419, 161, 392, 326).

### 2b. MARC statement-of-responsibility matches CCE author/claimants

When MARC `main_author` is the corporate body but the actual personal author
lives in `statement_of_responsibility`, CCE drops the personal name into
`author` or `claimants`. The matcher reads MARC `main_author`, misses the
overlap.

- **pair 30** (verdict `match`): MARC `main_author = "Historical Society of Temple, N.H"`;
  statement_of_responsibility = `"the Historical Society of Temple, New Hampshire"`;
  CCE `publishers / claimants = "Historical Society of Temple, New Hampshire, Inc."`.
  Labeler note: "Statement of responsibility matches claimant; CCE repeating claimant as publisher might be inaccurate."
- **pair 343** (verdict `match`): MARC `main_author` is the conference; the title
  string itself includes "R. P. Wei and R. I. Stephens, symposium cochairmen"; CCE
  publisher/claimants point to "American Society for Testing and Materials". The
  MARC publisher `"ASTM"` is an abbreviation of the CCE form (see 3 below).

Count in sample: at least 3 (pair_ids 30, 343, plus partial in 419).

### 2c. Whole-vs-part: CCE registers multi-volume work, MARC catalogs one piece

- **pair 406** (verdict `unsure` then `no_match` for the same pair): MARC title
  `"The Arab-Israeli conflict"`, extent `"v"`; CCE title `"The Arab-Israeli conflict. Vol.1-3"`.
  Same author, same publisher, same year. The `volume.compat` scorer fires at 0.0.
  Labeler note: "Whole/part. Hathi shows there is at least one more volume."

Count in sample: 1 distinct pair (labeled twice). Out of scope for normalization;
flagged here for the follow-up scorer-design conversation.

### 2d. Generic title carries the score on title alone

`"Proceedings"`, `"Report"`, `"Catalogue"`, `"Letters"` — when MARC and CCE
both use the same single-word generic title, `title.token_set = 1.0` and the
pair lands in `b70_80` even when nothing else lines up.

- **pair 317** (`no_match`): MARC title `"Proceedings"`, CCE title `"Proceedings"`;
  publishers, authors, places completely disjoint. Score 0.7240. Labeler note:
  "Generic title and nothing else matches."
- **pair 253** (`no_match`): MARC `"Report"` vs CCE `"Report."`; nothing else
  matches. Score 0.7540. Labeler note: "Generic title. No idea why there's any match between author and publisher??"
- **pair 103** (`no_match`): MARC `"Proceedings"` vs CCE `"Proceedings."`; corporate
  authors and publishers entirely disjoint. Score 0.7082.
- **pair 35** (`unsure`): same pattern, "Proceedings."
- **pair 294** (`no_match`): MARC series title `"Its Catalogues"`, CCE title
  `"Catalogue."`. Labeler note: "Series title and cce title should probably not
  indicate a match when nothing else matches."

Count in sample: 5. Out of scope for normalization (this is an IDF / corpus
frequency problem) but a major driver of `no_match` notes — flagged for the
follow-up.

### 2e. Publisher-only match on noise tokens

Publisher fields collide on noise (`& Co`, `Inc.`, `publishing`, `press`) when
the real publisher tokens are disjoint.

- **pair 378** (`no_match`): MARC `"Sam'l Gabriel Sons & Co"` vs CCE
  `"Dennis & co., inc."`. `name.publisher = 0.4118`, all overlap on `&`, `Co`.
  Labeler note: "Publisher only match and only on '& Co', which should be factored out."
- **pair 236** (`no_match`): MARC `"State Art Pub"` vs CCE `"Hebrew pub. co."`.
  `name.publisher = 0.4615` — the overlap is `pub.` Labeler note: "'pub' and
  other forms of the word 'publish' should be a stop word in publisher fields.
  That's probably why it scored higher than it should."
- **pair 189** (`no_match`): MARC `"published for the Palestine Labour Studies Group"`
  vs CCE `"College of Jewish studies"`. `name.publisher = 0.5484` largely on
  the `studies` overlap. Labeler note: "Matching publisher is never enough."

Count in sample: at least 3 (pair_ids 378, 236, 189). This is the largest
**addressable** pattern in the sample — and it's the central target of Commit 2.

## 3. Abbreviation patterns

Tokens observed in the dump where the MARC and CCE sides disagree on
abbreviated vs. expanded form (or use a noise word that should be a stopword
after normalization):

| Abbreviation | Expanded form | Pair_ids |
|---|---|---|
| `Inc.` / `inc.` | `incorporated` | 378 (CCE "inc."), 52 ("Carrick & Evans, inc"), 30 ("…N.H., Inc."), 18 (not present here but common), 326 (Polish "Wydawn." – out of scope) |
| `Co.` / `co.` | `company` | 378 (MARC "& Co", CCE "co., inc."), 236 (CCE "co.") |
| `&` | `and` | 378, 18 ("Mr. & Mrs."), 35 ("Testing & Materials"), 7 ("Testing and Materials") — MARC and CCE inconsistent on ampersand vs word |
| `Pub.` | `publishing` | 236 (MARC "State Art Pub", CCE "Hebrew pub. co.") |
| `Assn.` | `association` | 253 ("American Insurance Assn."), 343 (not abbreviated here but related corporate-suffix pattern) |
| `Bros.` | `brothers` | not observed in the 25-pair sample |
| `Ltd.` | `limited` | not observed in the 25-pair sample |
| `Corp.` | `corporation` | not observed in the 25-pair sample |
| `Soc.` | `society` | not abbreviated in the sample (full form "Society" appears in pair 343 / 35) |

The cataloging conventions for `Bros.`/`Ltd.`/`Corp.`/`Soc.` are well-attested
in the wider corpus even if absent from this 25-pair window; Commit 2 will
include them defensively so that future labeling sessions surface fewer of
these.

Personal-name initial expansion (`J. K.` vs `Joanne K.`) was **not** seen as
a recurring failure mode in this sample — `pair 419` has `"John L. McCarty"`
on both sides and the labeler's complaint there is the CCE title pollution,
not the initial. Skipping per plan.

## 4. Stopword candidates

Publisher noise words observed across the 25 pairs:

| Token | Why it is noise | Pair_ids |
|---|---|---|
| `&`, `and` | Pure conjunction | 378, 18, 35, 7 |
| `co`, `company` | Corporate suffix; carries no entity signal | 378, 236 |
| `inc`, `incorporated` | Corporate suffix | 378, 52, 30 |
| `pub`, `publ`, `pubs`, `publishing`, `publishers`, `publisher` | The word "publisher" cannot itself distinguish publishers | 236, 47 ("Printed at the Princeton university press") |
| `press` | Same logic | 343, 47, 161 |
| `society`, `soc`, `association`, `assn` | Generic organizational suffix | 343, 253, 35 |
| `books`, `book` | Generic suffix; `"Galahad Books"` vs `"Arlington House"` collides on no real signal | 18 |
| `bros`, `brothers` | Corporate suffix | (not in sample; defensive) |
| `ltd`, `limited`, `corp`, `corporation` | Corporate suffix | (not in sample; defensive) |

**Scope: publisher only.** The plan's critical constraint holds: adding
`press`, `publishing`, `books`, `company` to **title** stopwords would destroy
the signal in titles like `"Penguin Books"`, `"University Press"`, the inline
publisher-in-title cases observed in pattern 2a. The publisher field is the
right place.

`&` is a special case. The non-English stopword files already treat `&` as a
publisher and author stopword (alongside `et`, `und`, `e`, `y`); the English
file does not. Adding `&` to English publisher stopwords closes that gap.

Languages other than English: the diagnostic sample includes a Polish entry
(pair 326), Hebrew/Yiddish entries (pairs 236, 189, 130). The Polish case
("Państwowe Wydawn. Naukowe") would benefit from a Polish abbreviation
table, but pd_matcher has no Polish support today — out of scope. The
Hebrew/Yiddish cases are language-mismatch failures (MARC labels them `eng`),
which is a separate scope.

## 5. Recommendations for Commit 2

### Will ship in Commit 2

**Extend `_ABBREVIATIONS` in `src/pd_matcher/normalize/numbers.py`** with the
publisher-suffix and corporate-form abbreviations directly supported by the
diagnostic plus the defensively-included cataloging-conventional ones:

- `inc` → `incorporated`
- `corp` → `corporation`
- `co` → `company`
- `bros` → `brothers`
- `ltd` → `limited`
- `pub` → `publishing`
- `pubs` → `publishing`
- `publ` → `publishing`
- `soc` → `society`
- `assn` → `association`
- `assoc` → `association`

**Extend English publisher stopwords only** in
`src/pd_matcher/normalize/stopwords_data/english_stopwords.json`:

- `&`, `company`, `co`, `incorporated`, `inc`, `corporation`, `corp`,
  `brothers`, `bros`, `limited`, `ltd`, `publishing`, `publishers`,
  `publisher`, `pub`, `publ`, `pubs`, `press`, `society`, `soc`,
  `association`, `assn`, `books`, `book`

Both the abbreviated and expanded forms are listed so the stopword pass
catches the token whether or not the abbreviation regex (which requires a
trailing period) had a chance to fire.

**Do NOT add these to English title or author stopwords.** Titles like
`"Penguin Books"`, `"The Story of the Society"`, and the inline-publisher
cases in pattern 2a would lose their distinguishing tokens.

**Do NOT touch French/German/Italian/Spanish stopword files.** The diagnostic
did not surface non-English publisher-noise patterns.

### Out of scope for this commit; flagged for the follow-up

- **Generic-title problem** (pattern 2d): `"Proceedings"`, `"Report"`,
  `"Catalogue"` carrying `title.token_set = 1.0` with no other matching
  evidence. This is a corpus-frequency problem, not a normalization problem.
  Candidate fix is an IDF down-weight at scorer level or a guard that
  refuses to credit `title.token_set` when only one (or few) tokens overlap.
- **Cross-field leakage** (pattern 2a, 2b): a bag-of-tokens cross-side scorer
  that aggregates all text on each side and computes IDF-weighted Jaccard.
  Captures publisher-in-title and statement-of-responsibility-in-claimants
  without enumerating every possible pairing.
- **Whole/part volume mismatch** (pattern 2c): better `volume.compat` logic
  that treats MARC extent `"v"` (multi-volume work) as compatible with CCE
  `"Vol.1-3"`.
- **Personal-name initial expansion**: not warranted by this sample.
