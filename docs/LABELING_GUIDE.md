# Labeling guide

A decision guide for the human reviewer in the review UI. See `README.md` for pipeline mechanics; this doc is purely about how to label.

## Verdicts

- **match** — the CCE entry is the U.S. copyright registration for the work the MARC record describes. Same work, same edition, same language.
- **no_match** — the pair is not the same registration. Capture the *why* in the note field if anything about the call is worth flagging.
- **unsure** — there's genuine ambiguity even after looking. Prefer this to guessing. Use the note to record what the doubt is.
- **skip** — keyboard `s` / `space`. Use sparingly; the pair stays in the queue. Prefer `unsure` so the doubt is recorded.

## Different language → no_match

The most common false positive: same title, same author, different language. A 1953 Spanish translation and the 1925 English original are **different copyright registrations** with different registration numbers in the CCE — they share an intellectual work but they're not the same record. Mark these `no_match` and mention "translation" or the language in the note so the pattern is recoverable in later analysis.

CCE tells the labeler the registration is for a translation rather than the original:

- Phrases in the CCE title / note like "translated by", "translation of", "[Translation]"
- Author field on the CCE side names the *translator*, not the original author
- Claimant is the translator or translating publisher

When you genuinely can't tell whether the CCE entry registers the original or a translation → `unsure`, note the ambiguity.

## Series and sets — asymmetric rule

When MARC and CCE describe different scopes within the same series or set, the verdict depends on which side is the broader entity. The question we're answering is *"does this CCE registration cover the work the MARC describes?"* — not *"are these the same bibliographic entity?"* Coverage flows from the broader registration to the narrower work, not the other way around.

- **MARC = series/set; CCE = single member → `no_match`.** The CCE registration covers only the specific volume; calling the pair a match would imply the registration covers the whole series, which it doesn't. This is the common case (Princeton's MARC frequently describes whole multi-volume sets; CCE registrations are usually per-volume).
- **MARC = single member; CCE = series/set → `match`.** A series-level registration covers the members of the series, so the member-level MARC is genuinely covered by that registration. This is rarer but real. Capture *"series-level CCE"* (or similar) in the note so the inference is visible downstream — a future consumer needs to know this is a series-coverage match, not an exact-volume match.

When you can't tell whether a CCE registration is series-level or volume-level → `unsure`, note the doubt.

## Country/publisher divergence (transatlantic editions)

Divergence in publisher and publication country alone is never grounds for `no_match`.

- If the connection can be affirmatively established — a documented imprint or co-publishing relationship, renewal data confirming the same translation or setting, or overwhelming content identity (unique title + author + year) — label `match`, and add the `same_work_foreign_publication` category when the CCE registration's publication event is in a different country from the MARC's.
- If the connection cannot be established, label `unsure` with `[reasons: pub_differs]`.
- `no_match` requires a content-level disagreement: a different work, a different title, or an incompatible extent within the same publisher.

Evidence anchors:

- Same-year publication of the same title/author across countries is presumptively the same text. Simultaneous transatlantic publication was standard practice under the 1909 Act: its ad interim provisions pushed publishers of English-language works first published abroad to secure US publication quickly (this is the mechanism behind the AI registration class).
- Extent differences across different publishers are weak evidence of content difference. The manufacturing clause required US typesetting for full-term protection, so page counts routinely differ between printings of identical text. Extent is a strong differentiator only within the same publisher.

## E-book reprint badge

When the card shows a yellow **E-book reprint** badge, the MARC record's `extent` field contains "online resource" — Princeton's record describes a digital reissue, not the original publication. Year and publisher in the MARC side belong to the digital reissue (e.g. a modern aggregator), not to the original artifact the CCE entry would have registered.

E-book reprints are filtered out at acquire time ([#30](https://github.com/jpstroop/pd-matcher/issues/30)). **If you ever see this badge, the filter missed an indicator.** Note the `pair_id` and mention "e-book" in the free-text note; we'll extend the filter.

## The note field

The note is the only structured signal alongside the verdict. Optional — leave it blank when the call is obvious — but use it freely when something is worth flagging. Useful things to capture:

- **What surprised you.** "Author looks right but publisher is from a different country" / "title matches verbatim but the work is clearly different".
- **What made the call ambiguous.** "Could be a translation or the original — CCE entry has no language hint" / "two distinct works share this exact title in the same decade".
- **What the matcher seemed to get wrong.** "scorer underweighted the author here — Doe vs J. Doe" / "year window may be too tight — reg date is one year past the publication year".

Notes are not meant to be read individually after the fact; they get analyzed in aggregate to surface patterns the matcher and the labeling workflow should learn to capture. Be specific, but don't worry about consistency of phrasing — that's a job for the later analysis pass.
