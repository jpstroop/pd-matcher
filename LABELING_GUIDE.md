# Labeling guide

A decision guide for the human reviewer in the pd-groundtruth review UI. See
`README.md` for pipeline mechanics; this doc is purely about how to label.

## Verdicts

- **match** — same intellectual work, same edition, same language, same
  publisher (and edition where stated). The CCE entry is the U.S. copyright
  registration for the artifact the MARC record describes.
- **no_match** — the pair is not the same registration. Capture the *why* in
  the note field if anything about the call is worth flagging.
- **unsure** — there is genuine ambiguity even after looking. Prefer this to
  guessing. The note field is the right place to record what the doubt is.
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
  CCE describes a translation, or vice versa) → `no_match`. These are legally
  distinct registrations; mention the translation explicitly in the note so
  the pattern is recoverable.
- Same intellectual work, ambiguous whether the CCE entry is for the original
  or for a translation → `unsure` and note the language ambiguity.

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
artifacts of the reissue, not as scoring failures. If the underlying work
looks right but the digital reissue muddies the comparison, prefer `unsure`
and note "reprint / format mismatch" so the pattern surfaces in later
analysis.

E-book reprints are filtered out at acquire time
([#30](https://github.com/jpstroop/pd-matcher/issues/30)). If you ever see
the yellow **E-book reprint** badge on a card, it means the filter missed an
indicator — please note the `pair_id` so we can extend the filter. The
[label vault](README.md#the-label-vault) (implemented in
[#28](https://github.com/jpstroop/pd-matcher/issues/28)) is what makes that
rebuild safe — every adjudicated verdict is persisted to
`data/label_vault.jsonl` and re-applied automatically the next time
`build-queue` runs.

## The note field

The note field is the only structured signal carried alongside the verdict.
It is optional — leave it blank when the verdict is obvious — but use it
freely when something is worth flagging. Useful things to capture:

- **What surprised you.** "Author looks right but publisher is from a
  different country" / "title matches verbatim but the work is clearly
  different" / "scorer underweighted the author here".
- **What made the call ambiguous.** "Could be a translation or the original
  — CCE entry has no language hint" / "two distinct works share this exact
  title in the same decade".
- **What the matcher seemed to get wrong.** "edition score is high but the
  CCE entry is the 1st ed. and the MARC is the 3rd" / "year window may be
  too tight — the registration date is one year past the publication year".

These notes are not meant to be read individually after the fact; they are
meant to be analyzed in aggregate to surface patterns the matcher and the
labeling vocabulary should learn to capture. Be specific, but don't worry
about consistency of phrasing — that's a job for the later analysis pass.
