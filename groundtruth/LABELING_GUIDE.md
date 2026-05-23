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
`label_vault.jsonl` and re-applied automatically the next time
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
