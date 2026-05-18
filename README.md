# pd-matcher

A CLI that matches large MARC XML collections against NYPL's transcriptions of
the Library of Congress Catalog of Copyright Entries (registrations 1923–1977,
renewals 1950–2006) and assigns each MARC record a public-domain status with a
confidence score.

See `data/ANALYSIS_RESULTS.md` for the empirical findings that drive the
scoring strategy.

## Setup

After cloning, install dependencies with `pdm install`, then wire up the
pre-commit hook with `pdm run pre-commit install`. The hook formats and
lints touched Python files and applies standard whitespace hygiene on
every commit; the heavier gates (`mypy`, `pytest`) run via `pdm run gates`
and (eventually) CI.
