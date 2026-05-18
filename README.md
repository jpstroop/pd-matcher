# pd-matcher

A CLI that matches large MARC XML collections against the U.S. Copyright
Office's Catalog of Copyright Entries (CCE), published by the Library of
Congress and transcribed into XML/TSV by NYPL (registrations 1923–1977,
renewals 1950–2006), and assigns each MARC record a public-domain status with a
confidence score.

See `data/ANALYSIS_RESULTS.md` for the empirical findings that drive the
scoring strategy.

## Setup

After cloning, install dependencies with `pdm install`, then wire up the
pre-commit hook with `pdm run pre-commit install`. The hook formats and
lints touched Python files and applies standard whitespace hygiene on
every commit; the heavier gates (`mypy`, `pytest`) run via `pdm run gates`
and (eventually) CI.
