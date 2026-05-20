"""Princeton MARC dump acquisition for the public-domain ground-truth corpus.

This subproject streams Princeton ``bibdata`` MARC dumps, filters them to the
in-scope slice (English/French/German/Spanish/Italian monographs published in
the CCE-relevant window), and writes the survivors as lossless MARCXML shards
for a later human-review phase.
"""
