"""Scan the CCE registration corpus once, build blob-IDF, cache to msgpack.

Document frequency is computed over each registration's FULL blob (all
descriptive-text fields; see blob_common._cce_text_values). One document per
registration. Cached so the measurement script never re-scans.
"""

import sys
from pathlib import Path
from time import time

sys.path.insert(0, "/tmp/blob")

from msgspec.msgpack import Encoder

from blob_common import cce_blob
from pd_matcher.index.lookup import NyplIndexLookup

_INDEX = Path("caches/cce.lmdb")
_OUT = Path("/tmp/blob/blob_idf.msgpack")


def main() -> None:
    df: dict[str, int] = {}
    n = 0
    t0 = time()
    with NyplIndexLookup(_INDEX) as lk:
        source_hash = lk.stats().source_hash
        for rec in lk.iter_registrations():
            for token in cce_blob(rec):
                df[token] = df.get(token, 0) + 1
            n += 1
            if n % 200000 == 0:
                el = time() - t0
                print(f"  {n} regs, {len(df)} tokens, {el:.0f}s", flush=True)
    el = time() - t0
    print(f"done: {n} regs, {len(df)} distinct tokens, {el:.0f}s", flush=True)
    payload = {"document_count": n, "source_hash": source_hash, "df": df}
    _OUT.write_bytes(Encoder().encode(payload))
    print(f"wrote {_OUT} ({_OUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
