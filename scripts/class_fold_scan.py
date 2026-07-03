"""Measure join recovery from folding AI[O0]/AF[O0] class variants (+ zero-padded numerics)."""
from pathlib import Path
from re import compile as re_compile

from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.index.codec import make_renewal_keys
from pd_matcher.normalize.registration_numbers import normalize_regnum

_CLASS_FOLD = re_compile(r"^(A[IF])[O0](?=\d)")
_LEAD_ZEROS = re_compile(r"^([A-Z]+)0+(\d)")

def fold(norm: str) -> str:
    out = _CLASS_FOLD.sub(r"\1", norm)
    out = _LEAD_ZEROS.sub(r"\1\2", out)
    return out

print("building registration keyspace...", flush=True)
exact_keys: set[str] = set()
with NyplIndexLookup(Path("caches/cce.lmdb")) as lk:
    n = 0
    for reg in lk.iter_registrations():
        n += 1
        if reg.regnum and reg.reg_year:
            for k in make_renewal_keys(reg.regnum, reg.reg_year):
                exact_keys.add(k.decode())
        if n % 500000 == 0:
            print(f"  {n} regs scanned", flush=True)
folded_keys = {fold(k.split("|")[0]) + "|" + k.split("|")[1] for k in exact_keys}
print(f"regs: {n}, exact keys: {len(exact_keys)}, folded keys: {len(folded_keys)}", flush=True)

rows = 0
with_cite = 0
exact_hit = 0
recovered = 0
recovered_by_class: dict[str, int] = {}
for tsv in sorted(Path("data/nypl-ren/data").glob("*.tsv")):
    with tsv.open(encoding="utf-8", errors="replace") as fp:
        header = fp.readline()
        for line in fp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            oreg, odat = parts[7].strip(), parts[8].strip()
            rows += 1
            if not oreg or len(odat) < 4 or not odat[:4].isdigit():
                continue
            with_cite += 1
            key = f"{normalize_regnum(oreg)}|{odat[:4]}"
            if key in exact_keys:
                exact_hit += 1
                continue
            fkey = fold(key.split("|")[0]) + "|" + key.split("|")[1]
            if fkey in folded_keys and fkey != key:
                recovered += 1
                prefix = oreg[:3]
                recovered_by_class[prefix] = recovered_by_class.get(prefix, 0) + 1
print(f"renewal rows: {rows}, with oreg+odat: {with_cite}")
print(f"exact key hits: {exact_hit}")
print(f"RECOVERED by class/zero folding: {recovered}")
for k, v in sorted(recovered_by_class.items(), key=lambda x: -x[1])[:12]:
    print(f"  {k}: {v}")
