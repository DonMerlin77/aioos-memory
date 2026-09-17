"""
Closing the loop: does the memory-weighting layer fix the failure the RETRIEVAL EVAL found?

The eval (github.com/DonMerlin77/aioos-retrieval-eval) reports hard-negative leakage: how often the
embedder pulls a tempting-but-wrong memory into the top-k. Its hardest traps are STALE-vs-current pairs
(a bridge washed out but since rebuilt, a well contaminated but since cleaned). On those, pure embedding
similarity leaks the stale memory, because it is time-blind.

This takes the eval's OWN stale-trap queries and its OWN memory texts (bundled here as eval_memories.json)
and compares embed-only retrieval against the weighting layer (subject-grouping + recency,
memory_weighting.py). The recency is not fabricated: in each pair the "current" memory is genuinely the
later one. If the layer works, stale-leakage on these traps drops.

  python3 memory_weighting_eval.py
"""
import json
from pathlib import Path
from memory_weighting import MemoryStore, Memory

EVAL = Path(__file__).parent / "eval_memories.json"

# the eval's stale-vs-current traps: (query, current-correct id, stale id, shared subject)
STALE_TRAPS = [
    ("Is the north road passable now?",                     "m76", "m2",  "north_bridge"),
    ("Should I warn travelers away from the north bridge?", "m76", "m2",  "north_bridge"),
    ("Is the well water safe to drink today?",              "m78", "m10", "town_well"),
    ("Is the well still making people sick?",               "m78", "m10", "town_well"),
    ("Is the western pass usable for a wagon right now?",    "m77", "m13", "west_pass"),
]


def build_store():
    d = json.load(open(EVAL))
    text = {m["id"]: m["text"] for m in d["memories"]}
    current = {c for _, c, _, _ in STALE_TRAPS}
    stale = {s for _, _, s, _ in STALE_TRAPS}
    subj = {}
    for _, c, s, sub in STALE_TRAPS:
        subj[c] = subj[s] = sub
    store = MemoryStore()
    for mid, t in text.items():
        # real temporal order: stale memory older (tick 1), its current update newer (tick 5),
        # everything else neutral (tick 3). Each unrelated memory is its own subject so it never groups.
        learned = 1 if mid in stale else 5 if mid in current else 3
        store.add(Memory(mid, t, subj.get(mid, mid), learned, 0.8))
    return store


def top3(store, query, weighted):
    ranked, _, _ = store.rank(query, weighted=weighted)
    return [store.mems[i].id for i in ranked[:3]]


def main():
    store = build_store()
    print("== stale-leakage on the eval's stale-vs-current traps: embed-only vs weighted ==\n")
    emb_leaks = wtd_leaks = emb_current = wtd_current = 0
    for q, cur, stale, sub in STALE_TRAPS:
        e, w = top3(store, q, False), top3(store, q, True)
        el, wl = stale in e, stale in w
        ec, wc = cur in e, cur in w
        emb_leaks += el; wtd_leaks += wl; emb_current += ec; wtd_current += wc
        print(f"  {q}")
        print(f"    embed-only top3 {e}  stale-leak={'YES' if el else 'no'}  current@top3={'yes' if ec else 'NO'}")
        print(f"    weighted   top3 {w}  stale-leak={'YES' if wl else 'no'}  current@top3={'yes' if wc else 'NO'}")
    n = len(STALE_TRAPS)
    print(f"\n  STALE LEAKED into top3:   embed-only {emb_leaks}/{n}   ->   weighted {wtd_leaks}/{n}")
    print(f"  CURRENT reached top3:     embed-only {emb_current}/{n}   ->   weighted {wtd_current}/{n}")


if __name__ == "__main__":
    main()
