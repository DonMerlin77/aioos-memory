# AIOOS Memory

A small, honest memory layer for game NPCs: an NPC that **remembers, resolves stale and conflicting memories, evolves its behavior over time, and commits with calibrated doubt.**

It is the companion to the [AIOOS retrieval eval](https://github.com/DonMerlin77/aioos-retrieval-eval). That eval *measured* a problem; this repo *fixes* it, and measures the fix on the eval's own data.

## The problem the eval found

Embedding similarity is **time-blind**. "The bridge is washed out" and "the bridge was rebuilt" are both highly similar to "is the road passable?", and the embedder cannot tell which is *current*. On the eval's stale-vs-current traps, pure similarity surfaced the stale memory ~20% of the time. You cannot embed your way out of that; it is a **memory-layer** job.

## The core finding: you cannot detect contradictions with similarity

The first instinct is "cluster similar memories and prefer the newer one." It fails, and the failure is the interesting part: two memories that *contradict* each other about one subject can be embedding-**dissimilar**. "bandits at dusk" and "perfectly safe" are semantically opposite, so cosine puts them far apart even though they concern the same road. Similarity finds *topics*, not *conflicts*.

So detection needs **structure**, and confidence wants **entropy**:

- **Subject tags (detection).** Each memory carries the subject it is about (the place/person/thing), assigned at write-time. Memories are grouped by subject; within a group, a newer / more-trusted memory suppresses the stale one it contradicts.
- **Entropy (confidence).** Within a subject group, the belief mass (recency + trust) forms a distribution. Its entropy is the calibration: one dominant memory means low entropy, high confidence; two conflicting memories of similar standing means high entropy, so the NPC **commits with doubt** rather than either flip-flopping or freezing.

Two guardrails, both tested:
- a newer but **low-trust** memory does not silently override a trusted older one (trust enters the authority; the peddler who swears the road is safe does not erase the blacksmith's warning);
- it does **commit** to a best answer and a confidence, rather than refusing to choose.

## Results

**The living world (`world.py`).** Bran the tavern keeper is told the north bridge collapsed, warns a traveler away, then learns it was repaired. Asked the same "route north" question again, he now says the road is open, and the recalled set for the second answer no longer even contains the stale collapse memory (it was suppressed). Same question, different answer, because memory-writing changed behavior over time. Confidence drops on the second answer because the subject is now a resolved conflict.

**Closing the loop on the eval (`memory_weighting_eval.py`).** Run against the eval's own stale-trap queries and memory texts:

```
STALE LEAKED into top3:   embed-only 3/5   ->   weighted 0/5
CURRENT reached top3:     embed-only 4/5   ->   weighted 5/5
```

The layer eliminates stale-leakage on the exact cases the eval flagged.

## Honest limits

- **Stance-conflicts still need more than subject+recency.** "bandits" vs "safe" about the same road is handled *if* both carry the same subject tag, but a general system needs entailment/NLI (or an LLM check) to catch conflicts the tags miss. The eval-loop measurement shows one such distractor (a "road is safe" claim) that the layer correctly leaves alone rather than overreaching.
- **Recency and trust are simple heuristics here** (rank-normalized recency, a per-source trust scalar). Real trust and corroboration modeling is future work.
- **Single NPC, synthetic.** This demonstrates the mechanism; it is not a benchmark of memory systems in general.

## Files

- `memory_weighting.py` — the mechanism: subject-grouped, recency/trust-weighted retrieval with entropy-based confidence. No LLM needed.
- `world.py` — the living tick loop: an NPC forms structured memories (subjects assigned at write-time by an LLM) and its behavior evolves.
- `agent.py` — the baseline demo: the same situation answered WITH vs WITHOUT memory, to show the memory layer is not decorative.
- `memory_weighting_eval.py` — the closed loop: stale-leakage on the retrieval eval's own traps, embed-only vs weighted.
- `eval_memories.json` — the retrieval eval's corpus, bundled so the loop runs standalone.

## Run it

```sh
pip install -r requirements.txt
cp .env.example .env    # add your OPENROUTER_API_KEY (only world.py and agent.py call an LLM)

python memory_weighting.py         # mechanism demo + measurement (no API needed)
python memory_weighting_eval.py    # closed loop vs the retrieval eval (no API needed)
python world.py                    # the living NPC (needs OPENROUTER_API_KEY)
python agent.py                    # with/without memory (needs OPENROUTER_API_KEY)
```

The stand-in embedder is `all-MiniLM-L6-v2`; a production embedder swaps in at the top of `memory_weighting.py`.

## License

MIT.
