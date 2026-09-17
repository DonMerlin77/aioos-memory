"""
Memory weighting: resolving conflicting / stale memories.

The retrieval eval (github.com/DonMerlin77/aioos-retrieval-eval) proved embedding similarity is
TIME-BLIND: "the bridge is washed out" and "the bridge was rebuilt" are both similar to "is the road
passable?", and the embedder cannot tell which is CURRENT (it surfaced the stale memory ~20% of the
time on the eval's traps). That is a MEMORY-LAYER job, not a retrieval one.

Detection needs STRUCTURE, not similarity. Two memories that contradict each other about one subject
can be embedding-DISSIMILAR ("bandits at dusk" vs "perfectly safe"), so cosine cannot group them.
Instead each memory carries the SUBJECT it is about; memories are grouped by subject, and within a
group a newer / more-trusted memory suppresses the stale one it contradicts.

Confidence is ENTROPY-based. Within a subject group, put the belief mass (recency+trust) into a
distribution and take its entropy: one dominant memory => low entropy => confident; two conflicting
memories of similar standing => high entropy => commit with doubt. Entropy does not DETECT the
conflict (the tags do that); it MEASURES how contested the subject is, which is the right calibration.

Two guardrails against the obvious failure modes:
  - NOT supersede-by-recency-alone: a low-trust newer memory does not silently override a high-trust
    older one; trust enters the authority, and a contested subject yields low confidence, not a blind flip.
  - NOT hold-flat: it commits to a best answer and a confidence rather than refusing to choose.

  python3 memory_weighting.py     # demo + measurement vs embed-only
"""
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
W_RECENCY = 0.6          # recency is the stronger prior...
W_TRUST = 0.4            # ...but trust matters and can override it
SUPPRESS = 0.25          # down-weight factor for a memory a higher-authority same-subject rival contradicts
TEMP = 0.2               # softmax temperature turning authority gaps into belief mass for the entropy


@dataclass
class Memory:
    id: str
    text: str
    subject: str         # what this memory is ABOUT (assigned at write-time); the grouping key
    learned_at: int      # tick / day; higher = more recent
    trust: float = 0.7   # source reliability, 0..1


@dataclass
class Commitment:
    memory: Memory       # the memory the NPC should act on
    confidence: float    # 0..1, from the entropy of belief mass within the subject group
    doubt: str           # "" if none; a note when the answer is contested or overrode the top-similar memory


class MemoryStore:
    def __init__(self, model=None):
        self.model = model or SentenceTransformer(MODEL)
        self.mems = []
        self._vecs = None

    def add(self, m):
        self.mems.append(m)
        self._vecs = None

    def _ensure(self):
        if self._vecs is None:
            self._vecs = np.asarray(self.model.encode([m.text for m in self.mems], normalize_embeddings=True))

    def _authority(self):
        """Belief weight per memory: recency (rank-normalized 0..1) and trust. Deliberately separate
        from query similarity ('on-topic' is not the same as 'to be believed')."""
        ticks = np.array([m.learned_at for m in self.mems], float)
        rec = (ticks - ticks.min()) / (ticks.max() - ticks.min()) if ticks.max() > ticks.min() else np.ones_like(ticks)
        trust = np.array([m.trust for m in self.mems])
        return W_RECENCY * rec + W_TRUST * trust

    def _group(self, i):
        return [j for j, m in enumerate(self.mems) if m.subject == self.mems[i].subject]

    def rank(self, query, weighted=True):
        self._ensure()
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        sim = self._vecs @ qv
        if not weighted:
            return list(np.argsort(-sim)), sim, {}
        auth = self._authority()
        final = sim.copy()
        beaten_by = {}
        for i in range(len(self.mems)):
            rivals = [j for j in self._group(i) if j != i and auth[j] > auth[i]]
            if rivals:
                final[i] *= SUPPRESS
                beaten_by[i] = max(rivals, key=lambda j: auth[j])
        return list(np.argsort(-final)), sim, beaten_by

    def _entropy_confidence(self, group):
        """Confidence = 1 - normalized entropy of the belief mass across the subject group."""
        if len(group) < 2:
            return 0.75, 0.0
        a = self._authority()[group]
        z = np.exp(a / TEMP)
        p = z / z.sum()
        H = float(-(p * np.log(p + 1e-12)).sum())
        contested = H / np.log(len(group))
        return float(1 - contested), float(contested)

    def commit(self, query):
        ranked, sim, beaten_by = self.rank(query, weighted=True)
        top = ranked[0]
        group = self._group(top)
        confidence, contested = self._entropy_confidence(group)
        emb_top = int(np.argmax(sim))
        doubt = ""
        if emb_top in beaten_by and beaten_by[emb_top] == top:
            doubt = (f"the most on-topic memory ('{self.mems[emb_top].text[:38]}...') is lower-authority "
                     f"(older and/or less trusted); going with the higher-authority one")
        elif contested > 0.6:
            doubt = "this subject is contested (memories of similar standing disagree)"
        return Commitment(self.mems[top], confidence, doubt)


# --------------------------------------------------------------------------------------
# Demo + honest measurement
# --------------------------------------------------------------------------------------

def _build_world():
    s = MemoryStore()
    s.add(Memory("bridge_out",   "The bridge on the north road is washed out.",                  "north_bridge", 1, 0.9))
    s.add(Memory("bridge_ok",    "The north bridge was rebuilt and the road is open again.",      "north_bridge", 6, 0.9))
    s.add(Memory("well_sick",    "The well water made several villagers sick.",                   "town_well",    2, 0.85))
    s.add(Memory("well_clean",   "The well was cleaned and the water is safe again.",             "town_well",    7, 0.85))
    s.add(Memory("pass_blocked", "A rockslide blocked the western mountain pass.",                "west_pass",    1, 0.8))
    s.add(Memory("pass_clear",   "A courier uses the western pass daily now that it is cleared.",  "west_pass",    6, 0.8))
    # stance-conflict: a NEWER but LOW-TRUST memory that must not silently erase an older trusted warning
    s.add(Memory("mira_warn",    "Mira the blacksmith's daughter warned the east road has bandits at dusk.", "east_road", 2, 0.95))
    s.add(Memory("peddler_safe", "A peddler swears the east road is perfectly safe and quick.",   "east_road",    4, 0.2))
    return s

# queries phrased to FAVOR the stale memory's wording (where embed-only gets fooled), with the current-correct id
STALE_QUERIES = [
    ("Is the north bridge washed out?",            "bridge_ok"),
    ("Did the well water make people sick?",       "well_clean"),
    ("Is the western pass blocked by a rockslide?", "pass_clear"),
]


def main():
    s = _build_world()
    ids = [m.id for m in s.mems]

    print("== stale-vs-current (queries phrased to fool the embedder) ==")
    emb_hits = wtd_hits = 0
    for q, current in STALE_QUERIES:
        emb, _, _ = s.rank(q, weighted=False)
        wtd, _, _ = s.rank(q, weighted=True)
        emb_hits += ids[emb[0]] == current
        wtd_hits += ids[wtd[0]] == current
        c = s.commit(q)
        print(f"  {q}")
        print(f"    embed-only top: {ids[emb[0]]:<12} | weighted top: {ids[wtd[0]]:<12} | want: {current}")
        print(f"    commit: '{c.memory.text[:44]}'  conf={c.confidence:.2f}" + (f"  DOUBT: {c.doubt}" if c.doubt else ""))
    n = len(STALE_QUERIES)
    print(f"\n  current-memory@1:  embed-only {emb_hits}/{n}   weighted {wtd_hits}/{n}")

    print("\n== stance-conflict guardrail: newer LOW-trust must not blindly override a trusted warning ==")
    c = s.commit("Are there bandits on the east road?")
    print(f"  Are there bandits on the east road?")
    print(f"    commit: '{c.memory.text[:52]}'  conf={c.confidence:.2f}")
    print(f"    DOUBT: {c.doubt}" if c.doubt else "    (no doubt flagged)")
    print(f"    (mira trust=0.95@2 beats peddler trust=0.2@4 on authority; entropy => low confidence, contested)")


if __name__ == "__main__":
    main()
