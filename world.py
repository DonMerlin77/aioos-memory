"""
AIOOS living world: a tick loop where the NPC FORMS memories from events and its behavior EVOLVES.

Memories are written with STRUCTURE, not just text: at write-time the NPC assigns each memory a SUBJECT
(the place/person/thing it is about), a timestamp, and a trust. Retrieval then uses the memory-weighting
layer (memory_weighting.py): embedding finds candidates, subject-grouping + recency/trust resolves stale
vs current, and entropy sets a confidence. This is why Bran can be handed "bridge collapsed" and "bridge
repaired" and answer with the CURRENT one, which pure embedding similarity cannot do (it is time-blind).

The test: the SAME "route north" question is asked before and after the bridge is repaired. If memory
works, the two answers DIFFER, and the recalled set for the second one no longer leads with the stale
collapse memory.

  python3 world.py
"""
import re
from llm import chat, DEFAULT_MODEL
from memory_weighting import MemoryStore, Memory

NPC_SYS = ("You are Bran, keeper of a small-town tavern. Stay in character. Answer in 2-3 sentences, grounded "
           "ONLY in the memories you are given. They are listed most-current first; trust the current one when "
           "memories conflict. Do not invent facts you were not told.")
SUBJECT_SYS = ("Extract the enduring SUBJECT a memory is about: the place, person, or thing it concerns, NOT the "
               "event or its changing status. E.g. 'the north bridge collapsed' and 'the north bridge was repaired' "
               "share the subject 'north_bridge'. Reply with ONLY a 1-3 word snake_case tag.")


def extract_subject(text):
    out = chat(SUBJECT_SYS, f"MEMORY: {text}\nSubject tag:", max_tokens=16, temperature=0.0) or "unknown"
    tag = re.sub(r"[^a-z0-9_ ]", "", out.strip().lower()).strip()
    return re.sub(r"\s+", "_", tag).strip("_")[:40] or "unknown"


class NPC:
    def __init__(self):
        self.store = MemoryStore()
        self.tick = 0
        self._subjects = []   # (canonical_subject, vec) to keep the LLM's tags from fragmenting

    def _canon(self, subject):
        """Reuse a near-identical existing subject so 'north_bridge' and 'bridge_north' don't split a group."""
        v = self.store.model.encode([subject.replace("_", " ")], normalize_embeddings=True)[0]
        for s, sv in self._subjects:
            if float(sv @ v) > 0.7:
                return s
        self._subjects.append((subject, v))
        return subject

    def observe(self, text, trust=0.8):            # MEMORY WRITING: event -> structured memory
        self.tick += 1
        subject = self._canon(extract_subject(text))
        mid = f"m{len(self.store.mems) + 1}"
        self.store.add(Memory(mid, text, subject, self.tick, trust))
        return mid, subject

    def decide(self, situation, k=3):
        ranked, _, _ = self.store.rank(situation, weighted=True)
        recalled = [self.store.mems[i] for i in ranked[:k]]
        commit = self.store.commit(situation)
        block = "\n".join(f"- {m.text}" for m in recalled) or "(you recall nothing relevant)"
        user = f"SITUATION:\n{situation}\n\nWHAT YOU REMEMBER (most current first):\n{block}\n\nWhat do you say?"
        resp = chat(NPC_SYS, user, max_tokens=180, temperature=0.4)
        return resp.strip(), [m.id for m in recalled], commit


TIMELINE = [
    ("event", "You overhear two guards saying the north bridge collapsed in last night's storm."),
    ("event", "A merchant paid with a coin bearing the royal seal, then left in a hurry."),
    ("situation", "A traveler asks you the safest route to take heading north."),
    ("event", "Word arrives that the north bridge has been fully repaired and the road is open again."),
    ("event", "The mayor announces the harvest festival is moved to next week."),
    ("situation", "Another traveler asks you the safest route to take heading north."),
]


def main():
    bran = NPC()
    print(f"decision model: {DEFAULT_MODEL}\n" + "=" * 78)
    for kind, text in TIMELINE:
        if kind == "event":
            mid, subject = bran.observe(text)
            print(f"\n[EVENT -> {mid}, subject='{subject}', memories={len(bran.store.mems)}] {text}")
        else:
            resp, recalled, c = bran.decide(text)
            print(f"\n[SITUATION] {text}")
            print(f"   recalled (weighted): {recalled}   confidence={c.confidence:.2f}")
            if c.doubt:
                print(f"   doubt: {c.doubt}")
            print(f"   Bran: {resp}")
    print("\n" + "=" * 78)
    print("THE TEST: the two 'route north' answers should DIFFER. After the repair event, the weighting layer\n"
          "suppresses the stale 'collapsed' memory (same subject, older) so Bran leads with 'road is open'.")


if __name__ == "__main__":
    main()
