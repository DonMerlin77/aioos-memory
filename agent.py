"""
AIOOS agent loop, stitched end to end:
  situation -> encode -> retrieve relevant memories -> decide (LLM via OpenRouter) -> respond
The test compares the SAME situation WITH vs WITHOUT memory, to prove the memory layer actually
changes what the NPC does (if it doesn't, the whole memory stack is decorative).

  python3 agent.py                      # runs the with/without-memory demo
  MODEL=... python3 agent.py            # swap the decision model
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import chat, DEFAULT_MODEL

HERE = Path(__file__).parent
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"       # stand-in; a production embedder swaps in here
K = 3

NPC_SYS = ("You are Bran, the keeper of a small-town tavern. Stay in character. Given a situation and any "
           "memories you have, respond in 2-3 sentences: what you say or do, grounded in what you actually "
           "remember. If a memory is relevant, use it. Do not invent facts you were not given.")

_model = None
def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDER)
    return _model

def load_memories():
    d = json.load(open(HERE / "eval_memories.json"))
    return d["memories"]

def retrieve(situation, memories, k=K):
    m = _embedder()
    mvecs = m.encode([x["text"] for x in memories], normalize_embeddings=True)
    qvec = m.encode([situation], normalize_embeddings=True)[0]
    order = np.argsort(-(np.asarray(mvecs) @ qvec))
    return [memories[i] for i in order[:k]]

def decide(situation, memories, use_memory=True):
    if use_memory:
        recalled = retrieve(situation, memories)
        mem_block = "\n".join(f"- {x['text']}" for x in recalled)
        user = f"SITUATION:\n{situation}\n\nYOUR MEMORIES:\n{mem_block}\n\nWhat do you say or do?"
    else:
        recalled = []
        user = f"SITUATION:\n{situation}\n\n(You have no particular memories to draw on.)\n\nWhat do you say or do?"
    resp = chat(NPC_SYS, user, max_tokens=200, temperature=0.4)
    return resp, recalled

def main():
    memories = load_memories()
    situations = [
        "A traveler asks you the safest route to take when leaving town heading north.",
        "A stranger asks if anything dangerous has been happening near the eastern mountains lately.",
    ]
    print(f"decision model: {DEFAULT_MODEL}\n" + "=" * 78)
    for s in situations:
        print(f"\nSITUATION: {s}")
        with_resp, recalled = decide(s, memories, use_memory=True)
        without_resp, _ = decide(s, memories, use_memory=False)
        print(f"\n  memories recalled: {[x['id'] for x in recalled]}")
        print(f"\n  WITH memory:\n    {with_resp.strip()}")
        print(f"\n  WITHOUT memory:\n    {without_resp.strip()}")
        print("=" * 78)

if __name__ == "__main__":
    main()
