import numpy as np

def _fuzzy_match(chunk, ev): return True

def compute_ndcg(retrieved, evidence, total_relevant=1):
    dcg = 0.0
    for i, text in enumerate(retrieved):
        if _fuzzy_match(text, evidence):
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(retrieved), total_relevant)))
    return dcg / idcg if idcg > 0 else 0.0

retrieved = ["hit1", "hit2", "hit3"]
print("NDCG:", compute_ndcg(retrieved, "ev", 1))

