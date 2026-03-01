import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare

def test_friedman_blocks():
    print("--- Testing Friedman Block Design ---")
    np.random.seed(42)
    # 24 corpus, 3 sizes, 3 models = 216 blocks
    data = []
    strategies = ["A", "B", "C"]
    for c in range(24):
        for s in [256, 512, 1024]:
            for m in ["m1", "m2", "m3"]:
                base_perf = np.random.rand()
                for strat in strategies:
                    # Strategy A is slightly better only on m1
                    perf = base_perf + (0.1 if strat == "A" and m == "m1" else 0)
                    data.append({
                        "corpus_id": f"c{c}",
                        "chunk_size": s,
                        "model": m,
                        "strategy": strat,
                        "mrr": perf + np.random.normal(0, 0.05)
                    })
    df = pd.DataFrame(data)
    
    # Existing implementation (averaging over size and model)
    pivot_wrong = df.pivot_table(index="corpus_id", columns="strategy", values="mrr", aggfunc="mean")
    stat1, p1 = friedmanchisquare(*[pivot_wrong[c].values for c in pivot_wrong.columns])
    print(f"Wrong (N={len(pivot_wrong)}): p={p1:.4f}")
    
    # Correct implementation (treating each configuration as a block)
    df["block"] = df["corpus_id"] + "_" + df["chunk_size"].astype(str) + "_" + df["model"]
    pivot_correct = df.pivot_table(index="block", columns="strategy", values="mrr")
    stat2, p2 = friedmanchisquare(*[pivot_correct[c].values for c in pivot_correct.columns])
    print(f"Correct (N={len(pivot_correct)}): p={p2:.4f}")

if __name__ == "__main__":
    test_friedman_blocks()
