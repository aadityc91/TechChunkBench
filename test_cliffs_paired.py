import pandas as pd
import numpy as np
from src.stats import cliffs_delta_paired

def test_cliffs_paired():
    print("--- Testing Paired Cliff's Delta ---")
    data = []
    blocks = [("c1", 256, "m1"), ("c1", 512, "m1"), ("c2", 256, "m1")]
    
    # S1 is always slightly better than S2
    for b in blocks:
        data.append({"corpus_id": b[0], "chunk_size": b[1], "embedding_model": b[2], "strategy": "S1", "mrr": 0.8})
        data.append({"corpus_id": b[0], "chunk_size": b[1], "embedding_model": b[2], "strategy": "S2", "mrr": 0.7})
        
    df = pd.DataFrame(data)
    delta, interp = cliffs_delta_paired(df, "S1", "S2", "mrr")
    print(f"Paired Delta: {delta}, Interpretation: {interp}")

    # What if dates mismatch or there are missing values?
    data.append({"corpus_id": "c3", "chunk_size": 256, "embedding_model": "m1", "strategy": "S1", "mrr": 0.9})
    df2 = pd.DataFrame(data)
    delta2, interp2 = cliffs_delta_paired(df2, "S1", "S2", "mrr")
    print(f"Paired Delta (with unmatched S1): {delta2}")

if __name__ == "__main__":
    test_cliffs_paired()
