'''
Don't run this code, if the embeddings are already provided in .npy format.
For example:
    "BAAI_bge_large_en_v1_5": "sentence_embeddings_BAAI_bge_large_en_v1_5.npy",
    "all-mpnet-base-v2": "sentence_embeddings_sentence_transformers_all_mpnet_base_v2.npy",
    "all-MiniLM-L6-v2": "sentence_embeddings_sentence_transformers_all_MiniLM_L6_v2.npy",
'''

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def get_embeddings(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

def main():
    csv_path = "full_dataset_with_5scores.csv"
    df = pd.read_csv(csv_path)
    print(df.shape)

    # get embeddings
    X = get_embeddings(df['sentence'])
    print(X.shape)
    print(X[:5])

    # save embeddings
    np.save("sentence_embeddings.npy", X)

if __name__ == "__main__":
    main()