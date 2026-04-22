The code supports linear and shallow nonlinear probing, permutation tests, and visualization of embedding geometry.

## Contents

supplementary/
├── README.md
├── 1_get_sentences_embeddings.py
├── 2_UMAP_visualization.py
├── 3_probing_and_confusion_matrix.py
├── 4_permutation_test.py
├── masked_dataset_with_tiers_scores.csv
├── sentence_embeddings_BAAI_bge_large_en_v1_5.npy
├── sentence_embeddings_sentence_transformers_all_MiniLM_L6_v2.npy
└── sentence_embeddings_sentence_transformers_all_mpnet_base_v2.npy

## Data and privacy
The original sentence text is masked to avoid redistribution of sensitive content. The provided CSV file contains only tier labels and continuous energy scores. No raw sentence text is included.

## Embeddings
Precomputed sentence embeddings are provided as `.npy` files. These embeddings are used directly in all experiments reported in the paper. As a result, running `1_get_sentences_embeddings.py` is **not required** for reproducing the main results.

The following embedding models are included:
- BAAI/bge-large-en-v1.5
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2

## Environment
All experiments were run using Python 3.10+. The code depends on standard scientific Python libraries, including NumPy and pandas for data handling; sentence-transformers for sentence embedding generation; scikit-learn for regression, classification, and evaluation metrics; umap-learn for dimensionality reduction; and matplotlib and seaborn for visualization.

## Reproducing the experiments
python 2_UMAP_visualization.py
python 3_probing_and_confusion_matrix.py
python 4_permutation_test.py

## License
This supplementary material is provided for research and reproducibility purposes. Please cite the accompanying paper if you use or adapt this code.
