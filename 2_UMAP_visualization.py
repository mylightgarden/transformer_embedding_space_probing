import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_FILES = {
    "BAAI_bge_large_en_v1_5": BASE_DIR / "sentence_embeddings_BAAI_bge_large_en_v1_5.npy",
    "all-mpnet-base-v2": BASE_DIR / "sentence_embeddings_sentence_transformers_all_mpnet_base_v2.npy",
    "all-MiniLM-L6-v2": BASE_DIR / "sentence_embeddings_sentence_transformers_all_MiniLM_L6_v2.npy",
}

METADATA_CSV = BASE_DIR / "masked_dataset_with_tiers_scores.csv"
ENERGY_COLUMN = "score" 

SAVE_IMAGE_2D = True
SAVE_PATH_2D = BASE_DIR / "umap_plot_2d.png"

SAVE_IMAGE_3D = True
SAVE_PATH_3D = BASE_DIR / "umap_plot_3d.png"

# Color Scale: Using Energetic Chakra Color Palette As a Reference:
# -----------------------------------------------------------------
def make_energetic_colormap():
    energetic_colors = [
        "#ff3b3b",  # red
        "#ff8b3d",  # orange
        "#ffd93b",  # yellow
        "#34d399",  # green
        "#3b82f6",  # blue
        "#6366f1",  # indigo
        "#a855f7",  # violet  
    ]
    return LinearSegmentedColormap.from_list("chakra", energetic_colors)
energetic_cmap = make_energetic_colormap()

# Load Data
# -----------------------------------------------------------------
def load_energy_scores(csv_path, column):
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    return df[column].to_numpy()

def load_embeddings(embedding_files):
    loaded = {}
    for name, path in embedding_files.items():
        if not os.path.exists(path):
            print(f"Missing embedding file: {path}")
            continue
        loaded[name] = np.load(path)
    return loaded

# UMAP Reducer
# -----------------------------------------------------------------
def run_umap(embeddings, n_components = 2):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_components=n_components
    )
    return reducer.fit_transform(embeddings)

# Visualization 2D
# -----------------------------------------------------------------
def visualize_umap_2d(embeddings_dict, energies):
    norm = Normalize(vmin=energies.min(), vmax=energies.max())
    n_models = len(embeddings_dict)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(7 * n_models, 6),
        constrained_layout=True
    )
    axes = axes if n_models > 1 else [axes]

    mappable_for_cbar = None

    for ax, (model_name, emb) in zip(axes, embeddings_dict.items()):
        print(f"Running 2D UMAP: {model_name}")
        emb_2d = run_umap(emb, n_components=2)

        sc = ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=energies,
            cmap=energetic_cmap,
            norm=norm,
            s=16,
            alpha=0.95,
        )
        ax.set_title(model_name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        if mappable_for_cbar is None:
            mappable_for_cbar = sc

    # colorbar on the right
    cbar = fig.colorbar(
        mappable_for_cbar,
        ax=axes,
        location="right",
        fraction=0.035,
        pad=0.02,
    )

    cbar.set_label("Energy Score", fontsize=12)

    fig.suptitle("2D UMAP Visualization of Cognitive Levels in Embedding Space", fontsize=16)

    if SAVE_IMAGE_2D:
        fig.savefig(SAVE_PATH_2D, dpi=300, bbox_inches="tight")
        print(f"2D plot saved to: {SAVE_PATH_2D}")

    plt.show(block=False)
    plt.pause(0.1)

# Visualization 3D
# -----------------------------------------------------------------
def visualize_umap_3d(embeddings_dict, energies):
    norm = Normalize(vmin=energies.min(), vmax=energies.max())
    n_models = len(embeddings_dict)

    fig = plt.figure(figsize=(5.5 * n_models, 6.5), constrained_layout=True)
    axes = [fig.add_subplot(1, n_models, i + 1, projection="3d") for i in range(n_models)]
    fig.subplots_adjust(wspace=0.02)

    mappable_for_cbar = None

    for ax, (model_name, emb) in zip(axes, embeddings_dict.items()):
        print(f"Running 3D UMAP: {model_name}")
        emb_3d = run_umap(emb, n_components=3)

        sc = ax.scatter(
            emb_3d[:, 0],
            emb_3d[:, 1],
            emb_3d[:, 2],
            c=energies,
            cmap=energetic_cmap,
            norm=norm,
            s=12,
            alpha=0.9,
        )

        ax.set_title(model_name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        if mappable_for_cbar is None:
            mappable_for_cbar = sc

    cbar = fig.colorbar(
        mappable_for_cbar,
        ax=axes,
        location="right",
        fraction=0.035,
        pad=0.02,
    )
    cbar.set_label("Energy Score", fontsize=12)

    fig.suptitle("3D UMAP Visualization of Cognitive Levels in Embedding Space", fontsize=16)

    if SAVE_IMAGE_3D:
        fig.savefig(SAVE_PATH_3D, dpi=300, bbox_inches="tight")
        print(f"3D plot saved to: {SAVE_PATH_3D}")

    plt.show()

# Main
# -----------------------------------------------------------------
def main():
    energies = load_energy_scores(METADATA_CSV, ENERGY_COLUMN)
    embeddings_dict = load_embeddings(EMBEDDING_FILES)

    visualize_umap_2d(embeddings_dict, energies)
    visualize_umap_3d(embeddings_dict, energies)

if __name__ == "__main__":
    main()
