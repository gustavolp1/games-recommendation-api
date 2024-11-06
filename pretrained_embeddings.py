import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from openTSNE import TSNE
import os

def load_embeddings(file_path='pretrained_embeddings.pt'):
    print("Loading embeddings from file...")
    return torch.load(file_path)

def load_data(file_path='games.json'):
    print("Loading dataset from file...")
    return pd.read_json(file_path, orient='index')

def preprocess_labels(labels):
    processed_labels = []
    for label in labels:
        if isinstance(label, list):
            processed_labels.append(", ".join(label))
        else:
            processed_labels.append(label)
    return processed_labels

def run_tsne(embeddings, cache_file='tsne_cache.npy'):
    if os.path.exists(cache_file):
        print("Loading cached TSNE results...")
        embeddings_2d = np.load(cache_file)
    else:
        print("Running openTSNE to reduce embeddings to 2D...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            n_jobs=-1,
            verbose=True
        )
        embeddings_np = embeddings.cpu().numpy()
        with tqdm(total=1, desc="openTSNE Progress", unit="step") as pbar:
            embeddings_2d = tsne.fit(embeddings_np)
            pbar.update(1)
        np.save(cache_file, embeddings_2d)
        print(f"TSNE results cached in {cache_file}")
    return embeddings_2d

def visualize_embeddings(embeddings_2d, labels=None, output_file='pretrained_embedding_visualization.png'):
    print("Generating plot with progress tracking...")
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)

    if labels is not None:
        processed_labels = preprocess_labels(labels)
        unique_labels = list(set(processed_labels))
        colors = plt.cm.jet([i / len(unique_labels) for i in range(len(unique_labels))])
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        with tqdm(total=len(embeddings_2d), desc="Plotting points") as pbar:
            for i, label in enumerate(processed_labels[:len(embeddings_2d)]):
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color_map[label], s=5, alpha=0.7)
                pbar.update(1)
        plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels, title="Genres")

    plt.title("openTSNE Visualization of Game Embeddings")
    plt.xlabel("openTSNE Dimension 1")
    plt.ylabel("openTSNE Dimension 2")
    
    print(f"Saving figure to {output_file}")
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    embeddings = load_embeddings('pretrained_embeddings.pt')
    df = load_data('games.json')
    labels = df.get('genres').fillna("Unknown").tolist()
    embeddings_2d = run_tsne(embeddings, 'pretrained_tsne_cache.npy')
    visualize_embeddings(embeddings_2d, labels)
