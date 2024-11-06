import string
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from openTSNE import TSNE
import os
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path='games.json'):
    print("Loading dataset from file...")
    return pd.read_json(file_path, orient='index')

class EmbeddingAutoencoder(nn.Module):
    def __init__(self, embedding_dim=384):
        super(EmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Linear(embedding_dim, 256)
        self.decoder = nn.Linear(256, embedding_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

def fine_tune_embeddings(embeddings, num_epochs=5, batch_size=32):
    model = EmbeddingAutoencoder(embedding_dim=embeddings.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    embeddings = embeddings.clone().detach().requires_grad_(False)
    dataset = torch.utils.data.TensorDataset(embeddings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Fine-tuning embeddings...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    with torch.no_grad():
        fine_tuned_embeddings = model.encoder(embeddings).detach()
    return fine_tuned_embeddings

def run_tsne(embeddings, cache_file='finetuned_tsne_cache.npy'):
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
        embeddings_2d = tsne.fit(embeddings_np)
        np.save(cache_file, embeddings_2d)
        print(f"TSNE results cached in {cache_file}")
    return embeddings_2d

def visualize_embeddings(embeddings_2d, labels=None, output_file='finetuned_embedding_visualization.png'):
    print("Generating plot...")
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)

    if labels is not None:
        processed_labels = [", ".join(label) if isinstance(label, list) else label for label in labels]
        unique_labels = list(set(processed_labels))
        colors = plt.cm.jet([i / len(unique_labels) for i in range(len(unique_labels))])
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        with tqdm(total=len(embeddings_2d), desc="Plotting points") as pbar:
            for i, label in tqdm(enumerate(processed_labels[:len(embeddings_2d)]), total=len(embeddings_2d), desc="Plotting points"):
                plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color_map[label], s=5, alpha=0.7)
                pbar.update(1)
        plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels, title="Genres")

    plt.title("openTSNE Visualization of Fine-tuned Game Embeddings")
    plt.xlabel("openTSNE Dimension 1")
    plt.ylabel("openTSNE Dimension 2")
    
    print(f"Saving figure to {output_file}")
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = load_data('games.json')
    descriptions = [preprocess_text(desc) for desc in df['detailed_description'].fillna("").tolist()]

    if os.path.exists('finetuned_embeddings.pt'):
        print("Loading existing fine-tuned embeddings...")
        embeddings = torch.load('finetuned_embeddings.pt')
    else:
        print("Generating embeddings and fine-tuning...")
        with torch.no_grad():
            embeddings = torch.stack([model.encode(desc, convert_to_tensor=True) for desc in tqdm(descriptions, desc="Generating embeddings")])
        embeddings = fine_tune_embeddings(embeddings)
        torch.save(embeddings, 'finetuned_embeddings.pt')
    
    labels = df.get('genres').fillna("Unknown").tolist()
    embeddings_2d = run_tsne(embeddings, 'finetuned_tsne_cache.npy')
    visualize_embeddings(embeddings_2d, labels)
