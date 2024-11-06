# ------ imports ------

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import json
import os
import subprocess
import zipfile
from sentence_transformers import SentenceTransformer
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ------ function to download dataset from Kaggle ------

def download_kaggle_dataset():
    print("Downloading the dataset...")
    try:
        kaggle_command = "kaggle"
        subprocess.run([kaggle_command, "--version"], check=True)
        subprocess.run(
            [kaggle_command, "datasets", "download", "-d", "fronkongames/steam-games-dataset"],
            check=True
        )
        with zipfile.ZipFile('steam-games-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall()
        os.remove('steam-games-dataset.zip')
        os.remove('games.csv')
    except subprocess.CalledProcessError as e:
        print("Failed to download the dataset:", e)
        exit(1)

# ------ load dataset ------

if not os.path.exists('games.json'):
    download_kaggle_dataset()
with open('games.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient='index')

# ------ load or create embeddings with batch processing ------

model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists('embeddings.pt'):
    print("Loading existing embeddings...")
    embeddings = torch.load('embeddings.pt')
else:
    print("Generating new embeddings with batch processing...")
    descriptions = df['detailed_description'].fillna("").tolist()
    
    batch_size = 32
    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
        print(f"Processed batch {i // batch_size + 1} of {len(descriptions) // batch_size + 1}")
    
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, 'embeddings.pt')
    print("Embeddings generated and saved.")

# ------ app section ------
app = FastAPI()

class QueryResponse(BaseModel):
    name: str
    price: float
    release_date: str
    developers: str
    genres: str
    detailed_description: str
    relevance: float

# ------ queries ------

@app.get("/query", response_model=dict)
async def query(query: str = Query(..., description="Keywords to search for recommendations")):
    if not query:
        raise HTTPException(status_code=400, detail="Please type a query parameter.")

    query_embedding = model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy(), embeddings.cpu().numpy()).flatten()

    indices = similarities.argsort()[-10:][::-1]
    indices = [i for i in indices if similarities[i] > 0]

    results = []

    for i in indices:
        results.append({
            'name': df.iloc[i]['name'],
            'price': df.iloc[i]['price'],
            'release_date': df.iloc[i]['release_date'],
            'developers': df.iloc[i].get('developers', ''),
            'genres': df.iloc[i].get('genres', ''),
            'detailed_description': df.iloc[i]['detailed_description'],
            'relevance': float(similarities[i])
        })
    
    return {"results": results, "message": "OK"}

# ------ main ------

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=2909, log_level="info")
