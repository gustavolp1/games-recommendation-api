# ------ imports ------

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import json
import os
import subprocess
import zipfile

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

# ------ csv and mild cleanup ------

if not os.path.exists('games.json'):
    download_kaggle_dataset()
with open('games.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient='index')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['detailed_description'])

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
    
    query_processed = query.lower().strip()
    query_vector = vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vector, X).flatten()
    
    # only get top 10 results
    indices = similarities.argsort()[-10:][::-1]
    indices = [i for i in indices if similarities[i] > 0]

    results = []

    for i in indices:
        results.append({
            'name': df.iloc[i]['name'],
            'price': df.iloc[i]['price'],
            'release_date': df.iloc[i]['release_date'],
            'developers': df.iloc[i]['developers'],
            'genres': df.iloc[i]['genres'],
            'detailed_description': df.iloc[i]['detailed_description'],
            'relevance': float(similarities[i])
        })
    
    return {"results": results, "message": "OK"}

# ------ main ------

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=2909)
