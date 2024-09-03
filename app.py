# ------ imports ------

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# ------ csv and mild cleanup ------

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# clean up nan and null values
df['Genre'] = df['Genre'].fillna('').str.lower().str.strip()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Genre'])

# ------ app section ------
app = FastAPI()

class QueryResponse(BaseModel):
    title: str
    platform: str
    year_of_release: int
    genre: str
    publisher: str
    global_sales: float
    relevance: float

# ------ queries ------

@app.get("/query", response_model=dict)
async def query(query_text: str = Query(..., description="Genre to search for recommendations")):
    if not query_text:
        raise HTTPException(status_code=400, detail="Query parameter is missing")
    
    query_text_processed = query_text.lower().strip()
    query_vec = vectorizer.transform([query_text_processed])
    
    similarities = cosine_similarity(query_vec, X).flatten()
    
    # only get top 10 results
    indices = similarities.argsort()[-10:][::-1]
    indices = [i for i in indices if similarities[i] > 0]

    results = []
    for i in indices:
        results.append({
            'title': df.iloc[i]['Name'],
            'platform': df.iloc[i]['Platform'],
            'year_of_release': int(df.iloc[i]['Year_of_Release']) if pd.notnull(df.iloc[i]['Year_of_Release']) else None,
            'genre': df.iloc[i]['Genre'],
            'publisher': df.iloc[i]['Publisher'],
            'global_sales': df.iloc[i]['Global_Sales'],
            'relevance': float(similarities[i])
        })
    
    return {"results": results, "message": "OK"}

# ------ main ------

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=2909)
