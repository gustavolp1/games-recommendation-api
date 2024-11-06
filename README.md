# Steam Game Recommendation System

This project aims to create a recommendation system for games available on Steam. It uses FastAPI, vectorizes and processes data to recommend games based on user-written queries.

The system may be used for finding good deals, searching for games by similar characteristics, comparing prices, and more.

All data used is taken from the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?select=games.json) by Martin Bustos on Kaggle.

## Installation

Assuming you have Python installed:

1. On a command prompt or Powershell, clone the repository:

```
git clone https://github.com/gustavolp1/games-recommendation-api
```

2. Switch the directory to the newly-cloned repository:

```
cd games-recommendation-api
```

3. (Recommended) Create a virtual environment and activate it:

```
python -m venv env
.\env\Scripts\Activate.ps1
```

4. Install the required dependencies:

```
pip install -r "requirements.txt"
```

## Running and usage

Run `app.py`. You may do so through the command:

```
python app.py
```

This will initialize the server and automatically download a json with the data, if it has not been previously installed.

To send a query, use the following route (or the route specified on the terminal output):

```
http://10.103.0.28:2909/query?query=yourqueryhere
```

Where "yourqueryhere" should be replaced with a keyword related to your search.

This should output a json object with the following format:

```
{
  "results": [
    {
      "name": "A Cool Video Game",
      "price": 9.99,
      "release_date": "Jan 1, 2020",
      "developers": [
        "Bintendo"
      ],
      "genres": [
        "Adventure",
        "Action"
      ],
      "detailed_description": "This is a description of a video game.",
      "relevance": 0.235433
    },
    // more results, if the query yielded more than one
  ],
  "message": "OK"
}

```

A maximum of 10 results (the most relevant) will be received at a time, with a relevance value appended to it.

## Query examples

- A query that yields more than 10 results:

```
http://10.103.0.28:2909/query?query=action
```

Due to how popular action-oriented games are, alongside the word being frequently used on descriptions, it should show multiple results.

- A query that yields less than 10 results:

```
http://10.103.0.28:2909/query?query=relevancy
```

Not many games use the word "relevancy" in their descriptions or names. The word is too specific and unrelated to games in general, in turn generating less results.

- A query that returns something non-obvious:

```
http://10.103.0.28:2909/query?query=free
```

While most results for searching up "free" do show games that do not cost any money, the term is common enough to be featured in several other descriptions, including games that are not free-to-play.

## Dataset Description

The dataset used for this project was the Steam Games Dataset by Martin Bustos. It contains web-scraped information on over 97 thousand games from the online game store Steam. Each entry contains information such as the game's name, price in American dollars, release date, developers, genres, descriptions and more.

## Embedding Generation

The embedding generation uses the all-MiniLM-L6-v2 model from SentenceTransformer. It is an efficient variant of the BERT architecture. This model processes the text into a 384-dimensional vector.

The high-dimensional space enables comparisons between games from cosine similarity. A batch size of 32 was used for memory management, due to the dataset's size. Games were classified based on their descriptions.

![1730829141424](image/README/1730829141424.png)

## Training Process

This model is trained with a Contrastive Loss function. It minimizes the distance between embeddings of similar pairs, while maximizing the distance between less similar ones. This is relevant for the task due to its ability to capture semantic similarity, which is useful for finding related games.

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \left[ y_{i} \cdot \text{d}(x_{i}^{1}, x_{i}^{2}) + (1 - y_{i}) \cdot \max(0, \text{margin} - \text{d}(x_{i}^{1}, x_{i}^{2})) \right]
$$

Where:

- y_i is 1 for similar pairs, 0 for dissimilar pairs.
- d(x_i^1, x_i^2) is the distance between embeddings.
- *margin* is the separation for dissimilar pairs.

## Visualizing Embeddings

OpenTSNE was run to project the embeddings on to a three-dimensional space. One was done on a pre-trained model, and the other on a fine-tuned model. This can be replicated by running `pretrained_embeddings.py` and `finetuned_embeddings.py` respectively. Below are the achieved results.

### Pre-Trained Embeddings

![1730900002179](image/README/1730900002179.png)

This figure shows embeddings generated with a pre-trained model. Each point represents a game description, with colors indicating genre. Points do not form clear clusters, which suggests this model captures semantic information but without much genre differentiation.

### Fine-Tuned Embeddings

![1730900077552](image/README/1730900077552.png)

This figure shows embeddings generated with the same model but fine-tuned. A slight concentration of points of the same color appears on the right, which suggests minor clustering. However, clusters are still not clear, which indicates that fine-tuning was not too helpful for separating genres.

### Conclusion

Neither version of the model formed distinct clusters. This could reflect limitations of the model, such as struggling to differentiate game genres from description alone. Fine-tuning did help with clustering slightly, though further enhancements are needed to improve this.
