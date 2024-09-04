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

- A query that return something non-obvious:

```
http://10.103.0.28:2909/query?query=free
```

While most results for searching up "free" do show games that do not cost any money, the term is common enough to be featured in several other descriptions, including games that are not free-to-play.
