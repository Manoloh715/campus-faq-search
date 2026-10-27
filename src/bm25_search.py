import re

import pandas as pd
from rank_bm25 import BM25Okapi


# Load FAQ data
df = pd.read_csv("data/faqs.csv")


def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


# Combine each FAQ question and answer into one searchable document
documents = []

for _, row in df.iterrows():
    document = f"{row['question']} {row['answer']}"
    documents.append(document)


# Tokenize documents
tokenized_documents = [tokenize(document) for document in documents]


# Build BM25 index
bm25 = BM25Okapi(tokenized_documents)


def search(query, top_k=5):
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    results = df.copy()
    results["score"] = scores

    results = results.sort_values(
        by="score",
        ascending=False
    ).head(top_k)

    return results


if __name__ == "__main__":

    while True:
        query = input("\nEnter a campus question (or type 'quit'): ")

        if query.lower() == "quit":
            break

        results = search(query)

        print("\nTop Results:\n")

        for rank, (_, row) in enumerate(results.iterrows(), start=1):

            print(f"{rank}. {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Answer: {row['answer']}")
            print(f"BM25 Score: {row['score']:.4f}")
            print()