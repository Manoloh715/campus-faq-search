import math

import pandas as pd

from preprocess import tokenize


df = pd.read_csv("data/fake_faqs.csv")


documents = []

for _, row in df.iterrows():
    document = f"{row['question']} {row['answer']}"
    documents.append(document)

tokenized_documents = [tokenize(document) for document in documents]

num_documents = len(tokenized_documents)


# count how many docs each word appears in, then turn that into idf
document_frequency = {}

for i in range(num_documents):
    unique_words = set(tokenized_documents[i])

    for word in unique_words:
        if word in document_frequency:
            document_frequency[word] = document_frequency[word] + 1
        else:
            document_frequency[word] = 1

idf = {}

for word in document_frequency:
    idf[word] = math.log((1 + num_documents) / (1 + document_frequency[word])) + 1


def get_tfidf_vector(tokens):
    word_counts = {}

    for word in tokens:
        if word in word_counts:
            word_counts[word] = word_counts[word] + 1
        else:
            word_counts[word] = 1

    vector = {}

    for word in word_counts:
        if word in idf:
            tf = word_counts[word] / len(tokens)
            vector[word] = tf * idf[word]

    return vector


def cosine_similarity(vector_a, vector_b):
    dot_product = 0

    for word in vector_a:
        if word in vector_b:
            dot_product = dot_product + vector_a[word] * vector_b[word]

    length_a = 0
    for word in vector_a:
        length_a = length_a + vector_a[word] ** 2
    length_a = math.sqrt(length_a)

    length_b = 0
    for word in vector_b:
        length_b = length_b + vector_b[word] ** 2
    length_b = math.sqrt(length_b)

    if length_a == 0 or length_b == 0:
        return 0

    return dot_product / (length_a * length_b)


document_vectors = []

for i in range(num_documents):
    document_vectors.append(get_tfidf_vector(tokenized_documents[i]))


def search(query, top_k=5):
    tokenized_query = tokenize(query)

    query_vector = get_tfidf_vector(tokenized_query)

    scores = []

    for i in range(num_documents):
        score = cosine_similarity(query_vector, document_vectors[i])
        scores.append(score)

    results = df.copy()
    results["score"] = scores

    results = results[results["score"] > 0]

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

        if len(results) == 0:
            print("\nNo matching FAQs found.")
            continue

        print("\nTop Results:\n")

        for rank, (_, row) in enumerate(results.iterrows(), start=1):

            print(f"{rank}. {row['question']}")
            print(f"Category: {row['category']}")
            print(f"Answer: {row['answer']}")
            print(f"TF-IDF Score: {row['score']:.4f}")
            print()
