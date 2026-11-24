import re


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "am",
    "i", "me", "my", "you", "your", "we", "our", "it", "its",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "how", "what", "where", "when", "who", "which", "why",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "as",
    "and", "or", "if", "so", "that", "this", "there", "any", "some",
    "get", "need", "have", "has",
}


# simple stemming so things like library/libraries and class/classes match
def stem(word):
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]

    return word


def tokenize(text):
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)

    tokens = []

    for word in words:
        if word not in STOPWORDS:
            tokens.append(stem(word))

    return tokens
