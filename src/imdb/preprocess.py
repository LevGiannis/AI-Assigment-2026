import re

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def preprocess_texts(texts):
    return [simple_tokenize(t) for t in texts]
