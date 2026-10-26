"""See what embeddings do: similar meanings get vectors that point the same way.

This one really calls Gemini (a few tiny requests), so it needs your key and internet.
"""

import numpy as np

from bloom.llm import build_embedder

sentences = [
    "open a connection to the database",
    "connect to the SQL server",
    "draw a red button on the screen",
]

embedder = build_embedder()
vectors = embedder.embed_documents(sentences)

print("shape:", vectors.shape, "(sentences, numbers per sentence)")
print("length of first vector:", round(float(np.linalg.norm(vectors[0])), 3))
print()
print("similarity (1.0 = same meaning, near 0 = unrelated):")
scores = vectors @ vectors.T  # inner product of length-1 vectors = cosine similarity
for i, row in enumerate(scores):
    print(f"  {sentences[i][:34]:34}", " ".join(f"{score:5.2f}" for score in row))
