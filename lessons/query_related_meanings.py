"""Replace selected categorical values with Ollama semantic projections."""
import torchhd
from graph_to_hypervector.encoding import Encoder
from graph_to_hypervector.semantic import OllamaProjection
from graph_to_hypervector.storage import show
from graph_to_hypervector.runtime import query_two_edges
from query_shared_pattern import build_collection, pattern_query


def main():
    projection = OllamaProjection()
    encoder = Encoder(semantic_values=projection)
    # RE-ENCODE STORAGE: semantic queries cannot meaningfully compare against
    # unrelated random cooking/WORKS_AT values from the categorical lessons.
    # Separate tables keep both encoder versions available for comparison.
    build_collection(encoder, prefix="semantic_")
    print(f"Ollama nomic-embed-text: {projection.input_dimensions} -> 4096 dimensions")
    for interest, employment in [("baking", "WORKS_AT"), ("cooking", "EMPLOYED_AT")]:
        h_query = pattern_query(encoder, interest=interest, employment=employment)
        print(f"\nQuery interest={interest!r}, employment={employment!r}")
        show(query_two_edges(h_query, prefix="semantic_", limit=2))

    # MEASURE, don't assume, the similarity supplied by this installed model.
    for a, b in [("baking", "cooking"), ("employed at", "works at")]:
        score = float(torchhd.cosine_similarity(projection(a), projection(b)))
        print(f"Projected phrase cosine: {a!r} / {b!r} = {score:.6f}")
    print("These candidates are not proof of baking expertise or a guaranteed exact predicate match.")
    print("Two candidates and top-k=2 illustrate mechanics, not recall or precision.")


if __name__ == "__main__":
    main()
