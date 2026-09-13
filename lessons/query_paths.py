"""Search a stored one-edge path: tea drinker -> MENTORS -> cooking enthusiast."""
import torchhd
from graph_to_hypervector.encoding import Encoder, bundle
from graph_to_hypervector.storage import by_id, open_table, show


def main():
    encoder = Encoder()
    # QUERY: one edge has three positions, matching each mentors row's encoding.
    h_query = bundle([
        encoder.fact("interest", "tea"),
        torchhd.permute(encoder.fact("type", "MENTORS"), shifts=1),
        torchhd.permute(encoder.fact("interest", "cooking"), shifts=2),
    ])
    mentors = open_table("mentors")
    results = mentors.search(h_query.tolist()).distance_type("cosine").limit(3).to_list()
    print("Tea -> MENTORS -> cooking")
    show(results)
    if not results:
        return
    # LOOKUP: fetch the saved edge and endpoints. No precomputed longer path.
    edge = by_id(mentors, "edge_id", results[0]["edge_id"])
    persons = open_table("persons")
    show([by_id(persons, "node_id", edge["source_id"]),
          by_id(persons, "node_id", edge["target_id"])])
    print("The next lesson joins two relationship tables and composes longer paths at runtime.")


if __name__ == "__main__":
    main()
