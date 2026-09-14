"""Retrieve Nina, then Maya, by similarity to different partial cues."""
import json
import torchhd
from graph_to_hypervector.encoding import Encoder, bundle
from graph_to_hypervector.storage import by_id, open_table, show, tensor


def main():
    encoder = Encoder()
    persons = open_table("persons")
    # QUERY: bind each known fact, then bundle when we have more than one cue.
    # The two-cue query still leaves age and Nina's other interests unspecified.
    queries = [
        ("blue eyes", encoder.fact("eye_color", "blue"), "nina"),
        ("blue eyes + cooking", bundle([
            encoder.fact("eye_color", "blue"),
            encoder.fact("interest", "cooking"),
        ]), "nina"),
        ("brown eyes", encoder.fact("eye_color", "brown"), "maya"),
    ]
    for description, h_query, expected_id in queries:

        # RETRIEVAL: no exact eye-color filter, and no unbinding.
        # The persons table already restricts the entity type; both people are scored.
        results = (persons.search(h_query.tolist()).distance_type("cosine")
                   .where("vector IS NOT NULL")
                   .limit(2).to_list())
        print(f"\nQuery: {description}")
        show(results)
        assert results[0]["node_id"] == expected_id

        # LOOKUP: the returned ID retrieves the original stored row and hypervector.
        row = by_id(persons, "node_id", results[0]["node_id"])
        print("Retrieved by ID:", row["node_id"], json.loads(row["properties"]))
        # Verify LanceDB's cosine distance agrees with TorchHD on persisted vectors.
        for result in results:
            score = float(torchhd.cosine_similarity(h_query, tensor(result)))
            assert abs(score - (1 - result["_distance"])) < 1e-5


if __name__ == "__main__":
    main()
