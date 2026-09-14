"""Join stored one-edge rows and compose longer candidates only at query time."""
import torchhd
from .encoding import compose_edges
from .storage import by_id, open_table, tensor


def query_two_edges(h_query, prefix="", limit=2):
    mentors = open_table(prefix + "mentors")
    works_at = open_table(prefix + "works_at")
    persons = open_table(prefix + "persons")
    candidates = []
    # Small teaching dataset: enumerate all connected pairs. This is a graph
    # join, not a vector search that magically infers missing connectivity.
    for mentoring in mentors.to_arrow().to_pylist():
        for employment in works_at.to_arrow().to_pylist():
            if mentoring["target_id"] != employment["source_id"]:
                continue
            shared = by_id(persons, "node_id", mentoring["target_id"])
            # No property facts are re-encoded here: reuse persisted hypervectors.
            h_path = compose_edges(tensor(mentoring), tensor(employment), tensor(shared))
            similarity = float(torchhd.cosine_similarity(h_query, h_path))
            candidates.append({
                "node_ids": [mentoring["source_id"], mentoring["target_id"], employment["target_id"]],
                "edge_ids": [mentoring["edge_id"], employment["edge_id"]],
                "vector": h_path.tolist(), "_distance": 1 - similarity,
            })
    # No candidates means no result. Composability doesn't create graph edges.
    return sorted(candidates, key=lambda row: row["_distance"])[:limit]
