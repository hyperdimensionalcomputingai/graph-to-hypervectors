"""Add Omar and Pedro; retrieve both paths using one composable pattern query."""
from copy import deepcopy
import torchhd
from graph_to_hypervector.encoding import Encoder, bundle, encode_edge
from graph_to_hypervector.runtime import query_two_edges
from graph_to_hypervector.graph import NODES, EDGES, EXTRA_NODES, EXTRA_EDGES
from graph_to_hypervector.storage import (
    NODE_SCHEMA, EDGE_SCHEMA, record_row, show, write_table,
)


def build_collection(encoder, prefix=""):
    # STORAGE CHANGE: the new query mentions Cedar Lab by name. Add that fact
    # before rebuilding incident one-edge encodings; the old sector/city bundle didn't encode its name.
    nodes = deepcopy(NODES + EXTRA_NODES)
    next(node for node in nodes if node["node_id"] == "cedar")["properties"]["name"] = "Cedar Lab"
    edges = deepcopy(EDGES + EXTRA_EDGES)
    h_nodes = {n["node_id"]: encoder.properties(n["properties"]) for n in nodes}
    h_edges = {e["edge_id"]: encoder.properties({"type": e["type"], **e["properties"]}) for e in edges}
    node_rows = [record_row(n, h_nodes[n["node_id"]]) for n in nodes]
    write_table(prefix + "persons", [r for r in node_rows if r["label"] == "Person"], NODE_SCHEMA)
    write_table(prefix + "organizations", [r for r in node_rows if r["label"] == "Organization"], NODE_SCHEMA)
    # Persist only one-edge paths. Rebuild incident edges when node facts change.
    for label, table in [("MENTORS", "mentors"), ("WORKS_AT", "works_at")]:
        rows = []
        for edge in edges:
            if edge["type"] != label:
                continue
            h_edge = encode_edge(h_nodes[edge["source_id"]], h_edges[edge["edge_id"]],
                                 h_nodes[edge["target_id"]])
            rows.append(record_row(edge, h_edge))
        write_table(prefix + table, rows, EDGE_SCHEMA)


def pattern_query(encoder, interest="cooking", employment="WORKS_AT"):
    # QUERY: leave position 0 empty. Nothing about the mentor is required.
    h_employer = bundle([encoder.fact("name", "Cedar Lab"), encoder.fact("city", "Toronto")])
    return bundle([
        torchhd.permute(encoder.fact("type", "MENTORS"), shifts=1),
        torchhd.permute(encoder.fact("interest", interest), shifts=2),
        torchhd.permute(encoder.fact("type", employment), shifts=3),
        torchhd.permute(h_employer, shifts=4),
    ])


def main():
    encoder = Encoder()
    build_collection(encoder)
    h_query = pattern_query(encoder)
    results = query_two_edges(h_query, limit=2)
    print("Mentors -> cooking enthusiast -> Cedar Lab in Toronto")
    show(results)
    assert {tuple(row["edge_ids"]) for row in results} == {
        ("mentoring_1", "employment_1"), ("mentoring_2", "employment_2")}
    print("Both paths match the requested facts. Extra stored facts change cosine scores.")
    print("Omar's unknown properties add no contribution; no extra biography was invented.")


if __name__ == "__main__":
    main()
