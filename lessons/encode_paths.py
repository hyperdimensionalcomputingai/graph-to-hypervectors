"""Precompute ONLY length-one paths, in one table per relationship type."""
from graph_to_hypervector.encoding import Encoder, encode_edge
from graph_to_hypervector.graph import EDGES
from graph_to_hypervector.storage import EDGE_SCHEMA, by_id, open_table, record_row, show, tensor, write_table


def main():
    encoder = Encoder()
    persons = open_table("persons")
    organizations = open_table("organizations")
    rows_by_table = {"mentors": [], "works_at": []}
    for edge in EDGES:
        # RELATIONSHIP CONTENT: type and properties, without endpoints yet.
        h_relationship = encoder.properties({"type": edge["type"], **edge["properties"]})
        h_source = tensor(by_id(persons, "node_id", edge["source_id"]))
        target_table = persons if edge["type"] == "MENTORS" else organizations
        h_target = tensor(by_id(target_table, "node_id", edge["target_id"]))
        # ONE-EDGE ENCODING: source + rho(relationship) + rho²(target).
        h_edge = encode_edge(h_source, h_relationship, h_target)
        table = "mentors" if edge["type"] == "MENTORS" else "works_at"
        rows_by_table[table].append(record_row(edge, h_edge))
    for name, rows in rows_by_table.items():
        write_table(name, rows, EDGE_SCHEMA)
        print(f"\nStored one-edge paths in {name}:")
        show(rows)
    print("No two-edge paths are stored. The query lesson composes them at runtime.")


if __name__ == "__main__":
    main()
