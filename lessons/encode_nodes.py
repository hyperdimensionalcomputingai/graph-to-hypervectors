"""Bind property roles to values, bundle each node, and store Arrow rows."""
import torchhd
from graph_to_hypervector.encoding import Encoder, bundle
from graph_to_hypervector.graph import NODES
from graph_to_hypervector.storage import NODE_SCHEMA, record_row, show, write_table


def main():
    encoder = Encoder()

    # ENCODING: one role-value fact. These are the blog's binding strips.
    h_eye_color = encoder.role("eye_color")
    h_blue = encoder.value("eye_color", "blue")
    h_blue_eyes = torchhd.bind(h_eye_color, h_blue)
    print("Bound blue-eyes fact, first 10 components:", h_blue_eyes[:10].tolist())

    # ENCODING: bundling is addition, with no sign threshold afterward.
    h_age_33 = encoder.fact("age", 33)
    h_two_facts = bundle([h_age_33, h_blue_eyes])
    print("Age and eye color bundled:", h_two_facts[:10].tolist())

    # STORAGE: encode all five facts for each person, not just the query cue.
    # Cedar starts with sector and city only; query_shared_pattern.py adds its name.
    rows = [record_row(node, encoder.properties(node["properties"])) for node in NODES]
    # Separate entity tables share the same explicit Arrow schema.
    write_table("persons", [row for row in rows if row["label"] == "Person"], NODE_SCHEMA)
    write_table("organizations", [row for row in rows if row["label"] == "Organization"], NODE_SCHEMA)
    show(rows)
    print("Saved persons and organizations. Next: uv run python lessons/query_nodes.py")


if __name__ == "__main__":
    main()
