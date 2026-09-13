"""Verify algebra and the persisted, separate-process teaching workflow."""
import os
from pathlib import Path
import subprocess
import sys
import torch
import torchhd
import lancedb
from graph_to_hypervector.encoding import Encoder, bundle, encode_path, encode_edge, compose_edges
from graph_to_hypervector.graph import NODES, EDGES, connected_paths

ROOT = Path(__file__).resolve().parents[1]


def test_permutation_and_role_preserve_cosine():
    encoder = Encoder()
    h_a, h_b = encoder.fact("interest", "tea"), encoder.fact("interest", "cooking")
    h_role = encoder.role("example")
    before = torchhd.cosine_similarity(h_a, h_b)
    after = torchhd.cosine_similarity(
        torchhd.permute(torchhd.bind(h_role, h_a), shifts=2),
        torchhd.permute(torchhd.bind(h_role, h_b), shifts=2))
    assert torch.allclose(before, after)
    assert torch.equal(torchhd.permute(torchhd.permute(h_a, shifts=1), shifts=-1), h_a)


def test_order_and_connectivity():
    encoder = Encoder()
    h_nodes = [encoder.properties(n["properties"]) for n in NODES]
    h_edges = [encoder.properties({"type": e["type"], **e["properties"]}) for e in EDGES]
    original = encode_path(h_nodes, h_edges)
    reversed_people = encode_path([h_nodes[1], h_nodes[0], h_nodes[2]], h_edges)
    assert not torch.equal(original, reversed_people)
    # A cue in Nina's correct position distinguishes the hypothetical reversal.
    query = torchhd.permute(encoder.fact("interest", "cooking"), shifts=2)
    assert torchhd.cosine_similarity(query, original) > torchhd.cosine_similarity(query, reversed_people)
    disconnected = [EDGES[0], {**EDGES[1], "source_id": "maya"}]
    assert connected_paths(NODES, disconnected) == []


def test_lessons_in_separate_processes(tmp_path):
    env = {**os.environ, "GRAPH_HV_DB": str(tmp_path / "db")}
    for lesson in ["encode_nodes", "query_nodes", "understand_permutation", "encode_paths",
                   "query_paths", "query_shared_pattern"]:
        result = subprocess.run([sys.executable, str(ROOT / "lessons" / f"{lesson}.py")],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    db = lancedb.connect(tmp_path / "db")
    # Longer paths must not have been persisted by any lesson.
    assert set(db.list_tables().tables) == {"persons", "organizations", "mentors", "works_at"}
    mentors = db.open_table("mentors").to_arrow().to_pylist()
    assert {e["edge_id"] for e in mentors} == {"mentoring_1", "mentoring_2"}
    assert all(len(e["vector"]) == 4096 for e in mentors)
    persons = db.open_table("persons").to_arrow().to_pylist()
    organizations = db.open_table("organizations").to_arrow().to_pylist()
    assert {row["node_id"] for row in persons} == {"maya", "nina", "omar", "pedro"}
    assert {row["node_id"] for row in organizations} == {"cedar"}
    assert next(n for n in persons if n["node_id"] == "omar")["vector"] is None
    assert '"name": "Cedar Lab"' in next(n for n in organizations if n["node_id"] == "cedar")["properties"]


def test_runtime_composition_matches_direct_encoding():
    encoder = Encoder()
    h_nodes = [encoder.properties(n["properties"]) for n in NODES]
    h_edges = [encoder.properties({"type": e["type"], **e["properties"]}) for e in EDGES]
    h_first = encode_edge(h_nodes[0], h_edges[0], h_nodes[1])
    h_second = encode_edge(h_nodes[1], h_edges[1], h_nodes[2])
    assert torch.equal(compose_edges(h_first, h_second, h_nodes[1]), encode_path(h_nodes, h_edges))
    # This catches accidental double counting of the shared node.
    duplicated = bundle([h_first, torchhd.permute(h_second, shifts=2)])
    assert not torch.equal(duplicated, encode_path(h_nodes, h_edges))
