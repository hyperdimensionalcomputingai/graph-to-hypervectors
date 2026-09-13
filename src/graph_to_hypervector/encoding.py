"""MAP operations shared by storage and queries. Read lessons/encode_nodes.py first."""
import hashlib
import json
import torch
import torchhd
from .config import DIMENSIONS, SEED


class Encoder:
    """Stable categorical vocabulary, independent of script invocation order.

    query_related_meanings.py replaces only selected value encodings.
    Roles always remain random bipolar hypervectors.
    """
    def __init__(self, semantic_values=None):
        self.semantic_values = semantic_values
        self.cache = {}

    def symbol(self, namespace, value):
        # Unlike Python's hash(), SHA-256 is stable in separate processes.
        token = json.dumps([SEED, namespace, value], sort_keys=True)
        if token not in self.cache:
            seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % (2**63)
            generator = torch.Generator().manual_seed(seed)
            self.cache[token] = torchhd.random(1, DIMENSIONS, "MAP", generator=generator)[0]
        return self.cache[token]

    def role(self, key):
        return self.symbol("role", key)

    def value(self, key, value):
        # Normalize the spellings in the blog, not synonyms.
        if isinstance(value, str):
            value = value.lower().replace(" ", "_")
        if self.semantic_values is not None and key in {"interest", "type"}:
            return self.semantic_values(value.replace("_", " "))
        return self.symbol("value", value)

    def fact(self, key, value):
        return torchhd.bind(self.role(key), self.value(key, value))

    def properties(self, properties):
        facts = []
        for key, value in properties.items():
            # Source graph uses a list-valued 'interests' field. HDC binds each
            # member to the singular 'interest' role, exactly as in the article.
            if key == "interests":
                facts.extend(self.fact("interest", item) for item in value)
            else:
                facts.append(self.fact(key, value))
        return bundle(facts)


def bundle(hypervectors):
    """Unthresholded component sums: +1 and -1 inputs needn't stay bipolar."""
    hypervectors = list(hypervectors)
    if not hypervectors:
        return torch.zeros(DIMENSIONS).as_subclass(torchhd.MAPTensor)
    return torchhd.multiset(torch.stack(hypervectors))


def encode_path(node_vectors, edge_vectors):
    """Five positions; permutation is a right cyclic shift, not an edge ID."""
    h_mentor, h_mentee, h_employer = node_vectors
    h_mentoring, h_employment = edge_vectors
    return bundle([
        h_mentor,
        torchhd.permute(h_mentoring, shifts=1),
        torchhd.permute(h_mentee, shifts=2),
        torchhd.permute(h_employment, shifts=3),
        torchhd.permute(h_employer, shifts=4),
    ])


def encode_edge(h_source, h_relationship, h_target):
    """Precompute a length-one path: source(0), relationship(1), target(2)."""
    return bundle([h_source, torchhd.permute(h_relationship, shifts=1),
                   torchhd.permute(h_target, shifts=2)])


def compose_edges(h_first, h_second, h_shared):
    """Query-time composition from two stored length-one paths.

    Move the second edge two single-offset shifts to the right. Its source
    overlaps the first edge's target; subtract one copy of that shared node.
    This is exact for our unthresholded sums and a consistent encoder snapshot.
    """
    return bundle([h_first, torchhd.permute(h_second, shifts=2)]) - torchhd.permute(h_shared, shifts=2)
