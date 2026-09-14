"""The blog's source records. IDs describe records, never path positions."""
NODES = [
    {"node_id": "maya", "name": "Maya", "label": "Person", "properties": {
        "age": 34, "eye_color": "brown", "interests": ["tea", "climbing", "jazz"]}},
    {"node_id": "nina", "name": "Nina", "label": "Person", "properties": {
        "age": 33, "eye_color": "blue", "interests": ["tea", "cooking", "photography"]}},
    {"node_id": "cedar", "name": "Cedar Lab", "label": "Organization", "properties": {
        "sector": "robotics", "city": "Toronto"}},
]
EDGES = [
    {"edge_id": "mentoring_1", "source_id": "maya", "target_id": "nina",
     "type": "MENTORS", "properties": {"frequency": "weekly"}},
    {"edge_id": "employment_1", "source_id": "nina", "target_id": "cedar",
     "type": "WORKS_AT", "properties": {"employment": "full_time"}},
]
# The blog doesn't supply Omar's properties or Pedro's other properties.
# Don't invent them. An empty description contributes zero to a path.
EXTRA_NODES = [
    {"node_id": "omar", "name": "Omar", "label": "Person", "properties": {}},
    {"node_id": "pedro", "name": "Pedro", "label": "Person", "properties": {
        "interests": ["cooking"]}},
]
EXTRA_EDGES = [
    {"edge_id": "mentoring_2", "source_id": "omar", "target_id": "pedro",
     "type": "MENTORS", "properties": {}},
    {"edge_id": "employment_2", "source_id": "pedro", "target_id": "cedar",
     "type": "WORKS_AT", "properties": {}},
]


def connected_paths(nodes, edges):
    """Select actual directed Person -> Person -> Organization paths by IDs.

    Selection is separate from similarity. Matching properties alone cannot
    establish that the target of one edge is the source of the next.
    """
    by_id = {node["node_id"]: node for node in nodes}
    paths = []
    for mentoring in edges:
        if mentoring["type"] != "MENTORS":
            continue
        for employment in edges:
            if employment["type"] != "WORKS_AT":
                continue
            if mentoring["target_id"] != employment["source_id"]:
                continue
            node_ids = [mentoring["source_id"], mentoring["target_id"], employment["target_id"]]
            if [by_id[n]["label"] for n in node_ids] != ["Person", "Person", "Organization"]:
                continue
            paths.append({"node_ids": node_ids,
                          "edge_ids": [mentoring["edge_id"], employment["edge_id"]]})
    return paths
