"""Explicit Arrow schemas and a few database helpers, not an ORM."""
import json
import lancedb
import pyarrow as pa
import torch
import torchhd
from .config import DB_PATH, DIMENSIONS

# Float32 fixed-size lists retain signed sums and their magnitudes.
VECTOR = pa.list_(pa.float32(), DIMENSIONS)
NODE_SCHEMA = pa.schema([
    ("node_id", pa.string()), ("name", pa.string()), ("label", pa.string()),
    ("properties", pa.string()), ("vector", VECTOR),
])
EDGE_SCHEMA = pa.schema([
    ("edge_id", pa.string()), ("source_id", pa.string()), ("target_id", pa.string()),
    ("type", pa.string()), ("properties", pa.string()), ("vector", VECTOR),
])


def connect():
    return lancedb.connect(DB_PATH)


def write_table(name, rows, schema):
    arrow = pa.Table.from_pylist(rows, schema=schema)
    # Only the named lesson table is replaced; never delete the database folder.
    return connect().create_table(name, data=arrow, mode="overwrite", on_bad_vectors="null")


def open_table(name):
    try:
        return connect().open_table(name)
    except Exception as error:
        raise RuntimeError(f"Could not open {name!r}. Run the preceding encoding lesson first.") from error


def record_row(record, vector):
    return {**record, "properties": json.dumps(record["properties"], sort_keys=True),
            "vector": vector.tolist() if torch.count_nonzero(vector) else None}


def tensor(row):
    # Unknown properties have no query contribution. A null vector avoids storing
    # a zero-norm vector for cosine search (Omar's unspecified node properties).
    values = row["vector"] if row["vector"] is not None else [0.0] * DIMENSIONS
    return torch.tensor(values, dtype=torch.float32).as_subclass(torchhd.MAPTensor)


def by_id(table, column, record_id):
    # SQL string escaping also handles apostrophes in externally supplied IDs.
    literal = record_id.replace("'", "''")
    rows = table.search().where(f"{column} = '{literal}'").limit(1).to_list()
    if not rows:
        raise LookupError(f"Missing {column}={record_id!r}")
    return rows[0]


def show(rows):
    """Print record IDs and facts without dumping 4,096 components."""
    for row in rows:
        display = dict(row)
        values = display.pop("vector", None)
        if values is not None:
            display["hypervector_preview"] = values[:10]
        if "_distance" in display:
            display["cosine_similarity"] = round(1 - display.pop("_distance"), 6)
        if "properties" in display:
            display["properties"] = json.loads(display["properties"])
        print(json.dumps(display, indent=2))
