# From graph records to searchable hypervectors

This is a runnable companion repo to the HDC Labs [blog post on querying connected data with graphs and hypervectors](https://hyperdimensionalcomputing.ai/blog/from-a-property-graph-to-associative-search/).

We work through the story in the blog post one script at a time. We first retrieve **Nina's stored node hypervector** from the cue `eye_color: blue`, then retrieve **Maya** with `eye_color: brown`. We then encode directed paths and query them from partial descriptions. The final lesson brings semantic similarity into those same operations using a text embedding model that's projected up to hyperspace.

You don't need a graph database or prior experience with HDC. The code here is intended to educate! All source graph records are small Python dictionaries, and we use [LanceDB](https://docs.lancedb.com/) to persist their hypervectors and IDs locally. LanceDB uses PyArrow data types under the hood, and it also provides a graph query engine called [lance-graph](https://github.com/lance-format/lance-graph) to run Cypher queries over the same dataset.

## Set up

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```sh
cd graph-to-hypervector
uv sync --locked
```

TorchHD is distributed as `torch-hd` and imported as `torchhd`. `uv.lock` records the tested dependencies. Everything runs on CPU. No API key is needed, and the categorical lessons don't use an embedding service. The first dependency installation needs internet access.

## The graph we're encoding

We define a toy dataset that is shaped like the following property graph:

```text
Maya --MENTORS--> Nina --WORKS_AT--> Cedar Lab
Omar --MENTORS--> Pedro --WORKS_AT--> Cedar Lab  (added later)
```

| Record | Properties supplied by the article |
| --- | --- |
| Maya | age: 34; eye_color: brown; interests: tea, climbing, jazz |
| Nina | age: 33; eye_color: blue; interests: tea, cooking, photography |
| Cedar Lab | sector: robotics; city: Toronto; name added to the encoding later |
| Maya's mentoring edge | frequency: weekly |
| Nina's employment edge | employment: full_time |
| Omar | No property description supplied |
| Pedro | interests: cooking |
| Omar/Pedro's edges | Relationship types only; other properties unspecified |

Names and IDs are stored as metadata. Except for Cedar Lab's name in the pattern lesson, they aren't included in the encoded properties. No extra facts are invented for the added records for Omar or Pedro. In particular, Omar's empty property description has a zero contribution to the path query, because his standalone node has a null hypervector, which is excluded from cosine search.

## Read and run in this order

| Step | Command | What to look for |
| --- | --- | --- |
| 1 | `uv run python lessons/encode_nodes.py` | Bind eye_color to blue; add bound facts into each node; save Arrow rows. |
| 2 | `uv run python lessons/query_nodes.py` | Score both people, return the best node ID, and retrieve that stored row. |
| 3 | `uv run python lessons/understand_permutation.py` | Move components right by one place; reverse with left shifts. |
| 4 | `uv run python lessons/encode_paths.py` | Precompute one-edge paths in separate mentors and works_at tables. |
| 5 | `uv run python lessons/query_paths.py` | Search mentors directly for a tea drinker mentoring a cooking enthusiast. |
| 6 | `uv run python lessons/query_shared_pattern.py` | Add Omar and Pedro; join and compose two-edge candidates at query time. |
| 7 | `uv run python lessons/query_related_meanings.py` | Use Ollama projections for baking/cooking and employed-at/works-at. Requires setup below. |

Each script has comments separating encoding, storage, querying, and lookup. The shared functions keep repeated plumbing out of the lessons; they don't conceal a framework.

## 1. Node encoding: roles and values

A role and a value each have their own random bipolar hypervector. The codebook is deterministic: seed 2026 plus a SHA-256-derived token seed, so separate processes regenerate the same vocabulary independent of lookup order. There are **4,096 components** throughout.

```python
h_blue_eyes = torchhd.bind(h_eye_color, h_blue)
```

Binding multiplies corresponding components. Bundling adds the bound property facts. Nina has five contributions: age, eye color, and three interests. We keep their sums, including values beyond ±1. We don't threshold or normalize the stored bundles.

`properties` in the database is JSON containing the original graph properties; `vector` is a fixed-size list of 4,096 float32 values. Keeping the complete properties and IDs makes it possible to inspect the source facts after retrieval.

## 2. Node retrieval: compare, then look up the ID

The query is another bound role-value fact. We search the `persons` table; Cedar Lab is stored separately in `organizations`. There is no unbinding step:

```python
h_query = encoder.fact("eye_color", "blue")
results = (
    persons.search(h_query.tolist())
    .distance_type("cosine")
    .where("vector IS NOT NULL")
    .limit(2)
    .to_list()
)
```

The `persons` table contains only people, and the filter excludes null vectors; it does **not** filter on eye color. LanceDB returns cosine distance; the printed similarity is `1 - _distance`. `query_nodes.py` checks that this matches TorchHD's cosine calculation, then uses the winning `node_id` to fetch the stored hypervector and properties.

Measured with the committed lockfile and the initial two-person dataset:

| Query | Rank | node_id | Cosine similarity |
| --- | --- | --- | ---: |
| eye_color: blue | 1 | nina | 0.456691 |
| eye_color: blue | 2 | maya | -0.006118 |
| eye_color: blue + interest: cooking | 1 | nina | 0.632826 |
| eye_color: blue + interest: cooking | 2 | maya | 0.008446 |
| eye_color: brown | 1 | maya | 0.439610 |
| eye_color: brown | 2 | nina | -0.007703 |

Adding the cooking cue raises Nina's score from 0.456691 to 0.632826 while Maya stays near zero. Both facts occur in Nina's stored bundle. The script constructs the two-cue query by bundling the bound blue-eyes and cooking facts.

These are small-example measurements, not a retrieval benchmark. Nina's blue-eyes score isn't 1 because her stored hypervector includes four other facts. The article's band illustrations use a separate illustrative vocabulary; their first ten components needn't be identical to these scripts' arrays.

## 3. Position is a one-element cyclic shift

`rho` means shift right by **one** component, wrapping the last to the first. Applying the same shift twice is `rho²`:

```python
h_once = torchhd.permute(h_example, shifts=1)
h_twice = torchhd.permute(h_once, shifts=1)

# Undo it by shifting left one component, twice.
h_restored = torchhd.permute(h_twice, shifts=-1)
h_restored = torchhd.permute(h_restored, shifts=-1)
assert torch.equal(h_restored, h_example)
```

`shifts=2` is shorthand for two successive single-offset shifts. This is a cyclic permutation, not a random shuffle. The toy lesson displays all ten components; actual hypervectors wrap at component 4,096, not after the first ten displayed values.

The inverse shift restores a permuted hypervector exactly. Applied to a bundle, it moves **all** contributions, so it doesn't extract a clean node by itself. Retrieval still uses similarity and record IDs.

## 4–5. Store one-edge paths; query them directly

Each relationship row stores a **length-one path** with three contributions:

| Position / cyclic shift count | Contribution |
| --- | --- |
| 0 | Source node properties |
| 1 | Relationship type and properties |
| 2 | Target node properties |

`encode_paths.py` writes `mentors` and `works_at`, using `encoding.encode_edge`. Its `vector` column includes both endpoints, not just the relationship's own properties. The row retains its original `edge_id`, `source_id`, `target_id`, type, and edge properties.

```python
h_edge = bundle([
    h_source,
    torchhd.permute(h_relationship, shifts=1),
    torchhd.permute(h_target, shifts=2),
])
```

We store **persons, organizations, mentors, and works_at**. There is no `paths` table and no persisted `path_id`.

The tea/mentoring/cooking query has the same three positions and searches the `mentors` table directly with LanceDB cosine search. Its returned edge ID leads to the mentoring record, and the endpoint IDs retrieve Maya and Nina from `persons`. The first example doesn't request an employment edge; that extension comes next.

## 6. Compose longer candidates at query time

`query_shared_pattern.py` adds the Omar/Pedro records and adds Cedar Lab's name to its organization hypervector. Because an employment hypervector includes its target node, the lesson rebuilds the affected one-edge records too. Source changes require updating stored encodings; "precompute once" means reuse while the source and encoder stay unchanged.

The query is:

> Find people who mentor someone interested in cooking, where that person works at Cedar Lab in Toronto.

At query time, `runtime.query_two_edges`:

1. Reads `mentors` and `works_at`, joining `mentors.target_id` to `works_at.source_id`.
2. Shifts each matching employment hypervector twice to align it after mentoring.
3. Subtracts one copy of the shared node, which would otherwise be counted twice.
4. Scores the composed candidates with TorchHD cosine similarity and returns the top-k with their ordered edge IDs.

```python
h_path = bundle([
    h_mentoring,
    torchhd.permute(h_employment, shifts=2),
]) - torchhd.permute(h_shared_person, shifts=2)
```

This gives the five-position path `[mentor, MENTORS, mentee, WORKS_AT, employer]`. It is exactly equal to composing its five original contributions, because bundles are unthresholded sums. `tests/test_workflow.py` verifies this equality and catches double counting. Use the same encoder and node snapshot for both incident edges and the shared-node subtraction.

The query uses the same five positions, leaving position 0 empty. Both connected pairs contain the requested evidence, even though they involve different people. Their other properties can differ and affect their scores. **Longer candidates and queries are composed at runtime from reusable encodings; we don't precompute every possible longer path.**

The teaching code enumerates all connected pairs. This isn't a claim that joins are free: a larger graph needs a candidate-selection strategy. Prefiltering or per-edge top-k can reduce work but may miss a good full-path match. No connected pairs means no candidate result. One-edge retrieval uses LanceDB; the two-edge search uses TorchHD on in-memory compositions, not a database search over nonexistent stored paths.

IDs and path positions are separate. An edge keeps its stored ID and local positions 0–2; placing it later in a longer path shifts those positions. This example doesn't automatically align every possible query length and start position. Longer cycles eventually wrap around the hypervector dimension.

Rerun steps 1 and 4 to restore the initial four tables. Encoding lessons replace only their named tables. Use `GRAPH_HV_DB=/some/other/directory` for an isolated experiment.

## 7. Add semantic values with text embedding models

In this repo, we use a local Ollama service to pull the `nomic-embed-text` text embedding model. However, you can replace this part of the code with any model of your choice (e.g., `sentence-transformers`).

```sh
ollama pull nomic-embed-text
uv run python lessons/query_related_meanings.py
```

The initial model download requires internet, but all inference is done locally once the model is installed. Set `OLLAMA_HOST` to a full base URL if your service isn't at `http://localhost:11434`. External embedding model providers can use a similar approach.

`semantic.py` calls Ollama's `/api/embed` endpoint. For symmetric comparisons, every phrase gets the same `clustering: ` prefix. The tested model produced 768-dimensional embeddings. The script reads the actual dimension rather than assuming it.

The projection has three visible steps:

1. Normalize the text embedding `x`.
2. Generate one fixed Rademacher matrix `R` with shape `[text_dimensions, 4096]`, using equiprobable -1 and +1 entries and seed 2026.
3. Compute `x @ R`, then map nonnegative values to +1 and negative values to -1.

Only interest values and relationship descriptions use semantic encodings; roles and the other categorical values retain their deterministic random patterns. Relationship labels are embedded as phrases: `WORKS_AT` becomes `works at`, for example. Source graph labels remain unchanged.

The lesson rebuilds **semantic_persons**, **semantic_organizations**, **semantic_mentors**, and **semantic_works_at** with the new encoder before composing and scoring candidates at query time. Querying categorical storage with an unrelated semantic codebook would not be the same experiment.

Measured with `nomic-embed-text:latest` (local model ID `0a109f422b47`), 768 → 4,096 dimensions:

| Projected phrase pair | Cosine similarity |
| --- | ---: |
| baking / cooking | 0.670410 |
| employed at / works at | 0.736328 |

These are projected phrase scores, not path scores. The lesson prints both. Scores can change with the model build, prompt, projection, or data. We don't assert a guaranteed semantic ranking or hardcode synonyms. Asking for top two from two paths always returns both without a cutoff, so it isn't evidence of semantic search quality by itself.

Similarity suggests candidates. Exact graph checks can establish whether a required relationship or property actually holds. The example retains IDs and source records so those checks are possible, but doesn't require a Neo4j installation or pretend to execute Cypher.

## Where to look in the code

- `lessons/`: start here; one teaching stage per script.
- `src/graph_to_hypervector/graph.py`: source facts and a small connectivity check used by tests.
- `encoding.py`: vocabulary, role-value encoding, one-edge encoding, and exact path composition.
- `runtime.py`: endpoint-ID joins and cosine scoring of query-time compositions.
- `storage.py`: Arrow schemas, LanceDB writes, ID lookup, readable output.
- `semantic.py`: Ollama request and the Rademacher projection.
- `config.py`: dimension, seed, and database location.
- `tests/`: executable checks for order, inverse permutation, cosine preservation, and the complete categorical workflow in separate processes.

```sh
uv run pytest -q
```

Tests use a temporary database and don't require a running Ollama server. Run the semantic lesson for the real-model integration check. No local server is started or stopped by these scripts.

References: [TorchHD MAP](https://torchhd.readthedocs.io/en/stable/generated/torchhd.MAPTensor.html), [LanceDB vector search](https://docs.lancedb.com/search/vector-search), [PyArrow schemas](https://arrow.apache.org/docs/python/generated/pyarrow.schema.html), and [Ollama embeddings](https://docs.ollama.com/api/embed).
# graph-to-hypervectors
