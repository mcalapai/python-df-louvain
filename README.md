# Python DF-Louvain

Python bindings for the **DF-Louvain** algorithm, a "Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs," [originally implemented in C++ with OpenMP](https://github.com/puzzlef/louvain-communities-openmp-dynamic). This package allows Python users to leverage the performance of the C++ implementation for analyzing large dynamic networks.

Based on the paper: _DF Louvain: Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs_ by [Subhajit Sahu](https://arxiv.org/abs/2404.19634).

## Features

- **Static Community Detection:** Efficiently find communities in static graphs using the Louvain method.
- **Dynamic Community Detection:** Support for updating communities incrementally as the graph evolves through batch edge insertions and deletions, using:
  - Naive Dynamic Louvain
  - Delta-Screening Dynamic Louvain
  - Dynamic Frontier (DF) Louvain (the core contribution of the paper)
- **Performance:** Leverages the original C++ OpenMP implementation for speed.
- **Configurable:** Allows setting various Louvain algorithm parameters.

## Installation

### Installing from PyPI

Simply run:

```bash
pip install df-louvain==0.1.5
```

### Installing Pre-compiled Wheels

**From GitHub Releases:**
If wheels are provided with GitHub releases for your platform and Python version, you can install a specific version directly:

1.  Go to the [GitHub Releases page](https://github.com/mcalapai/python-df-louvain/releases).
2.  Find the desired release (e.g., `v0.1.5`).
3.  Download the appropriate wheel file (`.whl`) for your system (e.g., `df_louvain-0.1.5-cp311-cp311-manylinux_x86_64.whl`).
4.  Install the downloaded wheel:
    ```bash
    pip install /path/to/downloaded/df_louvain-0.1.5-cp311-cp311-manylinux_x86_64.whl
    ```

**Runtime Dependencies:** Even when installing from a wheel, you might need the OpenMP runtime libraries installed on your system if they were not statically linked into the wheel. For many Linux distributions, these are often available via the system package manager (e.g., `libgomp1` for GCC). On macOS, if OpenMP was linked via `brew install llvm`, the runtime should be handled. On Windows, the Microsoft VC++ Redistributable corresponding to the compiler used to build the wheel might be required.

### Building from Source

If a pre-compiled wheel is not available for your system, or if you prefer to build from source, you will need the following prerequisites:

- **C++ Compiler:** A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
- **CMake:** Version 3.18 or higher.
- **OpenMP:** Your C++ compiler must support OpenMP for parallel execution.
- **Python:** Version 3.7 or higher.
- **pip** and a Python virtual environment manager (like `venv`).
- **Build dependencies:** `pybind11`, `scikit-build-core`.

**Steps to build from source:**

1.  **Create and activate a Python virtual environment:**

    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    ```

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/mcalapai/python-df-louvain.git
    cd python-df-louvain
    ```

3.  **Install the package (this will compile the C++ extension):**
    ```bash
    pip install .
    ```

## Quick Start

Here's a basic example of using the static and dynamic Louvain algorithms:

```python
import df_louvain

# --- Static Louvain Example ---
print("--- Static Louvain Example ---")
# Edges: (u, v, weight). For undirected graph, provide both directions.
static_edges = [
    (0, 1, 1.0), (1, 0, 1.0),
    (1, 2, 1.0), (2, 1, 1.0),
    (0, 2, 1.0), (2, 0, 1.0), # Clique 0-1-2
    (3, 4, 1.0), (4, 3, 1.0),
    (4, 5, 1.0), (5, 4, 1.0),
    (3, 5, 1.0), (5, 3, 1.0), # Clique 3-4-5
    (2, 3, 0.1), (3, 2, 0.1)  # Weak bridge
]
num_vertices = 6 # Vertices are 0-indexed, so 0 to 5

static_options = df_louvain.LouvainOptions(repeat=1, resolution=1.0)
static_result = df_louvain.run_static_omp_cpp(static_edges, num_vertices, static_options)

print(f"Static Result: {static_result!r}")
print(f"Membership: {static_result.membership}")

# --- Dynamic Louvain Example using DynamicLouvainState ---
print("\n--- Dynamic Louvain Example ---")
# Initial graph state (same as static example for simplicity)
initial_edges_for_dynamic = static_edges
# Note: DynamicLouvainState runs an initial static Louvain pass internally.
dynamic_state_manager = df_louvain.DynamicLouvainState(
    initial_edges_tuples=initial_edges_for_dynamic,
    num_vertices=num_vertices,
    static_options=static_options # Options for the initial static run
)
print(f"Initial Dynamic State Membership: {dynamic_state_manager.membership}")

# Batch update 1: Effectively remove the bridge
# C++ tidyBatchUpdateU filters insertions if (u,v) pair exists.
# So, delete(2,3,0.1) + insert(2,3,2.0) becomes just delete(2,3,0.1) by C++ tidy.
# The Python state manager simulates this filtering.
deletions_batch1 = [(2, 3, 0.1), (3, 2, 0.1)]
insertions_batch1 = [(2, 3, 2.0), (3, 2, 2.0)] # This insertion part will be filtered by C++ tidy

dynamic_options = df_louvain.LouvainOptions(repeat=1, resolution=1.0)
dynamic_result1 = dynamic_state_manager.update_frontier(deletions_batch1, insertions_batch1, dynamic_options)

print(f"Dynamic Result 1 (bridge removed): {dynamic_result1!r}")
print(f"Membership after update 1: {dynamic_result1.membership}")
# Edge (2,3) should be gone from Python state due to simulated tidy in _update_python_edges_post_cpp
assert not any(e[:2] == (2,3) or e[:2] == (3,2) for e in dynamic_state_manager.current_edges)
```

## API Reference

The primary interface is through the `df_louvain` module.

### `df_louvain.LouvainOptions`

Configuration options for the Louvain algorithm.

- `__init__(self, repeat: int = 1, resolution: float = 1.0, tolerance: float = 0.01, aggregationTolerance: float = 0.8, toleranceDrop: float = 10.0, maxIterations: int = 20, maxPasses: int = 10)`

  - `repeat` (int, default: 1): Number of times the core Louvain algorithm is repeated (for dynamic methods, this applies to each update). Currently, the C++ `louvainInvokeOmp` uses this for `measureDurationMarked`, implying it's for averaging timings rather than algorithmic stability through multiple runs.
  - `resolution` (float, default: 1.0): The resolution parameter (gamma) for modularity calculation. Higher values tend to find more, smaller communities.
  - `tolerance` (float, default: 0.01): Modularity gain tolerance for convergence in the local moving phase. If the gain is below this, the phase stops.
  - `aggregationTolerance` (float, default: 0.8): If the ratio of new communities to old communities after an aggregation pass is above this threshold, the aggregation might be considered to have low shrinkage, and further passes might stop. (Range: 0.0 to 1.0).
  - `toleranceDrop` (float, default: 10.0): Factor by which the `tolerance` is divided after each pass, making convergence criteria stricter in later passes.
  - `maxIterations` (int, default: 20): Maximum number of iterations within a single local moving phase of a pass.
  - `maxPasses` (int, default: 10): Maximum number of passes (local moving + aggregation cycles).

- **Attributes:** All constructor arguments are available as read-write attributes (e.g., `opts.resolution = 1.2`).

### `df_louvain.LouvainResult`

Holds the results of a Louvain community detection run. All attributes are read-only from Python.

- `membership` (list[int]): Community ID assigned to each vertex. The length corresponds to the number of vertices processed by the C++ backend.
- `vertexWeight` (list[float]): Total weight of edges incident to each vertex (sum of degrees if unweighted, sum of edge weights if weighted). Corresponds to the initial state for dynamic calls (`utot` from C++).
- `communityWeight` (list[float]): Sum of `vertexWeight` for all vertices belonging to each community. Corresponds to the initial state for dynamic calls (`ctot` from C++).
- `iterations` (int): Total number of local-moving iterations performed across all passes.
- `passes` (int): Total number of passes (local-moving + aggregation cycles) performed.
- `time` (float): Total execution time for the C++ Louvain algorithm call (in milliseconds).
- `markingTime` (float): Time spent in initially marking affected vertices (in milliseconds, primarily for dynamic methods).
- `initializationTime` (float): Time spent initializing weights and community structures (in milliseconds).
- `firstPassTime` (float): Time spent in the first full pass of the Louvain algorithm (in milliseconds).
- `localMoveTime` (float): Total time spent in all local-moving phases (in milliseconds).
- `aggregationTime` (float): Total time spent in all aggregation phases (in milliseconds).
- `affectedVertices` (int): Number of vertices initially marked as affected and considered for processing in the first iteration of the first pass (primarily for dynamic methods, relies on C++ `louvain.hxx` fix).

### `df_louvain.DynamicLouvainState`

A helper class to manage state between dynamic updates. It runs an initial static Louvain pass upon instantiation.

- `__init__(self, initial_edges_tuples: list[tuple[int, int, float]], num_vertices: int, static_options: LouvainOptions = None)`

  - `initial_edges_tuples` (list[tuple[int, int, float]]): Edges `(u, v, weight)` for the initial graph. For undirected graphs, ensure both `(u,v,w)` and `(v,u,w)` are provided.
  - `num_vertices` (int): An estimate or exact number of vertices (max_vertex_id + 1). The actual graph span might be adjusted by the C++ backend if edge IDs exceed this or if `num_vertices` is 0.
  - `static_options` (`LouvainOptions`, optional): Options for the initial static Louvain run. Defaults to `LouvainOptions(repeat=1)`.

- `update_frontier(self, deletions_batch: list[tuple[int,int,float]], insertions_batch: list[tuple[int,int,float]], dynamic_options: LouvainOptions = None) -> LouvainResult`
  Performs a dynamic update using the Dynamic Frontier Louvain algorithm.

  - `deletions_batch` (list[tuple[int,int,float]]): Edges `(u,v,w_placeholder)` to delete. The weight is typically ignored for identifying the edge to delete by the C++ backend (uses u,v pair).
  - `insertions_batch` (list[tuple[int,int,float]]): Edges `(u,v,w)` to insert.
  - `dynamic_options` (`LouvainOptions`, optional): Options for this specific dynamic update.
  - **Returns:** `LouvainResult` for this update.

- `update_naive(self, deletions_batch: list[tuple[int,int,float]], insertions_batch: list[tuple[int,int,float]], dynamic_options: LouvainOptions = None) -> LouvainResult`
  Performs a dynamic update using the Naive Dynamic Louvain algorithm. Parameters are the same as `update_frontier`.

- `update_delta_screening(self, deletions_batch: list[tuple[int,int,float]], insertions_batch: list[tuple[int,int,float]], dynamic_options: LouvainOptions = None) -> LouvainResult`
  Performs a dynamic update using the Delta-Screening Dynamic Louvain algorithm. Parameters are the same as `update_frontier`.

- **Properties (accessible after initialization/updates):**
  - `current_edges` (list[tuple[int, int, float]]): Python's representation of the graph's edges after the latest update. Updated by `_update_python_edges_post_cpp` to simulate C++ `tidyBatchUpdateU` effects.
  - `membership` (list[int]): Current community IDs.
  - `vertex_weights` (list[float]): Current vertex weights.
  - `community_weights` (list[float]): Current community weights.
  - `num_vertices` (int): The original `num_vertices` hint, potentially updated if initial graph construction expanded span.
  - `effective_num_vertices` (int): The actual number of vertices (span) processed by the C++ backend in the last operation.

### Low-Level C++ Bound Functions (from `df_louvain.df_louvain_cpp`)

These functions are called by `DynamicLouvainState` or can be used directly if managing state manually. They all expect edge lists where undirected edges are represented by including both `(u,v,w)` and `(v,u,w)`. Vertex IDs are 0-indexed `int` (from C++ `uint32_t`). Edge weights are `float`. Louvain internal weights (`W_TYPE`) are `float` (from C++ `double`).

- `run_static_omp_cpp(edges: list[tuple[int,int,float]], num_vertices: int, options: LouvainOptions = LouvainOptions()) -> LouvainResult`
  Runs static Louvain.

  - `edges`: List of all edges `(u,v,w)`.
  - `num_vertices`: Max vertex ID + 1.
  - `options`: `LouvainOptions`.

- `run_dynamic_frontier_omp_cpp(current_edges: list[tuple[int,int,float]], deletions_batch: list[tuple[int,int,float]], insertions_batch: list[tuple[int,int,float]], initial_membership: list[int], initial_vertex_weights: list[float], initial_community_weights: list[float], num_vertices_hint: int, options: LouvainOptions = LouvainOptions()) -> LouvainResult`
  Runs Dynamic Frontier Louvain.

  - `current_edges`: Edges of the graph _before_ applying the current `deletions_batch` and `insertions_batch`. Used by C++ to establish `graph_y_before_batch` for `tidyBatchUpdateU`.
  - `deletions_batch`, `insertions_batch`: The batches of changes.
  - `initial_membership`, `initial_vertex_weights`, `initial_community_weights`: Louvain state from _after the previous_ update.
  - `num_vertices_hint`: Hint for graph span.
  - `options`: `LouvainOptions`.

- `run_naive_dynamic_omp_cpp(...) -> LouvainResult`
  (Same signature as `run_dynamic_frontier_omp_cpp`)

- `run_dynamic_delta_screening_omp_cpp(...) -> LouvainResult`
  (Same signature as `run_dynamic_frontier_omp_cpp`)

- `greet() -> str`
  A simple test function.

**Important Note on Edge Representation & Graph State:**
The C++ backend uses a `DiGraph`. For Louvain on undirected graphs, ensure edge lists passed from Python represent undirected edges by including both `(u,v,w)` and `(v,u,w)` for each conceptual undirected edge.
The `DynamicLouvainState.current_edges` attempts to mirror the graph state after C++-side batch processing (including the effects of `tidyBatchUpdateU`). Be aware that `tidyBatchUpdateU` filters insertions of `(u,v)` pairs if `(u,v)` already exists, which means an attempt to "change weight" by deleting an old edge and inserting a new one with the same `(u,v)` key might result in only the deletion being effectively processed by the C++ Louvain algorithm step.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
