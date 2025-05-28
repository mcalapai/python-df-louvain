# Python DF-Louvain

Python bindings for the **DF-Louvain** algorithm, a "Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs," originally implemented in C++ with OpenMP [https://github.com/puzzlef/louvain-communities-openmp-dynamic](<(github)>). This package allows Python users to leverage the performance of the C++ implementation for analyzing large dynamic networks.

Based on the paper: _DF Louvain: Fast Incrementally Expanding Approach for Community Detection on Dynamic Graphs_ by Subhajit Sahu. [https://arxiv.org/abs/2404.19634](<(arxiv)>).

## Features

- **Static Community Detection:** Efficiently find communities in static graphs using the Louvain method.
- **Dynamic Community Detection:** Support for updating communities incrementally as the graph evolves through batch edge insertions and deletions, using:
  - Naive Dynamic Louvain
  - Delta-Screening Dynamic Louvain
  - Dynamic Frontier (DF) Louvain (the core contribution of the paper)
- **Performance:** Leverages the original C++ OpenMP implementation for speed.
- **Configurable:** Allows setting various Louvain algorithm parameters.

## Installation

### Prerequisites

- **C++ Compiler:** A C++17 compliant compiler (e.g., GCC, Clang, MSVC).
- **CMake:** Version 3.18 or higher.
- **OpenMP:** Your C++ compiler must support OpenMP for parallel execution.
  - On macOS, you might need to install a compiler that provides OpenMP (e.g., `brew install llvm` for Clang, or GCC). Ensure the OpenMP runtime libraries are also available.
- **Python:** Version 3.7 or higher.
- **pip** and a Python virtual environment manager (like `venv`).

### Installation Steps

1.  **Create and activate a Python virtual environment:**

    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    ```

2.  **Install `pybind11` and build tools (if not already handled by the build process, though `pyproject.toml` should manage this):**

    ```bash
    pip install pybind11 scikit-build-core cmake
    ```

3.  **Clone the repository (if you haven't already):**

    ```bash
    git clone https://github.com/mcalapai/python-df-louvain.git
    cd python-df-louvain
    ```

4.  **Install the package:**
    From the root directory of the `python-df-louvain` project:
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
num_vertices = 6

static_options = df_louvain.LouvainOptions(repeat=1, resolution=1.0)
static_result = df_louvain.run_static_omp_cpp(static_edges, num_vertices, static_options)

print(f"Static Result: {static_result!r}")
print(f"Membership: {static_result.membership}")
print(f"Affected (Static): {static_result.affectedVertices}") # Should be all nodes for static

# --- Dynamic Louvain Example using DynamicLouvainState ---
print("\n--- Dynamic Louvain Example ---")
# Initial graph state (same as static example for simplicity)
initial_edges_for_dynamic = static_edges
dynamic_state_manager = df_louvain.DynamicLouvainState(initial_edges_for_dynamic, num_vertices, static_options)

print(f"Initial Dynamic State Membership: {dynamic_state_manager.membership}")

# Batch update 1: Strengthen the bridge
deletions_batch1 = [(2, 3, 0.1), (3, 2, 0.1)] # Remove old weak bridge
insertions_batch1 = [(2, 3, 2.0), (3, 2, 2.0)] # Add strong bridge

# Note: C++ tidyBatchUpdateU filters insertions if (u,v) pair exists.
# So, this effectively becomes delete(2,3,0.1) and the insertion of (2,3,2.0) is filtered out by C++ internal tidy.
# The Python _update_python_edges_post_cpp simulates this filtering for its internal state.
dynamic_options = df_louvain.LouvainOptions(repeat=1, resolution=0.8) # Lower res to encourage merge
dynamic_result1 = dynamic_state_manager.update_frontier(deletions_batch1, insertions_batch1, dynamic_options)

print(f"Dynamic Result 1: {dynamic_result1!r}")
print(f"Membership after update 1: {dynamic_result1.membership}")
print(f"Python state edges after update 1: {dynamic_state_manager.current_edges}")


# Batch update 2: Remove the bridge
deletions_batch2 = [(2,3,2.0), (3,2,2.0)] # Remove the strong bridge
insertions_batch2 = []
dynamic_result2 = dynamic_state_manager.update_frontier(deletions_batch2, insertions_batch2, dynamic_options)

print(f"Dynamic Result 2: {dynamic_result2!r}")
print(f"Membership after update 2: {dynamic_result2.membership}")
print(f"Python state edges after update 2: {dynamic_state_manager.current_edges}")
```

## API Overview

The primary interface is through the `df_louvain` module.

### Classes

- **`df_louvain.LouvainOptions`**:
  Used to configure parameters for the Louvain algorithm.
  Attributes: `repeat`, `resolution`, `tolerance`, `aggregationTolerance`, `toleranceDrop`, `maxIterations`, `maxPasses`.

- **`df_louvain.LouvainResult`**:
  Returned by Louvain functions, containing the results.
  Attributes: `membership` (list), `vertexWeight` (list), `communityWeight` (list), `iterations`, `passes`, `time` (ms), `markingTime` (ms), `initializationTime` (ms), `firstPassTime` (ms), `localMoveTime` (ms), `aggregationTime` (ms), `affectedVertices`.

- **`df_louvain.DynamicLouvainState`**:
  A helper class to manage state between dynamic updates.
  - `__init__(self, initial_edges_tuples, num_vertices, static_options=None)`
  - `update_frontier(self, deletions_batch, insertions_batch, dynamic_options=None)`
  - `update_naive(self, deletions_batch, insertions_batch, dynamic_options=None)`
  - `update_delta_screening(self, deletions_batch, insertions_batch, dynamic_options=None)`
    Properties: `current_edges`, `membership`, `vertex_weights`, `community_weights`, `num_vertices`, `effective_num_vertices`.

### Functions (Low-level C++ Bindings)

These are directly bound from the C++ implementation and are used by `DynamicLouvainState`.

- `df_louvain.run_static_omp_cpp(edges, num_vertices, options)`
- `df_louvain.run_dynamic_frontier_omp_cpp(current_edges, deletions, insertions, prev_membership, prev_v_weights, prev_c_weights, num_v_hint, options)`
- `df_louvain.run_naive_dynamic_omp_cpp(...)` (similar signature to frontier)
- `df_louvain.run_dynamic_delta_screening_omp_cpp(...)` (similar signature to frontier)

See the full API documentation for detailed parameter descriptions.

## Building Documentation

Documentation is built using Sphinx.

1.  **Install documentation dependencies:**
    ```bash
    pip install .[docs]
    # or: pip install sphinx sphinx-rtd-theme pybind11-stubgen myst-parser
    ```
2.  **Generate stubs for the C++ module (from project root):**

    ```bash
    pybind11-stubgen df_louvain_cpp -o df_louvain
    ```

    This creates `df_louvain/df_louvain_cpp.pyi`.

3.  **Build HTML documentation (from `docs/` directory):**
    ```bash
    cd docs
    make html
    # Or on Windows: .\make.bat html
    ```
    The documentation will be in `docs/_build/html/index.html`.

## Running Tests

Tests are written using `pytest`.

1.  **Install test dependencies:**
    ```bash
    pip install .[test]
    # or: pip install pytest
    ```
2.  **Run tests (from project root):**
    ```bash
    pytest -v
    ```

## How it Works (Briefly)

The Python functions call wrapped C++ functions. The graph data (list of edge tuples) is converted into the C++ `DiGraph` structure. For dynamic updates, the `DynamicLouvainState` class helps manage the community membership and weights from the previous state, which are required by the C++ dynamic algorithms. The C++ side performs the heavy computation using OpenMP for parallelism.

**Important Note on Edge Representation:** The underlying C++ `DiGraph` is a directed graph. For Louvain, which typically operates on undirected graphs, you must ensure that if an undirected edge `(u,v)` with weight `w` exists, you provide both `(u,v,w)` and `(v,u,w)` in the edge lists passed from Python. The helper function `make_edges_undirected` in the test files can be adapted for this.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
