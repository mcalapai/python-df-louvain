# df_louvain/__init__.py
import time

__version__ = "0.1.5"

try:
    from . import df_louvain_cpp

    greet = df_louvain_cpp.greet
    LouvainOptions = df_louvain_cpp.LouvainOptions
    LouvainResult = df_louvain_cpp.LouvainResult
    run_static_omp_cpp = df_louvain_cpp.run_static_omp
    run_dynamic_frontier_omp_cpp = df_louvain_cpp.run_dynamic_frontier_omp
    run_naive_dynamic_omp_cpp = df_louvain_cpp.run_naive_dynamic_omp
    run_dynamic_delta_screening_omp_cpp = df_louvain_cpp.run_dynamic_delta_screening_omp

except ImportError as e:
    print(
        f"Warning: C++ extension 'df_louvain_cpp' or its members not found. Error: {e}"
    )

    def greet():
        return "C++ extension not loaded."

    class LouvainOptions:
        def __init__(self, *args, **kwargs):
            pass

    class LouvainResult:
        def __init__(self):
            self.membership = []
            self.vertexWeight = []
            self.communityWeight = []

    def run_static_omp_cpp(*args, **kwargs):
        raise ImportError("C++ ext static not loaded.")

    def run_dynamic_frontier_omp_cpp(*args, **kwargs):
        raise ImportError("C++ ext frontier not loaded.")

    def run_naive_dynamic_omp_cpp(*args, **kwargs):
        raise ImportError("C++ ext naive not loaded.")

    def run_dynamic_delta_screening_omp_cpp(*args, **kwargs):
        raise ImportError("C++ ext delta not loaded.")


class DynamicLouvainState:
    """
    Manages the state for running dynamic Louvain algorithms.

    This class handles the initialization of community structures from an initial graph
    and provides methods to update these communities using various dynamic Louvain
    algorithms when provided with batches of edge deletions and insertions.

    Attributes:
        num_vertices (int): The number of vertices hint provided at initialization, or updated
                            if the graph span expanded.
        current_edges (list[tuple[int, int, float]]): The current list of edges in the graph
                                                     (u, v, weight), representing an undirected graph
                                                     (i.e., both (u,v,w) and (v,u,w) should be present).
        membership (list[int]): The current community membership for each vertex.
        vertex_weights (list[float]): The sum of weights of edges incident to each vertex.
        community_weights (list[float]): The sum of total edge weights of all vertices in each community.
        effective_num_vertices (int): The actual number of vertices (span) considered by the C++ backend,
                                     which might be larger than `num_vertices` if edge IDs exceed it.
    """

    def __init__(
        self,
        initial_edges_tuples: list,
        num_vertices: int,
        static_options: LouvainOptions = None,
    ):
        """
        Initializes the state with a static Louvain run on the initial graph.

        Args:
            initial_edges_tuples: List of edge tuples (u, v, w) for the initial graph.
                                  For Louvain on undirected graphs, ensure both (u,v,w) and (v,u,w)
                                  are provided if the C++ backend treats it as a DiGraph.
            num_vertices: An estimate or exact number of vertices (max_vertex_id + 1).
                          The graph span may be adjusted if edge IDs exceed this.
            static_options: `df_louvain.LouvainOptions` for the initial static run.
                            If None, default options are used with repeat=1.
        """
        self.num_vertices: int = num_vertices
        self.current_edges: list[tuple[int, int, float]] = [
            tuple(e) for e in initial_edges_tuples
        ]

        if static_options is None:
            static_options = LouvainOptions(repeat=1)

        print(
            f"[{time.strftime('%H:%M:%S')}] Initializing DynamicLouvainState with static Louvain run..."
        )
        static_result: LouvainResult = run_static_omp_cpp(
            self.current_edges, self.num_vertices, static_options
        )

        self.effective_num_vertices: int = (
            len(static_result.membership) if static_result.membership else 0
        )
        if (
            self.num_vertices == 0
            and self.effective_num_vertices == 1
            and not initial_edges_tuples
        ):
            self.num_vertices = 1
        elif self.effective_num_vertices > self.num_vertices and self.num_vertices > 0:
            print(
                f"Warning: Graph span expanded from num_vertices hint {self.num_vertices} "
                f"to effective {self.effective_num_vertices} due to edge IDs or default empty graph handling."
            )

        self.membership: list[int] = static_result.membership
        self.vertex_weights: list[float] = static_result.vertexWeight
        self.community_weights: list[float] = static_result.communityWeight

        num_initial_communities = len(set(self.membership)) if self.membership else 0
        print(
            f"[{time.strftime('%H:%M:%S')}] Initial static run complete. "
            f"Found {num_initial_communities} communities for {self.effective_num_vertices} effective vertices. Time: {static_result.time:.2f}ms"
        )

    def _update_python_edges_post_cpp(
        self,
        deletions_batch_orig: list[tuple[int, int, float]],
        insertions_batch_orig: list[tuple[int, int, float]],
    ):
        """
        Updates self.current_edges to reflect the state AFTER C++ processing,
        which includes C++ tidyBatchUpdateU filtering and then applyBatchUpdate.
        This Python version simulates that effect on self.current_edges.
        """

        # Store current edges (before this batch update cycle) to determine existence for filtering
        # This set contains (u,v) pairs for quick existence checks.
        graph_edges_before_this_batch_pairs = set()
        # This dict u -> {v -> w} represents the graph state before this batch, for easy modification.
        current_edges_dict_representation = {}
        for u, v, w in self.current_edges:
            graph_edges_before_this_batch_pairs.add((u, v))
            current_edges_dict_representation.setdefault(u, {})[v] = w

        # 1. Simulate filterEdgesByExistenceU(deletions, x, true) from tidyBatchUpdateU
        # Keep deletions only if (u,v) pair existed in the graph state before this batch.
        effective_deletions_for_apply = []
        for (
            u_del,
            v_del,
            w_del,
        ) in deletions_batch_orig:  # w_del is placeholder for filtering
            if (u_del, v_del) in graph_edges_before_this_batch_pairs:
                effective_deletions_for_apply.append(
                    (u_del, v_del, w_del)
                )  # Keep original tuple

        # 2. Simulate filterEdgesByExistenceU(insertions, x, false) from tidyBatchUpdateU
        # Keep insertions only if (u,v) pair did NOT exist in the graph state before this batch.
        effective_insertions_for_apply = []
        for u_ins, v_ins, w_ins in insertions_batch_orig:
            if (u_ins, v_ins) not in graph_edges_before_this_batch_pairs:
                effective_insertions_for_apply.append((u_ins, v_ins, w_ins))

        # 3. Apply these effectively tidied batches to the Python edge store.
        # Start from the current_edges_dict_representation (state before this batch).
        # The C++ applyBatchUpdateOmpU does removeEdge then addEdge. addEdge overwrites.

        # Apply effective deletions
        for (
            u_del,
            v_del,
            _,
        ) in effective_deletions_for_apply:  # Weight here is ignored for deletion
            if (
                u_del in current_edges_dict_representation
                and v_del in current_edges_dict_representation[u_del]
            ):
                del current_edges_dict_representation[u_del][v_del]
                if not current_edges_dict_representation[u_del]:
                    del current_edges_dict_representation[u_del]

        # Apply effective insertions. these are guaranteed to be for new (u,v) pairs by the filter above)
        for u_ins, v_ins, w_ins in effective_insertions_for_apply:
            current_edges_dict_representation.setdefault(u_ins, {})[v_ins] = w_ins

        # Reconstruct self.current_edges from the modified dictionary
        self.current_edges = []
        for u_dict, targets in current_edges_dict_representation.items():
            for v_dict, w_target in targets.items():
                self.current_edges.append((u_dict, v_dict, w_target))

    def _common_update_logic(
        self,
        deletions_batch: list,
        insertions_batch: list,
        dynamic_options: LouvainOptions = None,
        cpp_function=None,
        method_name="Unknown",
    ) -> LouvainResult:
        if dynamic_options is None:
            dynamic_options = LouvainOptions()
        if cpp_function is None:
            raise ValueError("cpp_function missing")

        print(
            f"[{time.strftime('%H:%M:%S')}] Running dynamic {method_name} update. "
            f"Orig Del: {len(deletions_batch)}, Orig Ins: {len(insertions_batch)}"
        )

        num_v_hint = (
            self.effective_num_vertices
            if self.effective_num_vertices > 0
            else self.num_vertices
        )

        result: LouvainResult = cpp_function(
            self.current_edges,
            deletions_batch,
            insertions_batch,
            self.membership,
            self.vertex_weights,
            self.community_weights,
            num_v_hint,
            dynamic_options,
        )

        self.membership = result.membership
        self.vertex_weights = result.vertexWeight
        self.community_weights = result.communityWeight
        self.effective_num_vertices = len(self.membership) if self.membership else 0

        self._update_python_edges_post_cpp(deletions_batch, insertions_batch)

        num_current_communities = len(set(self.membership)) if self.membership else 0
        print(
            f"[{time.strftime('%H:%M:%S')}] Dynamic {method_name} update complete. "
            f"Found {num_current_communities} communities for {self.effective_num_vertices} effective_vertices. Time: {result.time:.2f}ms. "
            f"Affected: {result.affectedVertices}"
        )
        return result

    def update_frontier(
        self,
        deletions_batch: list,
        insertions_batch: list,
        dynamic_options: LouvainOptions = None,
    ) -> LouvainResult:
        """
        Performs a dynamic update using the Dynamic Frontier Louvain algorithm.

        Args:
            deletions_batch: List of edge tuples (u, v, weight_placeholder) to delete.
            insertions_batch: List of edge tuples (u, v, weight) to insert.
            dynamic_options: `df_louvain.LouvainOptions` for this dynamic update.

        Returns:
            `df_louvain.LouvainResult` from the dynamic update.
        """
        return self._common_update_logic(
            deletions_batch,
            insertions_batch,
            dynamic_options,
            run_dynamic_frontier_omp_cpp,
            "frontier",
        )

    def update_naive(
        self,
        deletions_batch: list,
        insertions_batch: list,
        dynamic_options: LouvainOptions = None,
    ) -> LouvainResult:
        """
        Performs a dynamic update using the Naive Dynamic Louvain algorithm.

        Args:
            deletions_batch: List of edge tuples (u, v, weight_placeholder) to delete.
            insertions_batch: List of edge tuples (u, v, weight) to insert.
            dynamic_options: `df_louvain.LouvainOptions` for this dynamic update.

        Returns:
            `df_louvain.LouvainResult` from the dynamic update.
        """
        return self._common_update_logic(
            deletions_batch,
            insertions_batch,
            dynamic_options,
            run_naive_dynamic_omp_cpp,
            "naive",
        )

    def update_delta_screening(
        self,
        deletions_batch: list,
        insertions_batch: list,
        dynamic_options: LouvainOptions = None,
    ) -> LouvainResult:
        """
        Performs a dynamic update using the Delta-Screening Dynamic Louvain algorithm.

        Args:
            deletions_batch: List of edge tuples (u, v, weight_placeholder) to delete.
            insertions_batch: List of edge tuples (u, v, weight) to insert.
            dynamic_options: `df_louvain.LouvainOptions` for this dynamic update.

        Returns:
            `df_louvain.LouvainResult` from the dynamic update.
        """
        return self._common_update_logic(
            deletions_batch,
            insertions_batch,
            dynamic_options,
            run_dynamic_delta_screening_omp_cpp,
            "delta-screening",
        )


__all__ = [
    "greet",
    "LouvainOptions",
    "LouvainResult",
    "run_static_omp_cpp",
    "run_dynamic_frontier_omp_cpp",
    "run_naive_dynamic_omp_cpp",
    "run_dynamic_delta_screening_omp_cpp",
    "DynamicLouvainState",
]
