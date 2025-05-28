#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include <string>
#include <vector>
#include <tuple>
#include <algorithm> // For std::max, std::sort, std::unique
#include <functional> // For std::function

// C++ project headers
#include "_main.hxx"
#include "Graph.hxx"
#include "louvain.hxx"
#include "update.hxx"
#include "batch.hxx"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// --- Type Definitions ---
using K_TYPE = uint32_t;
using V_TYPE = float;
using W_TYPE = double;

using PyGraphType = DiGraph<K_TYPE, None, V_TYPE>;
using PyLouvainOptionsType = LouvainOptions;
using PyLouvainResultType = LouvainResult<K_TYPE, W_TYPE>;
using EdgeTuple = std::tuple<K_TYPE, K_TYPE, V_TYPE>;
using EdgeList = std::vector<EdgeTuple>;


// --- Graph Construction Helper ---
PyGraphType create_graph_from_python(
    const EdgeList& py_edges,
    size_t num_vertices_hint
) {
    PyGraphType x;
    K_TYPE max_id_from_edges = 0;
    if (!py_edges.empty()) {
        for (const auto& edge_tuple : py_edges) {
            max_id_from_edges = std::max({max_id_from_edges, std::get<0>(edge_tuple), std::get<1>(edge_tuple)});
        }
    }
    size_t final_num_vertices = std::max(num_vertices_hint, static_cast<size_t>(max_id_from_edges + 1));
    
    if (final_num_vertices > 0) {
        x.respan(final_num_vertices);
        for (K_TYPE i = 0; i < final_num_vertices; ++i) {
            x.addVertex(i);
        }
    }

    for (const auto& edge_tuple : py_edges) {
        x.addEdge(std::get<0>(edge_tuple), std::get<1>(edge_tuple), std::get<2>(edge_tuple));
    }
    updateOmpU(x);
    return x;
}

// --- Static Louvain Wrapper ---
PyLouvainResultType py_louvain_static_omp(
    const EdgeList& py_edges,
    size_t num_vertices,
    const PyLouvainOptionsType& options
) {
    PyGraphType cpp_graph = create_graph_from_python(py_edges, num_vertices);
    PyLouvainResultType result({}, {}, {}, 0, 0, 0.0f);
    {
        py::gil_scoped_release release_gil;
        result = louvainStaticOmp(cpp_graph, options);
    }
    return result;
}

// --- Common C++ Helper for Dynamic Louvain Wrappers ---
using CxxDynamicLouvainFunc = PyLouvainResultType (*)(
    const PyGraphType&, const EdgeList&, const EdgeList&,
    const std::vector<K_TYPE>&, const std::vector<W_TYPE>&, const std::vector<W_TYPE>&,
    const PyLouvainOptionsType&
);

PyLouvainResultType common_dynamic_wrapper(
    const EdgeList& current_edges_py,
    const EdgeList& deletions_batch_py,
    const EdgeList& insertions_batch_py,
    const std::vector<K_TYPE>& initial_membership_py,
    const std::vector<W_TYPE>& initial_vertex_weights_py,
    const std::vector<W_TYPE>& initial_community_weights_py,
    size_t num_vertices_hint,
    const PyLouvainOptionsType& options,
    CxxDynamicLouvainFunc cxx_dynamic_func_ptr
) {
    PyGraphType graph_y_before_batch = create_graph_from_python(current_edges_py, num_vertices_hint);
    EdgeList cpp_deletions = deletions_batch_py;
    EdgeList cpp_insertions = insertions_batch_py;
    tidyBatchUpdateU(cpp_deletions, cpp_insertions, graph_y_before_batch);
    PyGraphType graph_y_after_batch = graph_y_before_batch;
    applyBatchUpdateOmpU(graph_y_after_batch, cpp_deletions, cpp_insertions);

    PyLouvainResultType result({}, {}, {}, 0, 0, 0.0f);
    {
        py::gil_scoped_release release_gil;
        result = cxx_dynamic_func_ptr(
            graph_y_after_batch, cpp_deletions, cpp_insertions,
            initial_membership_py, initial_vertex_weights_py, initial_community_weights_py,
            options
        );
    }
    return result;
}

// --- Wrappers for specific dynamic Louvain functions ---
PyLouvainResultType py_louvain_dynamic_frontier_omp(
    const EdgeList& current_edges_py, const EdgeList& deletions_batch_py, const EdgeList& insertions_batch_py,
    const std::vector<K_TYPE>& im_py, const std::vector<W_TYPE>& ivw_py, const std::vector<W_TYPE>& icw_py,
    size_t nvh, const PyLouvainOptionsType& opts
) {
    return common_dynamic_wrapper(current_edges_py, deletions_batch_py, insertions_batch_py,
                                  im_py, ivw_py, icw_py, nvh, opts,
                                  &louvainDynamicFrontierOmp<PyGraphType, K_TYPE, V_TYPE, W_TYPE>);
}

PyLouvainResultType py_louvain_naive_dynamic_omp(
    const EdgeList& current_edges_py, const EdgeList& deletions_batch_py, const EdgeList& insertions_batch_py,
    const std::vector<K_TYPE>& im_py, const std::vector<W_TYPE>& ivw_py, const std::vector<W_TYPE>& icw_py,
    size_t nvh, const PyLouvainOptionsType& opts
) {
    return common_dynamic_wrapper(current_edges_py, deletions_batch_py, insertions_batch_py,
                                  im_py, ivw_py, icw_py, nvh, opts,
                                  &louvainNaiveDynamicOmp<PyGraphType, K_TYPE, V_TYPE, W_TYPE>);
}

PyLouvainResultType py_louvain_dynamic_delta_screening_omp(
    const EdgeList& current_edges_py, const EdgeList& deletions_batch_py, const EdgeList& insertions_batch_py,
    const std::vector<K_TYPE>& im_py, const std::vector<W_TYPE>& ivw_py, const std::vector<W_TYPE>& icw_py,
    size_t nvh, const PyLouvainOptionsType& opts
) {
    return common_dynamic_wrapper(current_edges_py, deletions_batch_py, insertions_batch_py,
                                  im_py, ivw_py, icw_py, nvh, opts,
                                  &louvainDynamicDeltaScreeningOmp<PyGraphType, K_TYPE, V_TYPE, W_TYPE>);
}

// --- Module Definition ---
PYBIND11_MODULE(df_louvain_cpp, m) {
    m.doc() = "pybind11 wrapper for DF-Louvain C++ library";

    // LouvainOptions binding
    py::class_<PyLouvainOptionsType>(m, "LouvainOptions")
        .def(py::init<int, double, double, double, double, int, int>(),
             py::arg("repeat") = 1, py::arg("resolution") = 1.0, py::arg("tolerance") = 1e-2,
             py::arg("aggregationTolerance") = 0.8, py::arg("toleranceDrop") = 10.0,
             py::arg("maxIterations") = 20, py::arg("maxPasses") = 10)
        .def_readwrite("repeat", &PyLouvainOptionsType::repeat)
        .def_readwrite("resolution", &PyLouvainOptionsType::resolution)
        .def_readwrite("tolerance", &PyLouvainOptionsType::tolerance)
        .def_readwrite("aggregationTolerance", &PyLouvainOptionsType::aggregationTolerance)
        .def_readwrite("toleranceDrop", &PyLouvainOptionsType::toleranceDrop)
        .def_readwrite("maxIterations", &PyLouvainOptionsType::maxIterations)
        .def_readwrite("maxPasses", &PyLouvainOptionsType::maxPasses)
        .def("__repr__", [](const PyLouvainOptionsType &opts) {
            return "<LouvainOptions repeat=" + std::to_string(opts.repeat) +
                   ", resolution=" + std::to_string(opts.resolution) +
                   ", tolerance=" + std::to_string(opts.tolerance) +
                   ", aggTol=" + std::to_string(opts.aggregationTolerance) +
                   ", tolDrop=" + std::to_string(opts.toleranceDrop) +
                   ", maxIter=" + std::to_string(opts.maxIterations) +
                   ", maxPass=" + std::to_string(opts.maxPasses) + ">";
        });

    // LouvainResult binding
    py::class_<PyLouvainResultType>(m, "LouvainResult")
        .def_readonly("membership", &PyLouvainResultType::membership)
        .def_readonly("vertexWeight", &PyLouvainResultType::vertexWeight)
        .def_readonly("communityWeight", &PyLouvainResultType::communityWeight)
        .def_readonly("iterations", &PyLouvainResultType::iterations)
        .def_readonly("passes", &PyLouvainResultType::passes)
        .def_readonly("time", &PyLouvainResultType::time)
        .def_readonly("markingTime", &PyLouvainResultType::markingTime)
        .def_readonly("initializationTime", &PyLouvainResultType::initializationTime)
        .def_readonly("firstPassTime", &PyLouvainResultType::firstPassTime)
        .def_readonly("localMoveTime", &PyLouvainResultType::localMoveTime)
        .def_readonly("aggregationTime", &PyLouvainResultType::aggregationTime)
        .def_readonly("affectedVertices", &PyLouvainResultType::affectedVertices)
        .def("__repr__", [](const PyLouvainResultType &res) {
            size_t num_communities = 0;
            if (!res.membership.empty()) {
                std::vector<K_TYPE> unique_comms = res.membership;
                std::sort(unique_comms.begin(), unique_comms.end());
                unique_comms.erase(std::unique(unique_comms.begin(), unique_comms.end()), unique_comms.end());
                num_communities = unique_comms.size();
            }
            return "<LouvainResult communities=" + std::to_string(num_communities) +
                   ", iterations=" + std::to_string(res.iterations) +
                   ", time=" + std::to_string(res.time) + "ms>";
        });

    // Expose Static Louvain
    m.def("run_static_omp", &py_louvain_static_omp,
          "Run static Louvain algorithm using OpenMP.",
          py::arg("edges"), py::arg("num_vertices"), py::arg("options") = PyLouvainOptionsType());

    // Expose Dynamic Methods
    m.def("run_dynamic_frontier_omp", &py_louvain_dynamic_frontier_omp,
          "Run dynamic frontier Louvain algorithm using OpenMP.",
          py::arg("current_edges"), py::arg("deletions_batch"), py::arg("insertions_batch"),
          py::arg("initial_membership"), py::arg("initial_vertex_weights"), py::arg("initial_community_weights"),
          py::arg("num_vertices_hint"), py::arg("options") = PyLouvainOptionsType());

    m.def("run_naive_dynamic_omp", &py_louvain_naive_dynamic_omp,
          "Run naive dynamic Louvain algorithm using OpenMP.",
          py::arg("current_edges"), py::arg("deletions_batch"), py::arg("insertions_batch"),
          py::arg("initial_membership"), py::arg("initial_vertex_weights"), py::arg("initial_community_weights"),
          py::arg("num_vertices_hint"), py::arg("options") = PyLouvainOptionsType());

    m.def("run_dynamic_delta_screening_omp", &py_louvain_dynamic_delta_screening_omp,
          "Run dynamic delta-screening Louvain algorithm using OpenMP.",
          py::arg("current_edges"), py::arg("deletions_batch"), py::arg("insertions_batch"),
          py::arg("initial_membership"), py::arg("initial_vertex_weights"), py::arg("initial_community_weights"),
          py::arg("num_vertices_hint"), py::arg("options") = PyLouvainOptionsType());

    m.def("greet", []() {
        std::string message = "Hello from C++.";
        #ifdef _OPENMP
            message += " OpenMP max threads reported by omp_get_max_threads(): " + std::to_string(omp_get_max_threads());
        #else
            message += " OpenMP not enabled/found at compile time for wrapper.";
        #endif
        return message;
    }, "A simple greeting function from C++");
}