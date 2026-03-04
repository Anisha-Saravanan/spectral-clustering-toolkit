"""
Spectral Clustering & Community Detection Toolkit
Single-file implementation (spectral_clustering_toolkit.py)

Features:
- Load graphs from NetworkX datasets, edge-list CSV, or generate synthetic graphs
- Compute adjacency, degree, and Laplacian matrices (unnormalized & normalized)
- Compute eigenvalues/eigenvectors (NumPy/SciPy fallback)
- Multiple clustering algorithms: KMeans, Agglomerative, DBSCAN
- Cluster evaluation: modularity, silhouette, conductance, normalized cut
- Visualization: matplotlib (static) and Plotly (interactive, optional)
- CLI entrypoint and simple experiment runner
- Save/load results and basic logging

Dependencies:
- numpy
- scipy
- networkx
- matplotlib
- scikit-learn
- python-louvain (optional for modularity comparison)
- plotly (optional for interactive plots)

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Linear algebra and graph libraries
import numpy as np
from numpy.linalg import eigh, eig, norm

try:
    import scipy
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

try:
    import community as community_louvain  
    LOUVAIN_AVAILABLE = True
except Exception:
    LOUVAIN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset: Optional[str] = None
    input_path: Optional[str] = None
    output_dir: str = "results"
    method: str = "spectral"  
    k: int = 2
    normalize: Optional[str] = None  
    clustering_algo: str = "kmeans"  
    seed: int = 42
    verbose: bool = False
    plotly: bool = False


# ------------- Graph Loader -------------

class GraphLoader:
    """Load or generate graphs. Supports NetworkX built-ins and edge-list CSVs."""

    @staticmethod
    def load_karate() -> nx.Graph:
        logger.info("Loading Karate Club graph from NetworkX built-ins")
        return nx.karate_club_graph()

    @staticmethod
    def load_from_edgelist(path: str, delimiter: str = ",") -> nx.Graph:
        logger.info("Loading graph from edge list: %s", path)
        G = nx.Graph()
        with open(path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for row in reader:
                if len(row) >= 2:
                    u, v = row[0].strip(), row[1].strip()
                    try:
                        u = int(u)
                        v = int(v)
                    except Exception:
                        pass
                    G.add_edge(u, v)
        return G

    @staticmethod
    def load_random_graph(n: int = 100, p: float = 0.05, seed: Optional[int] = None) -> nx.Graph:
        logger.info("Generating Erdős–Rényi random graph n=%d p=%.3f", n, p)
        return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def load_from_networkx_name(name: str) -> nx.Graph:
        name = name.lower()
        if name in ("karate", "karate_club"):
            return GraphLoader.load_karate()
        elif name in ("davis",):
            logger.info("Loading Davis Southern women bipartite graph")
            return nx.davis_southern_women_graph()
        else:
            raise ValueError(f"Unknown built-in dataset name: {name}")


# ------------- Laplacian & Matrix Builder -------------

class LaplacianBuilder:
    @staticmethod
    def adjacency_matrix(G: nx.Graph, nodelist: Optional[List[Any]] = None) -> np.ndarray:
        logger.debug("Computing adjacency matrix")
        if nodelist is None:
            nodelist = list(G.nodes())
        A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)
        return A

    @staticmethod
    def degree_matrix(A: np.ndarray) -> np.ndarray:
        logger.debug("Computing degree matrix from adjacency")
        d = np.sum(A, axis=1)
        D = np.diag(d)
        return D

    @staticmethod
    def laplacian_unnormalized(A: np.ndarray) -> np.ndarray:
        D = LaplacianBuilder.degree_matrix(A)
        L = D - A
        return L

    @staticmethod
    def laplacian_symmetric_normalized(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        D = LaplacianBuilder.degree_matrix(A)
        d = np.diag(D)
        # avoid division by zero
        inv_sqrt_d = np.where(d > eps, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt_d)
        L = D - A
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        return L_sym

    @staticmethod
    def laplacian_random_walk(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        D = LaplacianBuilder.degree_matrix(A)
        d = np.diag(D)
        inv_d = np.where(d > eps, 1.0 / d, 0.0)
        D_inv = np.diag(inv_d)
        L_rw = np.eye(A.shape[0]) - D_inv @ A
        return L_rw


# ------------- Eigen Solver -------------

class EigenSolver:
    """Compute eigenvalues and eigenvectors for symmetric matrices (Laplacian).

    Uses SciPy's sparse methods if available, otherwise NumPy's eigh.
    """

    @staticmethod
    def smallest_k_eigenpairs_dense(L: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Computing smallest %d eigenpairs using dense solver", k)
        w, v = eigh(L)
        idx = np.argsort(w)
        w_sorted = w[idx]
        v_sorted = v[:, idx]
        return w_sorted[:k], v_sorted[:, :k]

    @staticmethod
    def smallest_k_eigenpairs_sparse(L: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available, falling back to dense eigen solver")
            return EigenSolver.smallest_k_eigenpairs_dense(L, k)
        logger.info("Computing smallest %d eigenpairs using sparse solver (eigsh)", k)
        spL = csr_matrix(L)
        try:
            w, v = eigsh(spL, k=k, which='SM')
        except Exception as e:
            logger.warning("eigsh failed (%s), falling back to dense solver", e)
            return EigenSolver.smallest_k_eigenpairs_dense(L, k)
        idx = np.argsort(w)
        return w[idx], v[:, idx]

    @staticmethod
    def compute_eigenvectors(L: np.ndarray, k: int, use_sparse: bool = True) -> np.ndarray:
        n = L.shape[0]
        if k >= n:
            logger.info("Requested k >= n, returning full eigenvector matrix")
            w, v = eigh(L)
            return v
        if use_sparse and SCIPY_AVAILABLE:
            _, v = EigenSolver.smallest_k_eigenpairs_sparse(L, k)
            return v
        else:
            _, v = EigenSolver.smallest_k_eigenpairs_dense(L, k)
            return v


# ------------- Clustering -------------

class ClusteringEngine:
    @staticmethod
    def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1)
        norms = np.where(norms > eps, norms, 1.0)
        return X / norms[:, None]

    @staticmethod
    def kmeans(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 300) -> np.ndarray:
        logger.info("Running KMeans clustering k=%d", k)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=max_iter)
        labels = km.fit_predict(X)
        return labels

    @staticmethod
    def agglomerative(X: np.ndarray, k: int) -> np.ndarray:
        logger.info("Running Agglomerative clustering k=%d", k)
        cl = AgglomerativeClustering(n_clusters=k)
        labels = cl.fit_predict(X)
        return labels

    @staticmethod
    def dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        logger.info("Running DBSCAN clustering eps=%.3f min_samples=%d", eps, min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        return labels


# ------------- Evaluation Metrics -------------

class Evaluator:
    """Evaluate clustering outputs with various metrics.

    - Modularity (requires partition mapping)
    - Silhouette score (requires feature matrix)
    - Conductance / normalized cut approximations
    """

    @staticmethod
    def modularity(G: nx.Graph, labels: np.ndarray) -> float:
        partition = {}
        for i, node in enumerate(G.nodes()):
            partition[node] = int(labels[i])
        if LOUVAIN_AVAILABLE:
            try:
                mod = community_louvain.modularity(partition, G)
                logger.info("Computed modularity via python-louvain: %.6f", mod)
                return float(mod)
            except Exception:
                logger.warning("python-louvain modularity call failed; falling back to manual computation")
        A = nx.to_numpy_array(G)
        m = A.sum() / 2.0
        if m == 0:
            return 0.0
        degrees = A.sum(axis=1)
        Q = 0.0
        n = len(labels)
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += A[i, j] - (degrees[i] * degrees[j]) / (2.0 * m)
        Q = Q / (2.0 * m)
        logger.info("Computed modularity (manual): %.6f", Q)
        return float(Q)

    @staticmethod
    def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        unique_labels = set(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
            logger.warning("Silhouette score undefined for number of clusters=%d", len(unique_labels))
            return float('nan')
        try:
            s = silhouette_score(X, labels)
            logger.info("Silhouette score: %.6f", s)
            return float(s)
        except Exception as e:
            logger.warning("Silhouette computation failed: %s", e)
            return float('nan')

    @staticmethod
    def normalized_cut_approx(G: nx.Graph, labels: np.ndarray) -> float:
        label_set = set(labels)
        A = nx.to_numpy_array(G)
        degrees = A.sum(axis=1)
        n = len(labels)
        total_ncut = 0.0
        for c in label_set:
            indices = [i for i, lab in enumerate(labels) if lab == c]
            if len(indices) == 0:
                continue
            cut = 0.0
            vol = degrees[indices].sum()
            for i in indices:
                for j in range(n):
                    if j not in indices:
                        cut += A[i, j]
            if vol > 0:
                total_ncut += cut / vol
        logger.info("Approximate normalized cut: %.6f", total_ncut)
        return float(total_ncut)


# ------------- Visualization -------------

class Visualizer:
    @staticmethod
    def plot_graph(G: nx.Graph, labels: Optional[np.ndarray] = None, title: str = "Graph") -> None:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available: skipping plot_graph")
            return
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        if labels is None:
            nx.draw(G, pos, with_labels=True, node_size=300, cmap=cm.get_cmap('viridis'))
        else:
            unique = sorted(list(set(labels)))
            colors = [plt.cm.tab10(i % 10) for i in labels]
            nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_spectrum(eigenvalues: np.ndarray, title: str = "Laplacian Spectrum") -> None:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available: skipping plot_spectrum")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(eigenvalues)), np.sort(eigenvalues), marker='o')
        plt.xlabel('Index (sorted)')
        plt.ylabel('Eigenvalue')
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_embedding(G: nx.Graph, embedding: np.ndarray, labels: Optional[np.ndarray] = None, title: str = "Embedding (2D)") -> None:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available: skipping plot_embedding")
            return
        if embedding.shape[1] < 2:
            raise ValueError("Embedding must have at least 2 dimensions for 2D plot")
        x = embedding[:, 0]
        y = embedding[:, 1]
        plt.figure(figsize=(8, 6))
        if labels is None:
            plt.scatter(x, y, s=50)
            for i, node in enumerate(G.nodes()):
                plt.text(x[i], y[i], str(node), fontsize=8)
        else:
            scatter = plt.scatter(x, y, c=labels, s=50, cmap='tab10')
            plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title(title)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_interactive_network(G: nx.Graph, labels: Optional[np.ndarray] = None, title: str = "Interactive Graph") -> None:
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available: skipping interactive plot")
            return
        pos = nx.spring_layout(G, seed=42)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

        node_x = []
        node_y = []
        text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(str(node))
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=text, textposition='top center',
            marker=dict(size=10, color=[int(l) if labels is not None else 0 for l in (labels if labels is not None else [0]*len(G))], colorscale='Viridis')
        )
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=title, showlegend=False))
        fig.show()


# ------------- Experiments & Runner -------------

class SpectralClusteringToolkit:
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(self.config.seed)
        self.results: Dict[str, Any] = {}

    def load_graph(self) -> nx.Graph:
        if self.config.input_path:
            return GraphLoader.load_from_edgelist(self.config.input_path)
        elif self.config.dataset:
            return GraphLoader.load_from_networkx_name(self.config.dataset)
        else:
            # default sample
            return GraphLoader.load_karate()

    def build_laplacian(self, G: nx.Graph) -> Tuple[np.ndarray, List[Any]]:
        nodelist = list(G.nodes())
        A = LaplacianBuilder.adjacency_matrix(G, nodelist=nodelist)
        if self.config.normalize is None:
            L = LaplacianBuilder.laplacian_unnormalized(A)
        elif self.config.normalize == 'symmetric':
            L = LaplacianBuilder.laplacian_symmetric_normalized(A)
        elif self.config.normalize == 'random_walk':
            L = LaplacianBuilder.laplacian_random_walk(A)
        else:
            raise ValueError(f"Unknown normalization option: {self.config.normalize}")
        return L, nodelist

    def choose_k_eigenvectors(self, L: np.ndarray, k: int) -> np.ndarray:
        n = L.shape[0]
        if k <= 0:
            raise ValueError("k must be >= 1")
        use_sparse = True
        eigvecs = EigenSolver.compute_eigenvectors(L, k=k, use_sparse=use_sparse)
        if eigvecs.shape[1] < k:
            eigvecs = EigenSolver.compute_eigenvectors(L, k=min(n, k), use_sparse=False)
        X = ClusteringEngine.normalize_rows(eigvecs)
        return X

    def run_clustering(self, X: np.ndarray) -> np.ndarray:
        algo = self.config.clustering_algo
        if algo == 'kmeans':
            labels = ClusteringEngine.kmeans(X, self.config.k, seed=self.config.seed)
        elif algo == 'agglomerative':
            labels = ClusteringEngine.agglomerative(X, self.config.k)
        elif algo == 'dbscan':
            labels = ClusteringEngine.dbscan(X)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algo}")
        return labels

    def evaluate(self, G: nx.Graph, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        scores: Dict[str, Any] = {}
        scores['modularity'] = Evaluator.modularity(G, labels)
        scores['silhouette'] = Evaluator.silhouette(X, labels)
        scores['approx_ncut'] = Evaluator.normalized_cut_approx(G, labels)
        return scores

    def save_results(self, out_dir: str, G: nx.Graph, labels: np.ndarray, X: np.ndarray, scores: Dict[str, Any]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # save labels
        labels_path = os.path.join(out_dir, 'labels.json')
        mapping = {str(node): int(labels[i]) for i, node in enumerate(G.nodes())}
        with open(labels_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        # save scores
        scores_path = os.path.join(out_dir, 'scores.json')
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)
        # save embedding
        emb_path = os.path.join(out_dir, 'embedding.npy')
        np.save(emb_path, X)
        logger.info("Saved results to %s", out_dir)

    def run(self) -> Dict[str, Any]:
        start_t = time.time()
        G = self.load_graph()
        L, nodelist = self.build_laplacian(G)
        # Compute eigenvalues for spectrum plotting if desired
        try:
            evals, evecs = eigh(L)
        except Exception:
            evals = np.linalg.eigvals(L)
            evecs = np.eye(L.shape[0])
        self.results['eigenvalues'] = np.sort(np.real(evals))
        X = self.choose_k_eigenvectors(L, self.config.k)
        labels = self.run_clustering(X)
        self.results['labels'] = labels.tolist()
        self.results['embedding'] = X.tolist()
        scores = self.evaluate(G, X, labels)
        self.results['scores'] = scores
        # Save results
        if self.config.output_dir:
            self.save_results(self.config.output_dir, G, labels, X, scores)
        end_t = time.time()
        self.results['runtime'] = end_t - start_t
        logger.info("Experiment finished in %.3fs", self.results['runtime'])
        # Visualization
        if MATPLOTLIB_AVAILABLE:
            Visualizer.plot_spectrum(self.results['eigenvalues'])
            try:
                Visualizer.plot_embedding(G, np.array(self.results['embedding']), np.array(labels))
            except Exception as e:
                logger.warning("Embedding plot failed: %s", e)
            Visualizer.plot_graph(G, np.array(labels), title='Clustered Graph')
        if PLOTLY_AVAILABLE and self.config.plotly:
            Visualizer.plot_interactive_network(G, np.array(labels))
        return self.results


# ------------- CLI -------------

def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Spectral Clustering Toolkit CLI")
    parser.add_argument('--dataset', type=str, default='karate', help='Built-in dataset name (karate|davis)')
    parser.add_argument('--input', dest='input_path', type=str, default=None, help='Path to edge-list CSV')
    parser.add_argument('--output', dest='output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--method', type=str, default='spectral', help='Method: spectral (default)')
    parser.add_argument('--k', type=int, default=2, help='Number of clusters (k)')
    parser.add_argument('--normalize', type=str, default=None, choices=[None, 'symmetric', 'random_walk'], help='Laplacian normalization')
    parser.add_argument('--clustering', dest='clustering_algo', type=str, default='kmeans', choices=['kmeans', 'agglomerative', 'dbscan'], help='Clustering algorithm')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plotly', action='store_true', help='Enable Plotly interactive plots')
    args = parser.parse_args(argv)
    cfg = Config(dataset=args.dataset, input_path=args.input_path, output_dir=args.output_dir, method=args.method, k=args.k, normalize=args.normalize, clustering_algo=args.clustering_algo, seed=args.seed, plotly=args.plotly)
    return cfg


# ------------- Helpful Example Datasets & Utilities -------------

def write_sample_edgelist(path: str) -> None:
    # Save karate edges to a file as a convenience for users
    G = GraphLoader.load_karate()
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for u, v in G.edges():
            writer.writerow([u, v])
    logger.info("Wrote sample edge list to %s", path)


# ------------- Main Entrypoint -------------

def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    if cfg.verbose:
        logger.setLevel(logging.DEBUG)
    logger.info("Starting Spectral Clustering Toolkit with config: %s", cfg)
    toolkit = SpectralClusteringToolkit(cfg)
    results = toolkit.run()
    # print summary
    logger.info("Scores: %s", results.get('scores'))
    logger.info("Labels: %s", results.get('labels'))


if __name__ == '__main__':
    main()
