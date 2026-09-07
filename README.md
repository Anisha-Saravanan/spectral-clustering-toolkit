# Spectral Clustering & Community Detection Toolkit

A Python-based toolkit for **spectral clustering and community detection on graphs** using graph Laplacians, eigenvector-based embeddings, and multiple clustering algorithms.

The project provides a complete pipeline for loading or generating graphs, constructing Laplacian matrices, computing spectral embeddings, performing clustering, evaluating the resulting communities, and visualizing the results.

## Features

* Load graphs from:

  * NetworkX built-in datasets
  * CSV edge lists
  * Synthetic Erdős–Rényi random graphs
* Compute:

  * Adjacency matrices
  * Degree matrices
  * Unnormalized Laplacian
  * Symmetric normalized Laplacian
  * Random-walk Laplacian
* Compute eigenvalues and eigenvectors using:

  * NumPy
  * SciPy sparse eigenvalue solver when available
* Spectral embedding with row normalization
* Multiple clustering algorithms:

  * K-Means
  * Agglomerative Clustering
  * DBSCAN
* Evaluate clustering using:

  * Modularity
  * Silhouette Score
  * Approximate Normalized Cut
* Visualizations using:

  * Matplotlib
  * Plotly (optional)
* Save experiment results as:

  * JSON labels
  * JSON evaluation scores
  * NumPy embedding matrix
* Command-line interface (CLI)
* Logging and experiment runtime tracking
* Optional comparison with Louvain modularity

## Project Structure

```text
Spectral-Clustering-Toolkit/
│
├── .streamlit/                    # Streamlit configuration
├── assets/                         # Images and project assets
├── scripts/                        # Supporting scripts
│
├── app.py                          # Application / interface
├── spectral_clustering_toolkit.py  # Core spectral clustering implementation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```
