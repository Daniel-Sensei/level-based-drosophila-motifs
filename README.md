# 🧠 Detecting and Investigating Complex Connectome Motifs in the Adult Drosophila

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![SciPy](https://img.shields.io/badge/SciPy-Sparse%20Linear%20Algebra-005380?logo=scipy)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A scalable, parametric computational pipeline designed to impose a global hierarchical structure on massive, cyclic brain networks and automatically discover complex topological motifs. Validated on the first complete, synaptic-resolution connectome of the adult *Drosophila melanogaster* (130k+ neurons, 2.7M+ connections).

---

## 📖 Overview

Understanding neural computation requires mapping how signals propagate through complex circuits. However, raw connectomes are sparse, heavily cyclic graphs lacking a natural coordinate system. This project addresses this challenge in three main steps:

1. **Hierarchical Flow Ranking** – Formulates the connectome as a physical spring system. It solves a massive sparse linear equation system ($Ly=b$) via the Conjugate Gradient method to assign a vertical continuous coordinate to every neuron.
2. **Supervised Discretization** – Uses a Decision Tree Classifier to group continuous scores into 5 distinct processing layers ($L1$ to $L5$), optimizing for biological superclass purity.
3. **Parametric Motif Extraction** – Abstracts the search space from individual neurons to "Superclass-Layer Blocks", executing a constrained Depth-First Search (DFS) with algebraic path-counting via matrix-vector multiplication.

---

## 🎯 Architectural Principles Discovered

### 1. Vertical Processing & Local Feedback ($NH=1, NJ=1$)
Across all structural windows, Git LFS-tracked subgraphs revealed that pure feedforward pipelines are heavily bypassed by **top-down local feedback loops** (e.g., $L3 \rightarrow L2$), serving as biological persistence and noise-suppression circuits.

### 2. High Horizontal Complexity ($NH \ge 3$)
When cross-talk between multiple superclasses is allowed within the same layer, the framework detected **large directed cycles** spanning multiple functional populations, exposing how intermediate brain stages act as distributed integrators.

### 3. Fast-and-Slow Parallel Streams ($NJ \ge 2$)
Allowing long-range "skip connections" demonstrated that the *Drosophila* visual system bypasses intermediate layers via **shortcut projections**, drastically reducing loop delays for sensorimotor transformations.

---

## 💻 Dataset & Preprocessing

* **Source Data**: Publicly available FlyWire Codex data portal (Adult *Drosophila melanogaster* brain connectome).
* **Graph Scale**: 134,181 neurons and 2,700,513 synaptic connections.
* **Abstraction**: Neurons are grouped into 9 biological superclasses (`sensory`, `visual_centrifugal`, `optic`, `ascending`, `visual_projection`, `central`, `endocrine`, `descending`, `motor`) mapped across 5 hierarchical levels.

---

## 🛠️ Technical Implementation

### Core Technologies
* **Language**: Python (highly optimized sparse scientific stack).
* **Matrix Operations**: SciPy Sparse (Compressed Sparse Column - CSC format) to reduce memory footprint by 10-100x.
* **Statistical Validation**: Quadratic Assignment Procedure (QAP) permutation testing to filter out statistically insignificant subgraphs.

### Custom Algebraic Counting

Instead of relying on slow string hashing or network isomorphism libraries (e.g., DotMotif), this project calculates exact multi-step path occurrences using continuous matrix-vector multiplication:

$$v_{\text{next}} = v_{\text{current}} \cdot M_{ij}$$

The total unique neuronal chains instantiating a complex motif pattern are computed instantaneously via the $L1$-norm of the final state flow vector:

$$\text{count}(P) = \|v_{\text{final}}\|_1$$

---

## 👥 Authors

* **Daniel Curcio**
* **Sebastiano Antonio Piccolo**
* **Giorgio Terracina**

*University of Calabria, Department of Mathematics and Computer Science (DeMaCS)*

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📚 References

1. Lin, A., Yang, R., Dorkenwald, S., et al. (2024). *Network Statistics of the Whole-Brain Connectome of Drosophila*. Nature, 634(8032), 153-165.
2. Corradini, E., Parlapiano, F., Ronci, A., et al. (2025). *A complex network-based approach to detect and investigate connectome motifs in the larval drosophila*. Computers in Biology and Medicine, 192, 110135.
