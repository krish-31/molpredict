# Product Requirements Document (PRD)
## Multi-Task Graph Representation Learning for Molecular Property Prediction

**Version:** 1.0  
**Date:** May 2026  
**Project Code:** DAA-EL-MTGRL  
**Status:** Draft

---

## 1. Executive Summary

This project delivers a **multi-task deep learning platform** that simultaneously predicts 12 molecular properties — including toxicity, solubility, and binding affinity — from a molecule's graph structure (atoms as nodes, bonds as edges). The system uses a Graph Isomorphism Network (GIN) backbone with task-specific attention heads, gradient conflict resolution via PCGrad, and scaffold-based evaluation splits that reflect real-world drug discovery challenges.

---

## 2. Problem Statement

### 2.1 Background
Drug discovery and materials science require expensive, time-consuming lab experiments to characterize molecular properties. Predictive machine learning models can accelerate this pipeline — but single-task models must be trained independently for each property, leading to:
- Redundant computation and storage
- No knowledge sharing across related tasks
- Poor generalization to structurally novel molecules

### 2.2 Core Challenges
| Challenge | Description |
|-----------|-------------|
| **Negative Transfer** | Learning one property (e.g., toxicity) can degrade predictions of another (e.g., LogP) if their gradient directions conflict |
| **Data Imbalance** | Some molecular properties have far fewer labelled examples than others |
| **Structural Generalization** | Standard random splits leak structurally similar molecules into train/test — scaffold splits are required for realistic evaluation |
| **Task Weighting** | Naïve equal weighting of all 12 tasks causes dominant high-loss tasks to overwhelm others |

---

## 3. Product Vision

> **"One model, twelve predictions, zero compromise."**  
> A single unified representation of any molecule should simultaneously answer every downstream property question with the accuracy of a specialized model.

---

## 4. Target Users & Personas

### Persona A — Computational Chemist
- **Goal:** Quickly screen thousands of candidate molecules before synthesizing any
- **Pain Point:** Running 12 separate models is slow and hard to maintain
- **Needs:** Batch prediction API, confidence scores, interpretable attention maps

### Persona B — ML Researcher
- **Goal:** Benchmark new GNN architectures or loss strategies
- **Pain Point:** Reproducibility and fair comparison across methods
- **Needs:** Modular codebase, standard splits, clean logging

### Persona C — Drug Discovery Scientist
- **Goal:** Filter toxic compounds early in the pipeline
- **Pain Point:** No unified tool correlates toxicity with other ADMET properties
- **Needs:** Multi-property dashboard, molecular visualization, export to CSV/SDF

### Persona D — Academic / Student
- **Goal:** Learn multi-task GNN concepts via a working end-to-end example
- **Pain Point:** Scattered codebases, no documentation
- **Needs:** Clean code, detailed README, step-by-step notebook

---

## 5. Functional Requirements

### 5.1 Core ML Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Ingest SMILES strings and convert to molecular graphs (atoms as nodes, bonds as edges) | P0 |
| FR-02 | Implement a GIN backbone with configurable depth (layers) and width (hidden dim) | P0 |
| FR-03 | Attach 12 task-specific attention heads (one per property) | P0 |
| FR-04 | Implement PCGrad gradient projection to detect and resolve cross-task gradient conflicts | P0 |
| FR-05 | Use learned uncertainty weights (Kendall et al.) for multi-task loss balancing | P0 |
| FR-06 | Support scaffold-based train/val/test splits via RDKit Murcko scaffold decomposition | P0 |
| FR-07 | Report per-task metrics: ROC-AUC (classification), RMSE & R² (regression) | P0 |
| FR-08 | Provide gradient conflict rate monitoring per epoch | P1 |
| FR-09 | Save and load model checkpoints | P1 |
| FR-10 | Export predictions to CSV | P1 |

### 5.2 Web Interface

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-11 | SMILES input field with real-time 2D molecular visualization (RDKit.js or similar) | P0 |
| FR-12 | Dashboard showing all 12 predicted properties with confidence bars | P0 |
| FR-13 | Batch upload via CSV (SMILES column) | P1 |
| FR-14 | Training progress charts: loss curves, per-task AUC, gradient conflict rate | P1 |
| FR-15 | Attention heat-map overlay on molecular graph | P2 |
| FR-16 | Download results as CSV or PDF report | P1 |

### 5.3 API

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-17 | REST endpoint: `POST /predict` — accepts SMILES, returns 12 property scores | P0 |
| FR-18 | REST endpoint: `POST /predict/batch` — accepts CSV, returns predictions JSON | P1 |
| FR-19 | REST endpoint: `GET /model/metrics` — returns current validation metrics | P1 |
| FR-20 | REST endpoint: `POST /train` — triggers training run with config payload | P2 |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Performance** | Single-molecule prediction latency ≤ 500 ms; batch of 1,000 molecules ≤ 60 s |
| **Scalability** | System should handle datasets up to 500,000 molecules |
| **Reproducibility** | All random seeds fixed; results reproducible across runs |
| **Accuracy** | Match or exceed single-task baseline AUC on ≥ 10 of 12 tasks on Tox21 scaffold split |
| **Modularity** | GIN backbone, PCGrad optimizer, and task heads are independently swappable |
| **Documentation** | README, API docs, and annotated Jupyter notebooks |
| **Portability** | Docker container; runs on CPU (demo) and CUDA GPU (training) |

---

## 7. Molecular Properties — Target Tasks

| # | Property | Task Type | Dataset |
|---|----------|-----------|---------|
| 1 | NR-AR (Nuclear Receptor — Androgen Receptor) | Binary Classification | Tox21 |
| 2 | NR-AR-LBD | Binary Classification | Tox21 |
| 3 | NR-AhR | Binary Classification | Tox21 |
| 4 | NR-Aromatase | Binary Classification | Tox21 |
| 5 | NR-ER | Binary Classification | Tox21 |
| 6 | NR-ER-LBD | Binary Classification | Tox21 |
| 7 | NR-PPAR-gamma | Binary Classification | Tox21 |
| 8 | SR-ARE | Binary Classification | Tox21 |
| 9 | SR-ATAD5 | Binary Classification | Tox21 |
| 10 | SR-HSE | Binary Classification | Tox21 |
| 11 | SR-MMP | Binary Classification | Tox21 |
| 12 | SR-p53 | Binary Classification | Tox21 |

> **Extension targets (Phase 2):** Solubility (ESOL, regression), LogP, Lipophilicity, BBBP, BACE, HIV

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Average ROC-AUC across 12 Tox21 tasks (scaffold split) | ≥ 0.82 |
| PCGrad conflict reduction vs. naïve multi-task | ≥ 20% fewer conflicting gradient pairs |
| Inference latency (single molecule, CPU) | ≤ 500 ms |
| Training convergence | ≤ 100 epochs to best validation AUC |
| Negative transfer mitigation | MTL model ≥ single-task on ≥ 10/12 tasks |

---

## 9. Out of Scope (v1.0)

- 3D molecular conformer generation
- Protein–ligand docking integration
- Generative molecular design
- QSAR explainability beyond attention maps
- Federated learning across institutions

---

## 10. Constraints & Assumptions

- **Dataset:** Tox21 from MoleculeNet (publicly available, ~8,000 molecules, 12 binary labels)
- **Framework:** PyTorch + PyTorch Geometric
- **Graph Library:** RDKit for SMILES → graph conversion
- **Environment:** Python 3.10+, CUDA 11.8+ (training), CPU-only Docker image (inference)
- **Timeline:** 6-week development sprint
- **Team:** 2–4 engineers (ML + backend + frontend)

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PCGrad overhead slows training | Medium | Medium | Profile and cache gradient norms; use mixed precision |
| Negative transfer not fully resolved | Medium | High | Ablation study; fallback to task-grouping strategy |
| Class imbalance degrades AUC | High | High | Use weighted BCE loss + oversampling |
| Scaffold split causes severe data reduction | Low | Medium | Tune split ratio; report on both random and scaffold |

---

*End of PRD v1.0*
