# Webflow — UI/UX Design Document
## Multi-Task Graph Representation Learning for Molecular Property Prediction

**Version:** 1.0  
**Date:** May 2026

---

## 1. Design Philosophy

| Principle | Application |
|-----------|-------------|
| **Scientific clarity** | Data-dense layouts inspired by research dashboards (PubChem, ChEMBL); no decorative chrome |
| **Progressive disclosure** | Simple SMILES input first; advanced options tucked behind expandable panels |
| **Visual hierarchy** | Critical predictions (toxic/non-toxic) use bold color signals; secondary metrics subdued |
| **Dark mode first** | Chemistry tools traditionally use dark interfaces; reduces eye strain in lab settings |

**Color Palette:**
| Token | Hex | Use |
|-------|-----|-----|
| Background | `#0D1117` | Page background |
| Surface | `#161B22` | Card/panel background |
| Surface-Elevated | `#1F2937` | Input/hover state |
| Brand | `#58A6FF` | Primary actions, links |
| Accent | `#3FB950` | Success / non-toxic indicator |
| Danger | `#F85149` | Toxic / high-risk indicator |
| Warn | `#E3B341` | Medium risk / uncertain |
| Text-Primary | `#E6EDF3` | Headings and labels |
| Text-Secondary | `#8B949E` | Metadata, captions |
| Border | `#30363D` | Dividers, input borders |

**Typography:** Inter (Google Fonts), monospace code: JetBrains Mono

---

## 2. Navigation Architecture

```
App Root
├── Landing Page          [/]
├── Predict
│   ├── Single Molecule   [/predict]
│   └── Batch Upload      [/predict/batch]
├── Train
│   ├── Configure Run     [/train/configure]
│   └── Monitor Training  [/train/monitor]
├── Results               [/results/:run_id]
├── Model Info            [/model]
└── Docs                  [/docs]
```

**Global Nav Bar (top, sticky):**
- Left: Logo + "MolPredict" wordmark
- Center: Nav links (Predict | Train | Results | Docs)
- Right: Model status badge (● Loaded / ○ Loading) + GitHub link

---

## 3. Screen-by-Screen Wireframes

---

### Screen 1 — Landing Page `/`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV: MolPredict    Predict | Train | Results | Docs   ● v1.0] │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│         HERO SECTION                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  🔬  Multi-Task Molecular Property Prediction              │ │
│  │                                                            │ │
│  │  Predict 12 toxicity endpoints simultaneously from         │ │
│  │  a SMILES string using Graph Isomorphism Networks.         │ │
│  │                                                            │ │
│  │  [ Enter SMILES...                          ] [Predict →]  │ │
│  │                                                            │ │
│  │  Try: Aspirin  |  Caffeine  |  Paracetamol                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  FEATURE CARDS (3-column grid)                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ 🕸 GIN Model │ │ ⚡ PCGrad    │ │ 🧪 Scaffold  │            │
│  │              │ │              │ │   Split       │            │
│  │ 5-layer GIN  │ │ Conflict-    │ │ Tox21 bench- │            │
│  │ backbone     │ │ free multi-  │ │ mark, hard   │            │
│  │ 300-dim      │ │ task gradients│ │ evaluation   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  BENCHMARK STRIP                                                 │
│  "Avg ROC-AUC: 0.843 on Tox21 Scaffold Split — 12 tasks"       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Quick-pick chips ("Aspirin", "Caffeine") auto-fill SMILES and navigate to `/predict`
- Hero input → Enter → navigates to `/predict?smiles=...`

---

### Screen 2 — Single Molecule Predict `/predict`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV]                                                           │
├─────────────────────────────┬────────────────────────────────────┤
│   INPUT PANEL (left 40%)    │   RESULTS PANEL (right 60%)       │
│                             │                                    │
│  SMILES Input               │  ╔═══════════════════════════╗    │
│  ┌───────────────────────┐  │  ║ 2D Molecule Visualization ║    │
│  │ CC(=O)Oc1ccccc1C(=O)O│  │  ║   (RDKit.js rendered SVG) ║    │
│  └───────────────────────┘  │  ║                           ║    │
│  [✓ Valid SMILES]           │  ╚═══════════════════════════╝    │
│                             │                                    │
│  [  Predict Properties  ]   │  PREDICTION RESULTS               │
│                             │  ┌──────────────────────────────┐ │
│  ── Advanced Options ──▼    │  │ NR-AR         ██░░░░░░  12% │ │
│  • Return attention maps    │  │ NR-AR-LBD     █░░░░░░░   8% │ │
│  • Threshold: [0.5 ▼]      │  │ NR-AhR        ████░░░░  45% │ │
│  • Export format: [CSV ▼]  │  │ NR-Aromatase  ██░░░░░░  23% │ │
│                             │  │ NR-ER         ███░░░░░  31% │ │
│  MOLECULE STATS             │  │ NR-ER-LBD     █░░░░░░░  17% │ │
│  • Formula: C9H8O4          │  │ NR-PPAR-γ     █░░░░░░░   9% │ │
│  • MW: 180.04 Da            │  │ SR-ARE        █████░░░  52% ⚠│ │
│  • HBA: 4  HBD: 1           │  │ SR-ATAD5      █░░░░░░░  14% │ │
│  • LogP: 1.19               │  │ SR-HSE        ██░░░░░░  28% │ │
│                             │  │ SR-MMP        █░░░░░░░  11% │ │
│                             │  │ SR-p53        █░░░░░░░  19% │ │
│                             │  └──────────────────────────────┘ │
│                             │  [⬇ Download CSV]  [⬇ PDF Report]│
└─────────────────────────────┴────────────────────────────────────┘
```

**Interactions:**
- SMILES input → real-time validation badge (green ✓ / red ✗)
- Progress bars animate on results load (staggered entrance, 50ms delay per task)
- Bars color-coded: green < 30%, yellow 30–50%, red > 50%
- ⚠ icon on SR-ARE (>50%) with tooltip: "Above toxicity threshold"
- Hover on any task bar → tooltip with task description
- "Download CSV" → triggers `GET /predict/export?smiles=...`

---

### Screen 3 — Batch Upload `/predict/batch`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV]                                                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Batch Molecule Prediction                                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │      ☁  Drag & Drop CSV file here                       │   │
│  │         or  [Browse Files]                               │   │
│  │                                                          │   │
│  │      Expected format: CSV with column "smiles"           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [Download Template CSV]                                         │
│                                                                  │
│  ── Once uploaded ──────────────────────────────────────────    │
│                                                                  │
│  FILE: molecules.csv   (1,247 rows detected)                     │
│  Status: ████████████████████░░  83% complete (1,035 / 1,247)  │
│  ETA: ~8 seconds                                                 │
│                                                                  │
│  PREVIEW TABLE (first 5 results):                                │
│  ┌──────────────────┬───────┬────────┬─────────┬──────────┐    │
│  │ SMILES           │ NR-AR │ NR-AhR │ SR-ARE  │ ... +9  │    │
│  ├──────────────────┼───────┼────────┼─────────┼──────────┤    │
│  │ CC(=O)Oc1...    │  12%  │  45%   │  52% ⚠  │         │    │
│  │ c1ccccc1         │   3%  │   8%   │  11%    │         │    │
│  │ CCO              │   1%  │   2%   │   3%    │         │    │
│  └──────────────────┴───────┴────────┴─────────┴──────────┘    │
│                                                                  │
│  [⬇ Download Full Results CSV]   [⬇ Download PDF Summary]       │
└──────────────────────────────────────────────────────────────────┘
```

---

### Screen 4 — Train Configuration `/train/configure`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV]                                                           │
├──────────────────────────────────────────────────────────────────┤
│  Configure Training Run                                          │
│                                                                  │
│  ┌── Model Architecture ───────────────────────────────────┐    │
│  │  GIN Layers:        [5  ▼]                              │    │
│  │  Hidden Dim:        [300]                               │    │
│  │  Dropout:           [0.50]                              │    │
│  │  Graph Pooling:     [Sum ▼]  (Mean / Sum / Max)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── Training Config ──────────────────────────────────────┐    │
│  │  Dataset:           [Tox21 ▼]                           │    │
│  │  Split Strategy:    (●) Scaffold  ( ) Random            │    │
│  │  Split Ratio:       Train [80] Val [10] Test [10]       │    │
│  │  Batch Size:        [128]                               │    │
│  │  Max Epochs:        [200]                               │    │
│  │  Learning Rate:     [0.001]                             │    │
│  │  Early Stopping:    [20] epochs patience                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── Multi-Task Strategy ──────────────────────────────────┐    │
│  │  [✓] Use PCGrad (Projecting Conflicting Gradients)      │    │
│  │  [✓] Learned Uncertainty Weighting (Kendall et al.)     │    │
│  │  [ ] Task Grouping (experimental)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Run Name: [my_run_001          ]                                │
│                                                                  │
│  [  Start Training  ]   [Save Config as JSON]                    │
└──────────────────────────────────────────────────────────────────┘
```

---

### Screen 5 — Training Monitor `/train/monitor`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV]                              Run: my_run_001  [■ Stop]    │
├──────────────────────────────────────────────────────────────────┤
│  Epoch 47 / 200   ██████████████████░░░░  ETA: 38 min           │
│                                                                  │
│  ┌── Live Charts (2-column) ───────────────────────────────┐    │
│  │  [Chart 1: Total Loss vs Epoch]  [Chart 2: Avg AUC]     │    │
│  │  Line chart, train=blue          Line chart, val=orange  │    │
│  │  val=orange dashed               best marked with ★      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── Per-Task AUC Heatmap (current epoch) ─────────────────┐    │
│  │  Task      │ Epoch 1 │ ... │ Epoch 47 │ Best  │ Δ      │    │
│  │  NR-AR     │  0.61   │     │  0.82    │ 0.82  │ +0.21  │    │
│  │  NR-AhR    │  0.58   │     │  0.87    │ 0.87  │ +0.29  │    │
│  │  SR-ARE    │  0.63   │     │  0.79    │ 0.80  │ +0.16  │    │
│  │  ...       │  ...    │     │  ...     │ ...   │ ...    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── Gradient Conflict Monitor ────────────────────────────┐    │
│  │  Current conflict rate: 18.3%  (↓ from 31.2% at epoch 1)│    │
│  │  [Bar chart: conflict rate per epoch]                    │    │
│  │                                                          │    │
│  │  Most conflicting pair: NR-ER ↔ SR-MMP   (dot = -0.14) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── Uncertainty Weights ──────────────────────────────────┐    │
│  │  [Horizontal bar chart: σ_i per task, live updating]    │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

### Screen 6 — Results / Evaluation `/results/:run_id`

```
┌──────────────────────────────────────────────────────────────────┐
│  [NAV]                                          Run: my_run_001  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Final Evaluation — Scaffold Split Test Set                      │
│                                                                  │
│  Average ROC-AUC:  0.843  ▲ +0.028 vs. single-task baseline    │
│                                                                  │
│  ┌── Comparison Table ─────────────────────────────────────┐    │
│  │ Task       │ Single-Task │ MTL Naive │ Ours │  Δ Best  │    │
│  │ NR-AR      │   0.801     │   0.793   │ 0.821│  +0.020  │    │
│  │ NR-AR-LBD  │   0.844     │   0.832   │ 0.857│  +0.013  │    │
│  │ NR-AhR     │   0.876     │   0.869   │ 0.891│  +0.015  │    │
│  │ SR-ARE     │   0.782     │   0.771   │ 0.803│  +0.021  │    │
│  │ ...        │   ...       │   ...     │ ...  │  ...     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌── ROC Curves (per task, interactive) ──────────────────┐    │
│  │  [Tabs: NR-AR | NR-AhR | SR-ARE | ...]                 │    │
│  │  [SVG ROC curve with AUC annotation]                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  [⬇ Download Full Report]   [🔁 Compare Another Run]            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Inventory

| Component | Description | Used On |
|-----------|-------------|---------|
| `<SMILESInput>` | Validated text input + RDKit.js preview | Predict, Landing |
| `<MoleculeViewer>` | 2D SVG render of molecule | Predict |
| `<PropertyBar>` | Animated probability bar with color coding | Predict |
| `<BatchDropzone>` | Drag-and-drop CSV upload with progress | Batch |
| `<PredictionTable>` | Paginated table with highlighted toxicity flags | Batch |
| `<LossChart>` | Line chart (Recharts) for loss curves | Monitor |
| `<AUCHeatmap>` | Epoch × Task heatmap | Monitor |
| `<ConflictRateBar>` | Per-epoch conflict rate bar chart | Monitor |
| `<UncertaintyChart>` | Horizontal bar chart for σ_i | Monitor |
| `<ComparisonTable>` | Multi-model AUC comparison | Results |
| `<ROCCurve>` | Interactive SVG ROC curve | Results |
| `<ConfigForm>` | Training hyperparameter form | Train Config |
| `<ModelStatusBadge>` | Loaded/loading indicator in nav | Global |

---

## 5. Responsive Breakpoints

| Breakpoint | Layout Change |
|-----------|---------------|
| < 768px (mobile) | Single column; charts collapse to summary cards |
| 768–1024px (tablet) | 2-column; some charts use compact mode |
| > 1024px (desktop) | Full 2–3 column layouts as shown |

---

## 6. Accessibility

- All color-coded signals (green/yellow/red) additionally use icons (✓, ⚠, ✗)
- ARIA labels on all interactive elements
- Keyboard-navigable prediction form
- Minimum contrast ratio 4.5:1 for all text on backgrounds

---

*End of Webflow v1.0*
