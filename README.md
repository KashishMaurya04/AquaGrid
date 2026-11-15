# AquaGrid: Network-Aware Water Distribution & Decision-Support System

AquaGrid is an academic prototype and decision-support platform designed to address **United Nations Sustainable Development Goal 6 (Clean Water and Sanitation)**. The platform integrates core Data Structures and Algorithms (DSA) with statistical anomaly detection and weighted fairness principles to optimize municipal water network operations, detect conduit leaks, schedule field repairs, and evaluate network-aware distribution feasibility.

---

## Architecture Overview

AquaGrid is architected with a strict separation between zero-dependency mathematical and algorithmic engines, data persistence, and web presentation:

```
AquaGrid/
├── app.py                     # Flask web application, routing, CSRF protection, and controllers
├── database.py                # SQLite database access layer, schema migrations, and audit logging
├── requirements.txt           # Minimal runtime dependencies (Flask, Flask-WTF, pytest)
├── pytest.ini                 # Pytest configuration and Python path resolution
│
├── dsa/                       # Zero-Dependency Graph Algorithms
│   ├── graph.py               # Adjacency-list graph representation and detailed BFS simulation
│   ├── dijkstra.py            # Priority-queue Dijkstra shortest path algorithm
│   └── widest_path.py         # Priority-queue Maximum Bottleneck Path (Widest Path) algorithm
│
├── repair_queue/              # Smart Repair Priority Queue
│   ├── max_heap.py            # Custom 0-indexed contiguous array Binary Max Heap
│   ├── scoring.py             # Multi-factor explainable priority scoring engine (0-100 pts)
│   └── manager.py             # Synchronization between leak events and the priority queue
│
├── detection/                 # Statistical Leak Detection
│   └── engine.py              # Provenance-aware sliding-window anomaly detection (Z-scores)
│
├── simulation/                # Telemetry Simulator
│   └── engine.py              # Synthetic flow meter and pressure sensor telemetry generator
│
├── allocation/                # Fair Water Allocation Engine
│   └── engine.py              # Three-tier Weighted Max-Min Fairness with Decimal water-filling
│
├── distribution/              # Network-Aware Delivery Engine
│   ├── engine.py              # Delivery feasibility, bottleneck identification, and contention analysis
│   └── service.py             # Integration service between allocation records and graph routing
│
├── static/                    # CSS stylesheets, brand assets, and client-side JavaScript
├── templates/                 # Jinja2 HTML templates for all 11 core workspace dashboards
└── tests/                     # Comprehensive automated test suite (541 passing tests)
```

---

## Getting Started

### 1. Prerequisites
- **Python 3.10+** (Python 3.11, 3.12, 3.13, or 3.14 recommended)
- **pip** and **virtualenv**

### 2. Environment Setup

Clone or extract the repository, then create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the minimal production and testing dependencies:

```bash
pip install -r requirements.txt
```

*Note: All core DSA, graph routing, max heap, fair allocation, and anomaly detection algorithms are implemented purely in standard Python without external numerical libraries.*

### 4. Database Initialization

Initialize the SQLite database schema and automatic migrations:

```bash
python -c "import database; database.init_db()"
```

The database file will be created in `instance/aquagrid.db` with WAL (Write-Ahead Logging) mode and foreign-key enforcement enabled.

### 5. Running the Application

Launch the Flask web server:

```bash
python -m flask --app app run --port 5000
```

Open your browser to:
```
http://127.0.0.1:5000
```

---

## Executing the Test Suite

AquaGrid maintains an automated regression suite with complete test coverage across all modules:

```bash
# Run full regression suite quietly
python -m pytest -q

# Run with verbose output
python -m pytest -v

# Run specific domain test suites
pytest tests/test_graph.py tests/test_dijkstra.py tests/test_widest_path.py
pytest tests/test_water_allocation.py tests/test_distribution_engine.py
pytest tests/test_e2e_application_workflows.py
```

**Verified Test Baseline**: **541 tests passed** (0 failures, 0 skipped, 100% passing).

---

## Core Data Structures & Computational Complexity

| Module & Algorithm | Data Structure | Time Complexity | Space Complexity | Theoretical Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Network Graph** (`dsa/graph.py`) | Adjacency List (`dict[int, list[int]]`) | $O(V + E)$ construction | $O(V + E)$ | Models municipal water distribution topology, nodes, and pipes. |
| **Breadth-First Search** (`dsa/graph.py`) | FIFO Queue (`collections.deque`) | $O(V + E)$ | $O(V)$ | Evaluates network reachability, component discovery, and hop distance. |
| **Dijkstra Shortest Path** (`dsa/dijkstra.py`) | Min-Heap (`heapq`) | $O((V + E) \log V)$ | $O(V + E)$ | Minimizes cumulative routing resistance/cost across pipe networks. |
| **Widest Path** (`dsa/widest_path.py`) | Priority Queue (Max Bottleneck) | $O((V + E) \log V)$ | $O(V + E)$ | Maximizes the minimum pipe capacity conduit along transport paths. |
| **Smart Repair Queue** (`repair_queue/max_heap.py`) | Custom Binary Max Heap (0-indexed list) | $O(\log N)$ insert / extract, $O(N)$ heapify | $O(N)$ | Prioritizes pipe repair dispatch based on multi-factor urgency. |
| **Leak Anomaly Detector** (`detection/engine.py`) | Sliding Window Buffer (`collections.deque`) | $O(W)$ per reading | $O(W)$ | Detects statistical flow surges and pressure drops via z-scores. |
| **Fair Water Allocation** (`allocation/engine.py`) | Priority Sorted Vector (`Decimal`) | $O(N \log N)$ | $O(N)$ | Distributes scarce supply via three-tier Weighted Max-Min progressive filling. |
| **Distribution Feasibility** (`distribution/engine.py`) | Directed Path Analysis | $O(V + E)$ per zone | $O(V + E)$ | Reconciles allocated volume against physical conduit flow capacity. |

---

## Scope & Simulation Boundaries

AquaGrid is an **academic decision-support and algorithmic modeling prototype**:

1. **Hydraulic Approximation**:
   - The engine computes **volumetric delivery feasibility** ($Q_{\text{req}} = V_{\text{alloc}} / T$, $V_{\text{max}} = C_{\text{path}} \times T$) and identifies conduit bottleneck chokepoints.
   - It **does not** simulate dynamic physical hydraulics (such as EPANET differential Hazen-Williams friction headloss, transient water hammer waves, or nodal pressure-dependent leakage).
2. **Telemetry Simulation**:
   - The Sensor Simulator produces synthetic normal and leak telemetry for demonstration and testing of statistical anomaly detection.
3. **Database Architecture**:
   - Uses SQLite in WAL mode with connection pooling timeouts suitable for prototype exploration and single-writer concurrency. Production municipal deployments would target a distributed transactional database (e.g., PostgreSQL).
