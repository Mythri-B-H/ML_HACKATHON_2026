# PathSnap – Dataset Analysis Report

## 1. Executive Summary

This document presents the comprehensive dataset analysis for **PathSnap (Intelligent GPS Route Reconstruction)**.
The analysis was performed across all available raw trajectory and road network files in the workspace.

- **Total Trajectories:** 90 (80 training, 10 student testing, 0 evaluator testing)
- **Total Files:** 360 data files (4 per trajectory: `.track`, `.nodes`, `.arcs`, `.route`)
- **Total GPS Observations:** 227,577 points
- **Total Road Network Nodes:** 2,024,174 vertices
- **Total Road Network Arcs:** 4,061,945 directed edges
- **Total Ground Truth Route Arcs:** 34,813 traversed segments

---

## 2. Directory Structure & File Inventory

The dataset is partitioned into:
- `data/raw/train/`: 80 trajectory subdirectories (`00000001` through `00000099`)
- `data/raw/student_test/`: 10 trajectory subdirectories (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`)
- `data/raw/evaluator_test/`: Currently empty (reserved strictly for final blind evaluation; no label leakage)

### File Types and Schemas

| File Extension | Content Description | Columns / Types | Delimiter | 0-Index Semantic |
| :--- | :--- | :--- | :--- | :--- |
| **`.track`** | Noisy GPS trajectory points | `longitude` (float), `latitude` (float), `timestamp` (float) | Tab (`\t`) | Sequential GPS measurements |
| **`.nodes`** | Road network intersection / shape vertices | `longitude` (float), `latitude` (float) | Tab (`\t`) | Row number = Node ID ($0, 1, \dots, |V|-1$) |
| **`.arcs`** | Directed road segments connecting nodes | `source_node` (int), `target_node` (int) | Tab (`\t`) | Row number = Arc ID ($0, 1, \dots, |E|-1$) |
| **`.route`** | Ground truth sequence of traversed road arcs | `arc_id` (int) | Newline | Ordered list of road segments |

---

## 3. Dataset Summary Statistics

### Split Breakdown

| Metric | Training Set (80 Trajectories) | Student Test Set (10 Trajectories) | Total Combined (90 Trajectories) |
| :--- | :--- | :--- | :--- |
| **Trajectories** | 80 | 10 | 90 |
| **Total GPS Points** | 208,417 | 19,160 | 227,577 |
| **Avg GPS Points / Traj** | 2,605.2 (min: 361, max: 19,313) | 1,916.0 (min: 262, max: 3,958) | 2,528.6 |
| **Total Road Nodes** | 1,933,485 | 90,689 | 2,024,174 |
| **Avg Nodes / Traj** | 24,168.6 (min: 299, max: 155,003) | 9,068.9 (min: 1,770, max: 26,635) | 22,490.8 |
| **Total Road Arcs** | 3,876,904 | 185,041 | 4,061,945 |
| **Avg Arcs / Traj** | 48,461.3 (min: 649, max: 307,392) | 18,504.1 (min: 3,577, max: 56,027) | 45,132.7 |
| **Total Route Arcs** | 31,336 | 3,477 | 34,813 |
| **Avg Route Arcs / Traj** | 391.7 (min: 59, max: 2,266) | 347.7 (min: 95, max: 1,323) | 386.8 |

---

## 4. Geographic & Spatial Characteristics

- **Coordinate System:** WGS 84 (EPSG:4326), Longitude/Latitude in decimal degrees.
- **Longitude Range:** $[-100.418538, +137.237029]$
- **Latitude Range:** $[-34.640291, +61.699590]$
- **Spatial Scope:** Multi-city global benchmark dataset (ACM SIGSPATIAL GIS Cup standard).
- **Sub-network Architecture:** Each trajectory folder is fully self-contained, with its own cropped local road network graph (`.nodes` and `.arcs`) covering the spatial bounding box of the trajectory plus a buffer zone.

---

## 5. Temporal & Kinematic Characteristics

- **Sampling Interval ($\Delta t$):**
  - Minimum: $1.0\text{ s}$
  - Median: $1.0\text{ s}$ (1 Hz nominal sampling)
  - Mean: $2.18\text{ s}$
- **Speed Profile:**
  - 50th percentile (median): $39.85\text{ km/h}$ (urban driving)
  - 75th percentile: $65.99\text{ km/h}$
  - 90th percentile: $98.54\text{ km/h}$ (arterial / motorway)
  - 99th percentile: $166.81\text{ km/h}$
  - Maximum speed: $648.50\text{ km/h}$ (severe GPS multipath/jump outlier)
- **GPS Distance to Ground Truth Route:**
  - Median error: $2.64\text{ meters}$
  - Mean error: $3.54\text{ meters}$
  - Typical $\sigma_{GPS}$: $5 - 10\text{ meters}$

---

## 6. Structural & Graph Verification

1. **Node Reference Integrity:** $100\%$ valid. Every arc $(u, v)$ satisfies $0 \le u < |V|$ and $0 \le v < |V|$.
2. **Route Arc Reference Integrity:** $100\%$ valid. Every route entry $a_k$ satisfies $0 \le a_k < |E|$.
3. **Route Graph Continuity:**
   - Total transitions evaluated: $34,813 - 90 = 34,723$
   - Continuous transitions ($v_k == u_{k+1}$): $34,709$ ($99.96\%$)
   - Discontinuous transitions: $14$ ($0.04\%$) across 7 trajectories
   - Discontinuity cause: Minor unmodeled junction links, roundabout bypasses, or split segment artifacts in the benchmark network.
