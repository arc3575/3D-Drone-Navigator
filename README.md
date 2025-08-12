# 3D Drone Planner — Multi-Floor A* (with Batteries, Hatches, and Optional Enemies)

Overview
________
This project implements a 3D grid planner for a drone navigating across multiple floors using A*. The environment supports:

Multiple floors with hatch links (vertical transitions).

Obstacles (#), free cells (.), and recharge stations (+).

Optional moving enemies with cyclic trajectories.

Battery constraints with refueling at recharge stations.

Several cost functions (cf0–cf4) and heuristics (manhattan, euclidean, hybrid).

Dominance pruning, time-modulo visited states, safe edge checks (no enemy swaps), and conditional WAIT.

Repo Layout (expected)
AStar planner & analysis scripts (Jupyter/py)

Map folders:

Map2 … Map12: 13×20, 1–10 floors (used in Analyses #1–#3)

Map13: 30×45, 2 floors, dense obstacles & chargers (used in Analysis #4)

Each map folder contains:

meta.json

floor_0.csv … floor_(N-1).csv

hatches.csv

enemies.csv

Map File Semantics
meta.json: start [floor,y,x], goal [floor,y,x], num_floors, floor_size [rows,cols], max_battery.

floor_i.csv: symbolic grid for floor i (# wall, . free, + recharge; S/G optional for visualization).

hatches.csv: bidirectional links; columns: from_floor, from_y, from_x, to_floor, to_y, to_x.

enemies.csv (optional): floor, id, path; path is “(y,x);(y,x);…” and loops over time.

Setup
Python 3.9+ recommended.

Install: numpy, pandas, matplotlib (and jupyter if using notebooks).

Quick Start (Notebook)
Load a map and run the baseline analysis (cf0–cf4), enemies enabled:

df = run_analysis1("Map2", cf_list=("cf0","cf1","cf2","cf3","cf4"), heuristic="euclidean")

Enemy-aware planner details:

Safe goal acceptance (must be enemy-free at arrival time).

Edge conflict checks (forbid swapping with an enemy).

Time-modulo state keys using the global enemy period.

Enemy-free variant (pronounced floor shaping):

Use run_analysis1_no_enemies(...) or the gamma sweep helpers for cf2.

Reproducing the Analyses
Analysis #1 (with enemies):

Compare heuristics (manhattan/euclidean) and cost functions (cf0–cf4) on Map2.

Guidance is used as a tie-breaker to prevent search blow-ups with enemies.

Analysis #2 (with enemies):

Same setup but focused on how heuristics (euclidean slightly better) and cost functions behave with trajectories enabled.

Brief write-up explains pivotal algorithm changes: time-mod visited, edge conflict checks, conditional WAIT.

Analysis #3 (no enemies):

Study the floor-distance weight γ on Map3–Map12 (1–10 floors, 13×20).

Use cf2 (battery + floor) with additive floor shaping for a clearer effect.

Plots: Expanded Nodes vs Floors (per γ) or vs γ (per floor), plus normalized savings.

Analysis #4 (battery & recharge shaping, no enemies):

Map13 (2 floors, 30×45), dense belts/ribs + many chargers.

Compute minimal feasible battery B* (minimax segment).

Sweep max_battery around B* and recharge parameters: iota (lure strength), tau (smoothing).

Finding: varying iota strongly affects expansions; tau has minimal impact in the tested range.

Key Parameters
Heuristic: "manhattan", "euclidean", "hybrid".

Cost functions:

cf0: plain A*.

cf1: + battery shaping.

cf2: + floor_distance shaping.

cf3: + enemy proximity term (disabled in final enemy-aware tie-breaker mode).

cf4: + recharge lure (−iota/(d_re+tau), optionally gated by low-battery threshold).

Costs: move_cost=1.0, hatch_cost=0.5, wait_cost≈0.2–0.3.

Enforcement:

Enemy-aware mode: guidance = tie-breaker, optimality preserved; collisions hard-blocked.

Enemy-free mode: additive shaping allowed for clearer separation in experiments.

Tips
For tight, reproducible behavior with enemies, keep guidance as a tie-breaker.

For visible experimental effects (no enemies), use additive γ on floor_distance or an admissible tweak to h (add λ·|Δz| with λ ≤ hatch_cost).

Use provided plotting snippets to visualize expansions vs floors/γ, or normalized “nodes saved”.

License / Contribution
Add your license of choice (e.g., MIT).

PRs welcome for new map generators, enemy schedulers, or additional cost functions.
