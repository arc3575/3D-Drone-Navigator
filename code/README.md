# 3D Drone Planner — Code Overview
This folder contains the A* planner, map I/O helpers, and small runners used in the analyses.

# Key Modules / Functions
AStar3DPlanner

# Multi-floor A* with:

Grid moves (N/S/E/W) + hatch transitions between floors

Battery tracking and optional recharge (+ cells)

Optional enemy trajectories (time-cycled), safe-goal check, edge-swap prevention

Conditional WAIT

Dominance pruning & time-modulo visited states (enemy-aware mode)

Supports heuristics: "manhattan", "euclidean", "hybrid"

# Cost-function variants:

cf0: plain A*

cf1: + battery shaping

cf2: + floor_distance shaping

cf3: + enemy term (disabled in enemy-aware tie-breaker setup)

cf4: + recharge lure (−iota/(d_re+tau))

# Map loaders

load_map(map_dir): returns (floors, start, goal, max_battery, enemies, hatch_map)

CSV layout per floor: # wall, . free, + recharge (optional S/G for visualization)

hatches.csv: bidirectional links

enemies.csv: cyclic paths per enemy (optional)

# Analysis helpers

run_analysis1(map_dir, cf_list=..., heuristic=..., weights=...)
Enemy-aware baseline (guidance = tie-breaker to keep search stable).

run_analysis1_no_enemies(map_dir, ...)
Enemy-free runs; guidance can be additive for stronger separation.

run_gamma_sweep_cf2_no_enemies(map_dir, gamma_values=...)
Sweep floor-distance weight γ (cf2) with enemies off.

run_battery_iota_tau_grid(map_dir, battery_values, iotas, taus, ...)
Sweep recharge lure params (ι, τ) and battery capacities (cf4).

min_feasible_battery(map_dir)
Computes minimax segment length (start/chargers/goal graph) for a principled battery baseline.
