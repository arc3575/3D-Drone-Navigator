#Enemies disabled
import heapq
import math
import os
import json
import csv
import numpy as np
import pandas as pd

#Helpers that load in Maps

def load_floors(map_dir, num_floors):
    floors = []
    for i in range(num_floors):
        floor_path = os.path.join(map_dir, f"floor_{i}.csv")
        floor_df = pd.read_csv(floor_path, header=None)
        floors.append(floor_df.to_numpy())
    return floors

def load_meta(map_dir):
    meta_path = os.path.join(map_dir, "meta.json")
    with open(meta_path, "r") as f:
        return json.load(f)

def load_enemies(map_dir):
    enemy_path = os.path.join(map_dir, "enemies.csv")
    trajectories = {}
    if not os.path.exists(enemy_path):
        return trajectories
    with open(enemy_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = int(row["id"])
            floor = int(row["floor"])
            path = []
            for point in row["path"].split(";"):
                y, x = map(int, point.strip("()").split(","))
                path.append((y, x))
            trajectories[eid] = (floor, path)
    return trajectories

#Hatches mapped separately from the rest of the Map

def load_hatch_map(map_dir):
    hatch_path = os.path.join(map_dir, "hatches.csv")
    hatch_map = {}
    if not os.path.exists(hatch_path):
        return hatch_map
    with open(hatch_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_key = (int(row["from_floor"]), int(row["from_y"]), int(row["from_x"]))
            to_val  = (int(row["to_floor"]),   int(row["to_y"]),   int(row["to_x"]))
            hatch_map[from_key] = to_val
    return hatch_map

def load_map(map_dir):
    meta = load_meta(map_dir)
    floors = load_floors(map_dir, meta["num_floors"])
    enemies = load_enemies(map_dir)
    hatch_map = load_hatch_map(map_dir)
    return floors, tuple(meta["start"]), tuple(meta["goal"]), meta["max_battery"], enemies, hatch_map


#Astar planner class that can be customized depending on the map
#The following weights map to these respective terms

# beta = battery term
# gamma = floor_distance term
# epsilon = enemy_penalty term
# iota and tau = recharge station luring

#I have found that tau has little effect on the planning, however, iota is very pronounced

#Always set a max_expansions just incase something goes wrong with the wait_cost and the agent waits forever
#Depending on the Map, the move and wait cost will need to be tuned slightly
class AStar3DPlanner:
    def __init__(self, floors, hatch_map, heuristic="manhattan", enemy_trajectories=None,
                 max_battery=2, cf="cf0", beta=0.0, gamma=0.0, epsilon=0.0, iota=0.0, tau=1e-6,
                 wait_cost=0.2, move_cost=1.0, hatch_cost=0.5, max_expansions=100000):
        self.floors = floors
        self.hatch_map = hatch_map
        self.heuristic_type = heuristic
        self.enemy_trajectories = enemy_trajectories or {}
        self.max_battery = max_battery

        # stats
        self.expansions = 0
        self.max_expansions = int(max_expansions)

        # cost function params
        self.cf = cf
        self.beta = float(beta)      # battery shaping
        self.gamma = float(gamma)    # floor_distance shaping (pronounced)
        self.epsilon = float(epsilon)# enemy penalty (disabled in this analysis)
        self.iota = float(iota)      # recharge lure (disabled for cf2 runs)
        self.tau = float(tau)

        # movement costs
        self.wait_cost = float(wait_cost)
        self.move_cost = float(move_cost)
        self.hatch_cost = float(hatch_cost)

        # state trackers
        self.visited = set()
        self.came_from = {}
        self.cost_so_far = {}
        self.actions = {}
        self.path = []
        self.goal = None

        # enemy timing (irrelevant if enemy_trajectories == {})
        paths = [len(p[1]) for p in self.enemy_trajectories.values() if len(p[1]) > 0]
        self.enemy_period = max(paths) if paths else 1

        # precomputed enemy occupancy
        self.occ = self._precompute_enemy_occupancy()
        
        # (optional) dominance map; not needed for this simpler/no-enemy run
        self.best_at = {}
        self.max_time = None  # not used here

        # recharge: only incentivize when low battery
        self.recharge_threshold = 0.25

    # Heuristics
    def heuristic(self, a, b):
        dz, dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
        if self.heuristic_type == "euclidean":
            return math.sqrt(dx**2 + dy**2 + dz**2)
        elif self.heuristic_type == "manhattan":
            return dx + dy + dz
        elif self.heuristic_type == "hybrid":
            return 0.5 * (dx + dy + dz) + 0.5 * math.sqrt(dx**2 + dy**2 + dz**2)
        else:
            return dx + dy + dz

    # Helper methods that allow the agent to move about the grid
    def in_bounds(self, floor, y, x):
        return 0 <= floor < len(self.floors) and 0 <= y < self.floors[floor].shape[0] and 0 <= x < self.floors[floor].shape[1]

    def passable(self, floor, y, x):
        return self.floors[floor][y][x] != "#"

    def is_recharge_station(self, floor, y, x):
        return self.floors[floor][y][x] == "+"

    def get_move_neighbors(self, floor, y, x):
        neighbors = []
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if self.in_bounds(floor, ny, nx) and self.passable(floor, ny, nx):
                neighbors.append((floor, ny, nx))
        if (floor, y, x) in self.hatch_map:
            nf, ny, nx = self.hatch_map[(floor, y, x)]
            if self.in_bounds(nf, ny, nx) and self.passable(nf, ny, nx):
                neighbors.append((nf, ny, nx))
        return neighbors

    # Helper methods that assist the agent in keeping track of and avoiding enemies
    def _precompute_enemy_occupancy(self):
        occ = {}
        for _, (efloor, path) in self.enemy_trajectories.items():
            if not path:
                continue
            L = len(path)
            for t, (y, x) in enumerate(path):
                tmod = t % self.enemy_period
                occ.setdefault((efloor, tmod), set()).add((y, x))
        return occ

    def enemy_in_cell(self, floor, y, x, t):
        tmod = t % self.enemy_period
        return (y, x) in self.occ.get((floor, tmod), set())

    def should_wait(self, f, y, x, t):
        # With no enemies, this always returns False (so WAITs won't be enqueued).
        # Keeping it here for API parity.
        return False

    # Augmentation terms (in general Astar cost function)
    def floor_distance(self, n, goal):
        return abs(n[0] - goal[0])

    def enemy_penalty(self, n, t):
        return 1.0 if self.enemy_in_cell(n[0], n[1], n[2], t) else 0.0

    def d_recharge(self, n):
        f, y, x = n
        if not (0 <= f < len(self.floors)):
            return 1e9
        floor = self.floors[f]
        H, W = floor.shape
        best = 1e9
        for yy in range(H):
            for xx in range(W):
                if floor[yy][xx] == "+":
                    d = abs(yy - y) + abs(xx - x)
                    if d < best:
                        best = d
        return best if best < 1e9 else 1e6

    #Planning method uses heap instead of stack
    def plan(self, start, goal):
        self.goal = goal
        self.expansions = 0
        self.visited.clear()
        self.came_from.clear()
        self.cost_so_far.clear()
        self.actions.clear()
        self.path = []

        frontier = []
        heapq.heappush(frontier, (0.0, (start, 0, self.max_battery)))
        self.came_from[(start, 0, self.max_battery)] = None
        self.cost_so_far[(start, 0, self.max_battery)] = 0.0

        while frontier and self.expansions < self.max_expansions:
            _, (pos, t, b) = heapq.heappop(frontier)
            f, y, x = pos
            self.expansions += 1

            if pos == self.goal and not self.enemy_in_cell(f, y, x, t):
                self.reconstruct_path((pos, t, b))
                return True

            tmod = t % self.enemy_period
            visited_key = (f, y, x, b, tmod)
            if visited_key in self.visited:
                continue
            self.visited.add(visited_key)

            #Move neighbors
            for nf, ny, nx in self.get_move_neighbors(f, y, x):
                nt = t + 1
                nb = b - 1
                if nb < 0 and not self.is_recharge_station(f, y, x):
                    continue
                if self.is_recharge_station(nf, ny, nx):
                    nb = self.max_battery

                # Early accept if next is goal and safe
                if (nf, ny, nx) == self.goal and not self.enemy_in_cell(nf, ny, nx, nt):
                    self.came_from[((nf, ny, nx), nt, nb)] = ((f, y, x), t, b)
                    self.actions[((nf, ny, nx), nt, nb)] = "HATCH" if (f != nf) else "MOVE"
                    self.reconstruct_path(((nf, ny, nx), nt, nb))
                    return True

                if self.enemy_in_cell(nf, ny, nx, nt):
                    continue

                is_hatch_move = (f != nf)
                step_cost = self.hatch_cost if is_hatch_move else self.move_cost
                new_cost = self.cost_so_far[((f, y, x), t, b)] + step_cost

                # BASE f
                h = self.heuristic((nf, ny, nx), self.goal)
                f_base = new_cost + h

                # ADDITIVE GUIDANCE (pronounced)
                n_next = (nf, ny, nx)
                battery_used_proxy = float(self.max_battery - nb)
                add_batt  = self.beta  * battery_used_proxy if self.cf in ("cf1","cf2","cf3","cf4") else 0.0
                add_floor = self.gamma * self.floor_distance(n_next, self.goal) if self.cf in ("cf2","cf3","cf4") else 0.0
                # enemies disabled for this analysis
                add_enemy = 0.0
                d_re = self.d_recharge(n_next)
                add_rechg = (- self.iota / (d_re + self.tau)) if self.cf in ("cf4",) else 0.0
                recharge_bias = 0.0  # disable charger bias for clean cf2 runs

                priority = f_base + add_batt + add_floor + add_enemy + add_rechg + recharge_bias

                state = ((nf, ny, nx), nt, nb)
                if state not in self.cost_so_far or new_cost < self.cost_so_far[state]:
                    self.cost_so_far[state] = new_cost
                    heapq.heappush(frontier, (priority, state))
                    self.came_from[state] = ((f, y, x), t, b)
                    self.actions[state] = "HATCH" if is_hatch_move else "MOVE"

            #Only triggered if Map enabled enemies
            if self.should_wait(f, y, x, t):
                nt = t + 1
                # refill if on a charger (just in case)
                nb = self.max_battery if self.is_recharge_station(f, y, x) else b
                if not self.enemy_in_cell(f, y, x, nt):
                    new_cost = self.cost_so_far[((f, y, x), t, b)] + self.wait_cost
                    h = self.heuristic((f, y, x), self.goal)
                    f_base = new_cost + h

                    battery_used_proxy = float(self.max_battery - nb)
                    add_batt  = self.beta  * battery_used_proxy if self.cf in ("cf1","cf2","cf3","cf4") else 0.0
                    add_floor = self.gamma * self.floor_distance((f, y, x), self.goal) if self.cf in ("cf2","cf3","cf4") else 0.0
                    add_rechg = (- self.iota / (self.d_recharge((f, y, x)) + self.tau)) if self.cf in ("cf4",) else 0.0

                    priority = f_base + add_batt + add_floor + add_rechg
                    state = ((f, y, x), nt, nb)
                    if state not in self.cost_so_far or new_cost < self.cost_so_far[state]:
                        self.cost_so_far[state] = new_cost
                        heapq.heappush(frontier, (priority, state))
                        self.came_from[state] = ((f, y, x), t, b)
                        self.actions[state] = "WAIT"

        return False  # frontier exhausted

    def reconstruct_path(self, goal_state):
        current_state = goal_state
        path = []
        while current_state in self.came_from and self.came_from[current_state] is not None:
            pos, t, b = current_state
            action = self.actions.get(current_state, "MOVE")
            path.append((pos, t, b, action))
            current_state = self.came_from[current_state]
        pos, t, b = current_state
        path.append((pos, t, b, "START"))
        path.reverse()
        # trim trailing WAIT at goal
        if len(path) >= 2:
            last = path[-1]
            second_last = path[-2]
            if (last[0] == self.goal and second_last[0] == self.goal and last[3] == "WAIT"):
                path.pop()
        self.path = path

    def plan_steps(self):
        return len(self.path) - 1 if self.path else None


#runner method, no enemies by default

def run_analysis1_no_enemies(
    map_dir,
    cf_list=("cf2",),                 # battery + floor only
    heuristic="euclidean",
    weights=dict(beta=0.2, gamma=1.0, epsilon=0.0, iota=0.0, tau=1e-6),
    wait_cost=0.2, move_cost=1.0, hatch_cost=0.5,
    max_expansions=200000
):
    floors, start, goal, max_battery, _enemies, hatch_map = load_map(map_dir)
    enemy_trajectories = {}  # force enemies OFF

    rows = []
    for cf in cf_list:
        planner = AStar3DPlanner(
            floors=floors,
            hatch_map=hatch_map,
            max_battery=max_battery,
            enemy_trajectories=enemy_trajectories,
            heuristic=heuristic,
            cf=cf,
            wait_cost=wait_cost,
            move_cost=move_cost,
            hatch_cost=hatch_cost,
            max_expansions=max_expansions,
            **weights
        )
        found = planner.plan(start=start, goal=goal)
        rows.append({
            "map": os.path.basename(map_dir.rstrip("/")),
            "heuristic": heuristic,
            "cf": cf,
            "found": found,
            "expansions": planner.expansions,
            "steps": planner.plan_steps()
        })
    return pd.DataFrame(rows)


#Gamma sweep method, also no enemies (by default)
def run_gamma_sweep_cf2_no_enemies(map_dir, gamma_values=(0.0, 0.5, 1.0, 2.0, 3.0),
                                   beta=0.2, heuristic="euclidean"):
    out = []
    for gamma in gamma_values:
        df = run_analysis1_no_enemies(
            map_dir,
            cf_list=("cf2",),
            heuristic=heuristic,
            weights=dict(beta=beta, gamma=gamma, epsilon=0.0, iota=0.0, tau=1e-6),
            wait_cost=0.2, move_cost=1.0, hatch_cost=0.5,
            max_expansions=200000
        )
        df["gamma"] = gamma
        out.append(df)
    return pd.concat(out, ignore_index=True)