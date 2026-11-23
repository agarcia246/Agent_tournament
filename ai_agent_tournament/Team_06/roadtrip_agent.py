"""European Road Trip agent. Team 06.

Shortest road route Madrid -> 24 required cities -> Madrid, with no city
visited twice (intermediate cities included).

Course material used:
    Session 2    PEAS and task-environment properties          section 1
    Session 3    problem formulation, state-space graph        sections 1, 3, 4
    Session 4    Node, expand, frontier/reached, UCS           sections 5, 6
    Session 5    A*, admissible and consistent heuristic       sections 2, 6
    Session 6    hill climbing, simulated annealing, restarts  section 7
    Session 10   complete-but-inconsistent -> repair conflicts section 8
"""
import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from itertools import count


# ============================================================
# 0. SETTINGS
# ============================================================

STRATEGY = "improved"      # "baseline" = first strategy, kept for the comparison
SAFETY_MARGIN_S = 0.5      # handbook: keep about 0.5 s in reserve on Colab
RANDOM_SEED = 6
EARTH_RADIUS_KM = 6371.0


# ============================================================
# 1. THE AGENT AND ITS TASK ENVIRONMENT (Sessions 2-3)
# ============================================================
#
# PEAS
#   P  Performance:  km of our best LEGAL route submitted before the limit
#                    (no legal route = the challenge is lost)
#   E  Environment:  road map of 105 cities / 262 roads, the 25 required
#                    cities, the judge and its clock
#   A  Actuators:    submit(route)
#   S  Sensors:      the problem dictionary: cities, coordinates, roads,
#                    required cities, time limit
#
# Environment properties
#   Fully observable   the whole map and the required cities are given
#   Single-agent       the rival cannot change our route; it only matters
#                      when the two km are compared
#   Deterministic      a road always leads to the same city, same km
#   Sequential         each leg decides which cities are left for later legs
#   Semi-dynamic       the map never changes, but the clock keeps running
#   Discrete           cities and roads
#   Known              the rules and the costs are published
#
# Agent type: utility-based problem-solving agent. It does not stop at the
# first legal route (goal-based); it keeps searching for a SHORTER one and
# submits every improvement (anytime behaviour).
#
# Problem formulation, two levels:
#
# (a) Path between two cities  -> UCS / A*
#       State:        current city
#       Initial:      city A
#       Actions:      drive to a neighbouring city that is not blocked
#       Transition:   RESULT(s, GoTo(c)) = c
#       Goal test:    state == city B
#       Action cost:  road km
#
# (b) Visiting order of the required cities  -> local search
#       State:        complete tour = ordering of the 25 required cities,
#                     Madrid first, closing back to Madrid
#       Neighbours:   2-opt (reverse a segment), Or-opt (move 1-3 cities),
#                     double-bridge (large jump)
#       Objective:    km of the LEGAL route the tour produces (minimise)
#
# Constraint: no city twice. Joining the cities by their shortest paths
# gives a COMPLETE route that may be INCONSISTENT (two legs share a city).
# Section 8 repairs those conflicts.
# ============================================================


# ============================================================
# 2. HEURISTIC: STRAIGHT-LINE (HAVERSINE) DISTANCE
# ============================================================
#
# h(n) = great-circle km from n to the goal. Same unit as g(n) (km).
#
# Admissible: every road is at least as long as the straight line between
#   its two ends, so any road path n -> goal is at least h(n):
#       h(n) <= h*(n)
# Consistent: triangle inequality on the sphere:
#       h(n) <= c(n, n') + h(n')
#   so f(n) never decreases along a path.
# ============================================================

def straight_line_km(a, b):
    """a and b are entries of problem["cities"]. Formula from the handbook."""
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"]))
         * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


# ============================================================
# 3. STATE-SPACE GRAPH
# ============================================================

def build_graph(roads):
    """{city: {neighbour: km}}. Every road can be driven both ways."""
    graph = {}
    for road in roads:
        city_a, city_b, km = road["from"], road["to"], road["km"]
        graph.setdefault(city_a, {})[city_b] = km
        if road.get("bidirectional", True):
            graph.setdefault(city_b, {})[city_a] = km
    return graph


# ============================================================
# 4. SEARCH PROBLEM
# ============================================================
#
# 'blocked' removes cities from ACTIONS(s): this is how a leg is forced
# to avoid the cities already used by the route.
# goal_state=None: no goal test succeeds, so the search reaches every
# city (used to compute shortest paths from one city to all others).
# ============================================================

class RouteProblem:

    def __init__(self, graph, initial_state, goal_state=None, blocked=frozenset()):
        self.graph = graph
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.blocked = blocked

    def actions(self, state):
        return [city for city in self.graph[state]
                if city not in self.blocked or city == self.goal_state]

    def result(self, state, action):
        return action

    def is_goal(self, state):
        return state == self.goal_state

    def action_cost(self, state, action, next_state):
        return self.graph[state][next_state]


# ============================================================
# 5. SEARCH NODE, EXPAND, SOLUTION PATH
# ============================================================

@dataclass
class Node:
    state: str
    parent: object = None
    action: object = None
    path_cost: float = 0
    depth: int = 0


def expand(problem, node):
    children = []
    for action in sorted(problem.actions(node.state)):
        next_state = problem.result(node.state, action)
        step_cost = problem.action_cost(node.state, action, next_state)
        children.append(Node(state=next_state, parent=node, action=action,
                             path_cost=node.path_cost + step_cost,
                             depth=node.depth + 1))
    return children


def solution_path(node):
    path = []
    while node is not None:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))


# ============================================================
# 6. BEST-FIRST SEARCH: UCS AND A*
# ============================================================
#
#   UCS:  priority = g(n)          "What has been cheapest so far?"
#   A*:   priority = g(n) + h(n)   "Which node looks cheapest overall?"
#
# reached[state] = best node found for that state.
# An older, more expensive copy of a state can still be in the frontier:
# it is skipped when popped.
#
# IMPORTANT: the goal test is done when a node is POPPED. Generating the
# goal does not prove it is the cheapest route.
#
# A node is EXPANDED when it is removed from the frontier to generate its
# children (the judge's definition), so the popped goal is not counted.
#
# UCS is optimal because every road has positive km.
# A* is optimal because h is admissible, and since h is also consistent
# it only expands nodes with f(n) <= C*, a subset of the nodes with
# g(n) <= C* that UCS expands.
# ============================================================

def best_first_search(problem, priority):
    """Returns (goal node or None, expanded_nodes, max_frontier, reached)."""
    root = Node(state=problem.initial_state, path_cost=0)
    tie_breaker = count()
    frontier = [(priority(root), next(tie_breaker), root)]
    reached = {root.state: root}
    expanded_nodes = 0
    max_frontier = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)

        if reached[node.state] is not node:
            continue

        if problem.is_goal(node.state):
            return node, expanded_nodes, max_frontier, reached

        expanded_nodes += 1
        for child in expand(problem, node):
            if (child.state not in reached
                    or child.path_cost < reached[child.state].path_cost):
                reached[child.state] = child
                heapq.heappush(frontier, (priority(child), next(tie_breaker), child))
        max_frontier = max(max_frontier, len(frontier))

    return None, expanded_nodes, max_frontier, reached


def make_result(node, expanded_nodes, max_frontier):
    if node is None:
        return {"path": [], "km": math.inf,
                "expanded": expanded_nodes, "max_frontier": max_frontier}
    return {"path": solution_path(node), "km": node.path_cost,
            "expanded": expanded_nodes, "max_frontier": max_frontier}


def ucs(graph, start, goal):
    """Uniform-cost search. graph = {"cities": ..., "roads": ...}."""
    problem = RouteProblem(build_graph(graph["roads"]), start, goal)
    node, expanded, max_frontier, _ = best_first_search(
        problem, priority=lambda n: n.path_cost)
    return make_result(node, expanded, max_frontier)


def astar(graph, start, goal):
    """A* with h(n) = straight-line km from n to the goal."""
    cities = graph["cities"]
    problem = RouteProblem(build_graph(graph["roads"]), start, goal)
    h = {city: straight_line_km(cities[city], cities[goal]) for city in problem.graph}
    node, expanded, max_frontier, _ = best_first_search(
        problem, priority=lambda n: n.path_cost + h[n.state])
    return make_result(node, expanded, max_frontier)


def ucs_all(graph, start):
    """UCS without a goal (Dijkstra): cheapest node for EVERY city."""
    _, _, _, reached = best_first_search(RouteProblem(graph, start),
                                         priority=lambda n: n.path_cost)
    return reached


def astar_avoiding(graph, start, goal, blocked, h):
    """A* that never enters a blocked city. h[city] = straight-line km to goal.
    Returns (km, path), or None if the blocked cities cut start from goal."""
    problem = RouteProblem(graph, start, goal, blocked)
    node, _, _, _ = best_first_search(problem, priority=lambda n: n.path_cost + h[n.state])
    if node is None:
        return None
    return node.path_cost, solution_path(node)


# ============================================================
# 7. LOCAL SEARCH ON THE VISITING ORDER (Session 6)
# ============================================================
#
# State:      order = indices into a distance matrix; order[0] is the
#             start and never moves
# Objective:  closed tour length (lower is better)
#
# HILL CLIMBING: apply improving 2-opt / Or-opt moves until no neighbour
# is better -> a local minimum, not necessarily the global one.
#
# Escaping local minima:
#   SIMULATED ANNEALING  a worse neighbour is accepted with
#                            P(accept) = e^(-dE / T)
#                        dE = how much longer it is, T = temperature.
#                        T <- alpha * T, so the search becomes greedier.
#   RANDOM RESTART       when T is cold, or nothing improved for a long
#                        time, go back to the best state and reheat.
#   DOUBLE-BRIDGE        a jump that 2-opt cannot undo in one move.
# ============================================================

def tour_length(order, matrix):
    total = matrix[order[-1]][order[0]]
    for i in range(len(order) - 1):
        total += matrix[order[i]][order[i + 1]]
    return total


def hill_climbing(order, matrix):
    order = list(order)
    n = len(order)
    improved = True
    while improved:
        improved = False

        # 2-opt: replace roads (a,b) and (c,d) by (a,c) and (b,d)
        # by reversing the segment b..c
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[(j + 1) % n]
                delta = matrix[a][c] + matrix[b][d] - matrix[a][b] - matrix[c][d]
                if delta < -1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True

        # Or-opt: take out a block of 1-3 cities and put it back at the
        # best position, in either direction
        for size in (1, 2, 3):
            i = 1
            while i + size <= n:
                before, first = order[i - 1], order[i]
                last, after = order[i + size - 1], order[(i + size) % n]
                saving = (matrix[before][first] + matrix[last][after]
                          - matrix[before][after])
                rest = order[:i] + order[i + size:]
                best_delta, best_k, best_reversed = -1e-9, None, False
                for k in range(len(rest)):
                    p, q = rest[k], rest[(k + 1) % len(rest)]
                    gap = matrix[p][q] + saving
                    forward = matrix[p][first] + matrix[last][q] - gap
                    backward = matrix[p][last] + matrix[first][q] - gap
                    if forward < best_delta:
                        best_delta, best_k, best_reversed = forward, k, False
                    if backward < best_delta:
                        best_delta, best_k, best_reversed = backward, k, True
                if best_k is not None:
                    block = order[i:i + size]
                    if best_reversed:
                        block.reverse()
                    order = rest[:best_k + 1] + block + rest[best_k + 1:]
                    improved = True
                i += 1
    return order


def reverse_segment(order, rng):
    i, j = sorted(rng.sample(range(1, len(order)), 2))
    return order[:i] + order[i:j + 1][::-1] + order[j + 1:]


def move_city(order, rng):
    i = rng.randrange(1, len(order))
    rest = order[:i] + order[i + 1:]
    k = rng.randrange(1, len(order))
    return rest[:k] + [order[i]] + rest[k:]


def double_bridge(order, rng):
    """A B C D -> A C B D."""
    if len(order) < 8:
        return reverse_segment(order, rng)
    i, j, k = sorted(rng.sample(range(1, len(order)), 3))
    return order[:i] + order[j:k] + order[i:j] + order[k:]


RANDOM_NEIGHBOUR = (double_bridge, reverse_segment, move_city)


def accept(delta, temperature, rng):
    """Simulated-annealing acceptance rule, for minimisation."""
    if delta <= 0:
        return True
    if delta == math.inf:
        return False
    return rng.random() < math.exp(-delta / temperature)


def local_search(tour, distance, time_limit_s):
    """Simulated annealing + hill climbing + random restarts.

    tour      different city ids; tour[0] stays first and is not repeated
    distance  {a: {b: km}} for every pair of cities in the tour
    Returns a reordering of tour that is never longer than the input.
    """
    n = len(tour)
    if n <= 3:
        return list(tour)
    deadline = time.monotonic() + max(0.05, min(0.8 * time_limit_s, time_limit_s - 0.3))
    matrix = [[0.0 if a == b else distance[a][b] for b in tour] for a in tour]
    rng = random.Random(RANDOM_SEED)

    current = hill_climbing(list(range(n)), matrix)
    current_len = tour_length(current, matrix)
    best, best_len = current[:], current_len
    start_temperature = 0.05 * current_len / n
    temperature = start_temperature

    while time.monotonic() < deadline:
        candidate = hill_climbing(rng.choice(RANDOM_NEIGHBOUR)(current, rng), matrix)
        candidate_len = tour_length(candidate, matrix)
        if accept(candidate_len - current_len, temperature, rng):
            current, current_len = candidate, candidate_len
            if current_len < best_len - 1e-9:
                best, best_len = current[:], current_len
        temperature *= 0.995
        if temperature < 0.01 * start_temperature:
            current, current_len = best[:], best_len
            temperature = start_temperature

    return [tour[i] for i in best]


# ============================================================
# 8. FROM A TOUR TO A LEGAL ROUTE (Session 10: repair conflicts)
# ============================================================

def route_km(graph, route):
    return sum(graph[a][b] for a, b in zip(route, route[1:]))


def is_legal(route, start, required, graph):
    """Same checks as the judge: closed at the start, real roads only,
    no repeated city, every required city visited."""
    return (len(route) >= 2 and route[0] == start and route[-1] == start
            and all(b in graph[a] for a, b in zip(route, route[1:]))
            and len(set(route[:-1])) == len(route) - 1
            and set(required) <= set(route))


class RoadTripSolver:

    def __init__(self, problem):
        self.start = problem["start"]
        self.cities = problem["cities"]
        self.graph = build_graph(problem["roads"])
        for city in self.cities:
            self.graph.setdefault(city, {})
        others = [c for c in problem["required"] if c != self.start]
        self.names = [self.start] + others            # index 0 = Madrid
        self.n = len(self.names)
        self.required = set(self.names)
        self.index = {name: i for i, name in enumerate(self.names)}

        # h(n) towards every required city, computed once
        self.h_to = {goal: {city: straight_line_km(self.cities[city], self.cities[goal])
                            for city in self.graph}
                     for goal in self.names}

        # UCS from every required city: 25 x 25 shortest km + their paths
        self.matrix = [[0] * self.n for _ in range(self.n)]
        self.paths = [[None] * self.n for _ in range(self.n)]
        for i, a in enumerate(self.names):
            reached = ucs_all(self.graph, a)
            for j, b in enumerate(self.names):
                if i != j:
                    node = reached.get(b)
                    self.matrix[i][j] = node.path_cost if node else math.inf
                    self.paths[i][j] = solution_path(node) if node else None
        self.cache = {}

    def legal_route(self, order):
        """Build the route leg by leg.

        A leg keeps its shortest path unless that path CONFLICTS with the
        route: it passes through a city already used, or through a
        required city that is still to come. Then the conflict is repaired
        with A*, blocking those cities. None if a leg cannot be repaired.
        """
        names = self.names
        route = [names[order[0]]]
        used = {route[0]}
        to_come = {names[i] for i in order[1:]}
        n = len(order)
        for k in range(n):
            a, b = order[k], order[(k + 1) % n]
            target = names[b]
            to_come.discard(target)
            path = self.paths[a][b]
            conflict = path is None or any(c in used or c in to_come for c in path[1:-1])
            if conflict:
                repaired = astar_avoiding(self.graph, names[a], target,
                                          used | to_come, self.h_to[target])
                if repaired is None:
                    return None
                path = repaired[1]
            route.extend(path[1:])
            used.update(path[1:])
        return route

    def evaluate(self, order):
        """Objective of a tour: km of its legal route, trying both directions
        (the repair is not symmetric). math.inf if neither can be repaired."""
        key = tuple(order)
        if key in self.cache:
            return self.cache[key]
        best = (math.inf, None)
        for candidate in (order, [order[0]] + order[1:][::-1]):
            route = self.legal_route(candidate)
            if route is not None:
                km = route_km(self.graph, route)
                if km < best[0]:
                    best = (km, route)
        if len(self.cache) > 200000:
            self.cache.clear()
        self.cache[key] = best
        return best

    def improve_legs(self, route):
        """Hill climbing on the legs of a legal route: re-plan one leg with A*,
        blocking every city of the other legs; keep it if it is shorter."""
        improved = True
        while improved:
            improved = False
            stops = [i for i, city in enumerate(route) if city in self.required]
            for s, e in zip(stops, stops[1:]):
                leg = route[s:e + 1]
                blocked = (set(route[:s]) | set(route[e + 1:])) - {leg[-1]}
                found = astar_avoiding(self.graph, leg[0], leg[-1], blocked,
                                       self.h_to[leg[-1]])
                if found is not None and found[0] < route_km(self.graph, leg):
                    route = route[:s] + found[1] + route[e + 1:]
                    improved = True
                    break
        return route

    def legal_hill_climbing(self, order, km, deadline):
        """Hill climbing on the REAL objective: Or-opt and 2-opt neighbours of
        the tour, each scored by the km of its legal route. First improvement
        is taken. Stops at a local minimum or at the deadline."""
        n = len(order)
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False

            # Or-opt: move a block of 1-3 cities, in either direction
            for size in (1, 2, 3):
                for i in range(1, n - size + 1):
                    if time.monotonic() > deadline:
                        return order, km
                    block = order[i:i + size]
                    rest = order[:i] + order[i + size:]
                    for k in range(1, len(rest) + 1):
                        for piece in (block, block[::-1]):
                            neighbour = rest[:k] + piece + rest[k:]
                            neighbour_km = self.evaluate(neighbour)[0]
                            if neighbour_km < km:
                                order, km, improved = neighbour, neighbour_km, True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

            # 2-opt: reverse a segment
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    neighbour = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    neighbour_km = self.evaluate(neighbour)[0]
                    if neighbour_km < km:
                        order, km, improved = neighbour, neighbour_km, True
                        break
                if improved:
                    break
        return order, km

    def order_of(self, route):
        order, seen = [], set()
        for city in route[:-1]:
            if city in self.index and city not in seen:
                order.append(self.index[city])
                seen.add(city)
        return order

    # --------------------------------------------------------
    # Fallback, adapted from the provided template: insert the
    # cities one by one in random order, each with the detour of
    # fewest roads (BFS) that avoids the cities already used.
    # Long routes, but a legal one is found on every draw.
    # --------------------------------------------------------

    def fallback_route(self, rng, deadline):
        while time.monotonic() < deadline:
            others = self.names[1:]
            rng.shuffle(others)
            route = [self.start, self.start]
            for city in others:
                if city not in route:
                    route = self._insert(route, city, rng)
                    if route is None:
                        break
            if route is not None and is_legal(route, self.start, self.required, self.graph):
                return route
        return None

    def _insert(self, route, city, rng):
        used = set(route)
        positions = list(range(len(route) - 1))
        rng.shuffle(positions)
        for i in positions:
            there = self._bfs(route[i], city, used)
            if there is not None:
                back = self._bfs(city, route[i + 1], used | set(there))
                if back is not None:
                    return route[:i] + there + back[1:] + route[i + 2:]
        return None

    def _bfs(self, start, goal, used):
        previous = {start: None}
        frontier = deque([start])
        while frontier:
            city = frontier.popleft()
            if city == goal:
                path = [goal]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return path[::-1]
            for nxt in self.graph[city]:
                if nxt not in previous and (nxt not in used or nxt == goal):
                    previous[nxt] = city
                    frontier.append(nxt)
        return None


# ============================================================
# 9. SUBMITTING
# ============================================================

class Submitter:
    """Calls submit() only for a legal route shorter than the best so far."""

    def __init__(self, solver, submit):
        self.solver = solver
        self.submit = submit
        self.best_km = math.inf
        self.best_route = None

    def offer(self, route):
        s = self.solver
        if route is None or not is_legal(route, s.start, s.required, s.graph):
            return False
        km = route_km(s.graph, route)
        if km < self.best_km:
            self.best_km, self.best_route = km, route
            self.submit(route)
            return True
        return False


def nearest_neighbour_order(matrix):
    order, left = [0], set(range(1, len(matrix)))
    while left:
        nxt = min(left, key=lambda j: (matrix[order[-1]][j], j))
        order.append(nxt)
        left.remove(nxt)
    return order


# ============================================================
# 10. THE AGENT
# ============================================================

def solve_baseline(problem, submit):
    """FIRST STRATEGY: optimise on the distance table, repair once.
    Nearest-neighbour tour + hill climbing on TABLE km, then one repair
    into a legal route (fallback if the repair fails)."""
    t0 = time.monotonic()
    limit = problem["time_limit_s"]
    deadline = t0 + max(0.1, limit - SAFETY_MARGIN_S)
    solver = RoadTripSolver(problem)
    out = Submitter(solver, submit)
    order = hill_climbing(nearest_neighbour_order(solver.matrix), solver.matrix)
    out.offer(solver.legal_route(order))
    if out.best_route is None:
        out.offer(solver.fallback_route(random.Random(RANDOM_SEED), deadline))


def solve_improved(problem, submit):
    """IMPROVED STRATEGY: local search on the tour where the objective is
    the km of the REPAIRED (legal) route, not the table km.

    Why: a tour that is short on the table can need long detours once
    its conflicts are repaired. Measured on one test draw: a tour of
    13,091 table km became 13,410 legal km, while a tour of 13,252
    table km became 13,275 legal km.
    """
    t0 = time.monotonic()
    limit = problem["time_limit_s"]
    deadline = t0 + max(0.1, min(0.9 * limit, limit - SAFETY_MARGIN_S))
    rng = random.Random(RANDOM_SEED)
    solver = RoadTripSolver(problem)
    out = Submitter(solver, submit)

    def climb_from_best():
        """Legal hill climbing from the best route; submits what it finds."""
        order, km = solver.legal_hill_climbing(solver.order_of(out.best_route),
                                               out.best_km, deadline)
        if km < out.best_km:
            out.offer(solver.improve_legs(solver.evaluate(order)[1]))
        return solver.order_of(out.best_route), out.best_km

    # ---- 1. a legal route as early as possible --------------
    start_order = hill_climbing(nearest_neighbour_order(solver.matrix), solver.matrix)
    out.offer(solver.evaluate(start_order)[1])
    if out.best_route is None:
        out.offer(solver.fallback_route(rng, min(deadline, time.monotonic() + 1.0)))
    if out.best_route is not None:
        out.offer(solver.improve_legs(out.best_route))

    # ---- 2. simulated annealing + random restarts ------------
    # (legal hill climbing is NOT done here: starting it from the
    # table-optimal tour anchors the search near that tour; tested slower)
    if out.best_route is not None:
        current, current_km = solver.order_of(out.best_route), out.best_km
    else:
        current, current_km = start_order, math.inf

    start_temperature, alpha = 400.0, 0.9993
    temperature = start_temperature
    without_improvement = 0

    while time.monotonic() < deadline:
        if out.best_route is None:
            out.offer(solver.fallback_route(rng, min(deadline, time.monotonic() + 0.3)))
            if out.best_route is not None:
                current, current_km = solver.order_of(out.best_route), out.best_km
            continue

        # random neighbour; only 10% of the time also hill climbing on the
        # table, because the table pulls back towards tours that are short
        # on the table but long once made legal
        candidate = rng.choice(RANDOM_NEIGHBOUR)(current, rng)
        if rng.random() < 0.1:
            candidate = hill_climbing(candidate, solver.matrix)
        km, route = solver.evaluate(candidate)

        if km < out.best_km:
            out.offer(solver.improve_legs(route))
            without_improvement = 0
        else:
            without_improvement += 1

        if current_km == math.inf or accept(km - current_km, temperature, rng):
            current, current_km = candidate, km
        temperature = max(2.0, temperature * alpha)

        # random restart: back to the best route, climb, reheat
        if without_improvement > 1500:
            current, current_km = climb_from_best()
            temperature = 150.0
            without_improvement = 0


def solve(problem, submit):
    if STRATEGY == "baseline":
        solve_baseline(problem, submit)
    else:
        solve_improved(problem, submit)
