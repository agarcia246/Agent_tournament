# Team 06

## Members

- Alex Garcia
- Alejandro Zapata
- Alesia Dako
- Kohana Rakipi
- Viktoriia Stepanenko
- Laura Somogyi

## Algorithms implemented

| Challenge | Algorithms | Where in the file |
|---|---|---|
| Road Trip | Uniform-cost search, A* with a straight-line heuristic, conflict repair with A*, hill climbing (2-opt, Or-opt), simulated annealing with random restarts | `roadtrip_agent.py`: sections 4–6 search, 7 local search, 8 legal route and repair, 10 agent |
| Chess | | |
| Battleship | | |
| Catan | | |

## Training

Road Trip: no training. The agent plans from scratch on every draw, within the time limit.

## Tests

**Road Trip**

`check_requirements()` output:
```
PASS  Road Trip  Uniform-cost search / Dijkstra     optimal on 8 city pairs, 384 nodes expanded
PASS  Road Trip  A* with a geographic heuristic     optimal on 8 city pairs, 176 nodes expanded
PASS  Road Trip  A* expands no more nodes than UCS  A* 176 vs UCS 384
PASS  Road Trip  Local search / optimization        tour improved from 16810 to 7480
```

| Test | Result |
|---|---|
| 30 races against `demo` | 30 wins. Mean route 12,917 km vs 15,919 km |
| 30 races against our first strategy | 18 wins, 12 draws, 0 losses. Mean 12,918 km vs 14,303 km |
| 1,000 random draws | A legal route found on every draw, within 0.05 s |
| 40 draws against the best route we ever found for each | Reached on 79 of 80 runs with 5 s. With only 1.5 s, to simulate a slower Colab machine, reached on 40 of 40 |
| All tests | 0 invalid submissions, 0 incidents |

## Dependencies and known limitations

Road Trip: Python standard library only.

## Technical explanation

### European Road Trip

#### Problem formulation

**Graph.** 105 cities are nodes and 262 two-way roads are edges, weighted by km. We store them as an adjacency dictionary `{city: {neighbour: km}}` (`build_graph`).

**Agent and environment.** Performance is the km of our best legal route submitted before the limit; no legal route means the challenge is lost. The environment is the map, the 25 required cities, the judge and its clock. The actuator is `submit(route)`, and the sensors are the problem dictionary.

The environment is fully observable, deterministic, discrete and known. It is sequential: each leg uses up cities that later legs cannot use. It is semi-dynamic: the map is fixed, but the clock runs. For the route itself it is single-agent, since the rival cannot change our route.

Ours is a utility-based agent. It does not stop at the first legal route; it keeps searching for a shorter one and submits every improvement.

**Formulation, at two levels:**

| | Path between two cities | Order of the required cities |
|---|---|---|
| State | Current city | A complete tour of the 25 required cities, starting at Madrid |
| Actions / neighbours | Drive to a neighbouring city that is not blocked | 2-opt, Or-opt (move 1–3 cities), double-bridge |
| Goal / objective | Reach the target city | Minimise the km of the **legal** route the tour produces |
| Cost | Road km | — |

**Constraint.** No city may appear twice, including cities we only drive through. Joining a tour by shortest paths gives a complete route that may be inconsistent, because two legs can pass through the same city. The agent repairs these conflicts.

#### Algorithms

**UCS and A\*** (sections 4–6). Both run the same `best_first_search` over a `RouteProblem`, with `Node`, `expand` and a `reached` table. The goal is tested when a node is popped from the frontier, not when it is generated.

- UCS orders the frontier by g(n). It is optimal because every road has positive km.
- A* orders the frontier by g(n) + h(n), where h(n) is the straight-line (Haversine) km to the goal.
- **Why straight-line distance is valid.** Every road is at least as long as the straight line between its two ends. By the triangle inequality, any road path from n to the goal is therefore at least h(n), so h never overestimates: it is *admissible*.
- It is also *consistent*: h(n) ≤ c(n, n') + h(n'). So A* returns the same optimal km as UCS while expanding fewer nodes: 176 vs 384 in the checks.

**How the agent uses them.** UCS without a goal, run from each required city, gives a 25 × 25 table of shortest distances and their paths. A* re-plans a leg when it must avoid cities.

**Valid route with no city twice** (section 8, `legal_route`). The route is built one leg at a time. A leg keeps its shortest path unless that path passes through a city already used, or a required city still to come. If it does, A* re-plans the leg with those cities blocked: they are removed from the actions. Both directions of the tour are tried. Every route is checked against the judge's rules (`is_legal`) before it is submitted.

**Optimisation of the visiting order** (sections 7 and 10). There are 24!/2 possible tours, far too many to enumerate, so we use local search.

- **Neighbourhood:** 2-opt (reverse a segment), Or-opt (move a block of 1–3 cities) and double-bridge (a large jump that 2-opt cannot undo).
- **Objective:** the km of the *legal* route, not the km on the distance table.
- **Acceptance rule (simulated annealing):** a better neighbour is always accepted; a worse one with probability e^(−ΔE/T). T starts at 400 and is multiplied by 0.9993 each step.
- **Escaping local optima:** besides annealing, after 1,500 steps without improvement we restart from the best route, run hill climbing on legal km (2-opt and Or-opt) down to a local minimum, and reheat.
- **Polishing:** every new best route has each leg re-planned with A* while avoiding the other legs (`improve_legs`).
- **Fallback:** adapted from the provided template. It inserts cities one by one with breadth-first detours, and guarantees a legal route on every draw.

**Why we chose this.** Local search gives a good route within milliseconds and keeps improving it, which suits a time limit. Optimising the *legal* km mattered most. A tour that is short on the table can become long once its conflicts are repaired: on one draw, a 13,091 km table tour became 13,410 legal km, while a 13,252 km table tour became only 13,275.

#### Baseline against improved version

- **First strategy** (`STRATEGY = "baseline"`): nearest-neighbour tour, hill climbing on table km, then one repair into a legal route.
- **Improved strategy:** the local search above, scored on legal km.

In 30 head-to-head races, the improved strategy won 18 and drew 12, with no losses. Its mean route was 12,918 km against 14,303 km. The draws are maps where the table route is already legal and optimal.

Tuning on 40 draws showed the same lesson. Measured as how often the agent reached the best route we ever found:

| Setting | Best route reached |
|---|---|
| Table hill climbing on 70% of steps | 35 of 40 draws |
| Table hill climbing on 10% of steps | 38 of 40 draws |
| 10%, plus hill climbing on legal km at each restart (final) | 79 of 80 runs |

#### Time and memory

The agent reads `time_limit_s` and stops at min(0.9 × limit, limit − 0.5 s), keeping the half-second reserve recommended for Colab.

A legal route is submitted within 0.05 s, and every improvement is submitted at once. If time runs short, the best route is already with the judge and nothing is lost. Each search step is short, and hill climbing on legal km checks the clock, so the agent never overruns. On every test draw the final route was found within 2 s.

Memory is small: a 25 × 25 table, and a cache of evaluated tours capped at 200,000 entries.

#### Limits

- Local search does not prove that the route is optimal.
- With more time, we would add an exact method (branch and bound) to prove optimality.

## Use of generative AI
