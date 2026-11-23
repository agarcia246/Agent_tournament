"""Battleship agent. Team: <your team name>

Start here: docs/students/README.md
Full rules: games/battleship/README.md
Examples of what you receive: docs/students/sample_observations/battleship_placement.json and battleship_turn.json

Ships can move one cell per turn until they take their first hit, so an old miss
proves nothing about where a ship is now. Cells are [row, col], with row 0 at the top.
North is row - 1 and east is col + 1.

Keep the function names and signatures below. Add any helpers you need.
"""
import random
import traceback

import numpy as np


def act(observation: dict):
    """Called once for placement, then once per turn. Module-level variables survive.

    Placement phase, observation["phase"] == "placement":
        observation["board_size"], observation["fleet"] = [{"ship", "length"}, ...]
        Return {"placements": [{"ship": "carrier", "row": 0, "col": 0, "orientation": "H"}, ...]}
        with one entry per ship, or {"placements": None} to let the judge place at random.

    Turn phase, observation["phase"] == "turn":
        observation["my_ships"]         [{"ship", "length", "orientation", "cells",
                                          "damaged_cells", "immobile", "sunk"}, ...]
        observation["legal_moves"]      [{"ship", "direction"}, ...]
        observation["my_shots"]         [{"turn", "target", "result"}, ...]
        observation["my_sonar"]         [{"turn", "center", "reading"}, ...]
        observation["sonar_remaining"]  uses left
        observation["enemy_ships_sunk"] lengths of enemy ships you have sunk
        observation["damage_received"]  hits your own fleet has taken
        observation["turn"], ["max_rounds"], ["time_remaining_s"]

        Return {"shot": [row, col]} or {"sonar": [row, col]}.
        To move a ship as well, add "move": <one entry of legal_moves>.
    """
    try:
        if observation["phase"] == "placement":
            return _placement(observation)
        return _turn(observation)
    except Exception:
        traceback.print_exc()
        return _safe_shot(observation)


RIVAL_MOVE_RATES = (0.0, 0.25, 1.0)
MULTI_HIT_BONUS = 20
SONAR_AFTER_MISSES = 6
TIME_RESERVE_S = 0.5

DIRECTIONS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

STATE = {}


def _ship_cells(row, col, orientation, length):
    if orientation == "H":
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def _inside(cell, size):
    return 0 <= cell[0] < size and 0 <= cell[1] < size


# placement 


def _placement(observation):
    """Random placement where no two ships touch, not even diagonally.

    Rivals search around their hits, so a ship touching a hit ship is easy to find.
    """
    STATE.clear()
    size = observation["board_size"]
    fleet = sorted(observation["fleet"], key=lambda s: -s["length"])
    for _ in range(500):
        placements, used = [], set()
        for ship in fleet:
            length = ship["length"]
            for _ in range(200):
                orientation = random.choice("HV")
                row = random.randrange(size - (length - 1 if orientation == "V" else 0))
                col = random.randrange(size - (length - 1 if orientation == "H" else 0))
                cells = _ship_cells(row, col, orientation, length)
                halo = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if not halo & used:
                    used.update(cells)
                    placements.append({"ship": ship["ship"], "row": row, "col": col, "orientation": orientation})
                    break
            else:
                break  # this ship found no room: start the whole fleet again
        if len(placements) == len(fleet):
            return {"placements": placements}
    return {"placements": None}  # the judge places at random


# Belief grids


def _zero_covering(belief, length, cell, size):
    """Rule out every placement of this ship that covers cell."""
    r, c = cell
    lo, hi = max(0, c - length + 1), min(c, size - length)
    if lo <= hi:
        belief["H"][r, lo : hi + 1] = 0.0
    lo, hi = max(0, r - length + 1), min(r, size - length)
    if lo <= hi:
        belief["V"][lo : hi + 1, c] = 0.0


def _legal_placements(length, size, hits):
    """1 where the ship's top-left cell may be: on the board and not overlapping a hit ship."""
    mask = {"H": np.zeros((size, size)), "V": np.zeros((size, size))}
    mask["H"][:, : size - length + 1] = 1.0
    mask["V"][: size - length + 1, :] = 1.0
    for cell in hits:
        _zero_covering(mask, length, cell, size)
    return mask


def _uniform_belief(length, size, hits):
    """Every legal placement equally likely."""
    belief = _legal_placements(length, size, hits)
    total = belief["H"].sum() + belief["V"].sum()
    return {o: grid / total for o, grid in belief.items()} if total else belief


def _normalize(belief, length, size, hits):
    total = belief["H"].sum() + belief["V"].sum()
    if total == 0:
        # The evidence contradicts our model (the rival moved more than we assumed),
        # so forget the old belief and start again from "anywhere legal".
        belief.update(_uniform_belief(length, size, hits))
        return
    belief["H"] /= total
    belief["V"] /= total


def _occupancy(state, model):
    """P(an enemy segment is on each cell) under one model, summed over the ships afloat."""
    size = state["size"]
    occ = np.zeros((size, size))
    for length, count in state["pool"].items():
        belief = model["beliefs"][length]
        span = size - length + 1
        for k in range(length):
            occ[:, k : k + span] += count * belief["H"][:, :span]
            occ[k : k + span, :] += count * belief["V"][:span, :]
    for cell in state["hits"]:
        occ[cell] = 0.0  # damaged cells are never worth shooting again
    return occ


def _model_weights(state):
    """P(model | everything we have seen), from the log-likelihoods of past shots."""
    logs = np.array([m["log_weight"] for m in state["models"]])
    weights = np.exp(logs - logs.max())
    return weights / weights.sum()


# The filter: update with evidence, predict the rival's move


def _new_state(observation):
    size = observation["board_size"]
    pool = {}
    for ship in observation["my_ships"]:  # the enemy fleet is the same as ours
        pool[ship["length"]] = pool.get(ship["length"], 0) + 1
    STATE.clear()
    STATE.update(
        size=size,
        pool=pool,  # {length: how many enemy ships of that length are still afloat}
        models=[  # one belief filter per guess of how often the rival moves
            {"rate": rate, "log_weight": 0.0, "beliefs": {length: _uniform_belief(length, size, {}) for length in pool}}
            for rate in RIVAL_MOVE_RATES
        ],
        pending=None,  # what each model predicted for our last hunting shot
        hits={},  # {cell: turn it was hit}
        misses=[],  # [(cell, turn)]
        resolved=set(),  # hit cells that belong to ships we already sank
        shots_seen=0,
        sonar_seen=0,
        last_action_turn=0,
        misses_in_a_row=0,
    )
    return STATE


def _apply_shot(state, shot):
    if shot.get("already_damaged"):
        return
    cell, size = tuple(shot["target"]), state["size"]
    hit = shot["result"] in ("HIT", "SUNK")

    # Score each movement model on how well it predicted this hunting shot (Bayes' rule
    # over models: weight *= P(result | model)).
    pending = state["pending"]
    if pending is not None and pending["cell"] == cell:
        for model, p in zip(state["models"], pending["p_hit"]):
            p = min(max(p, 0.01), 0.99)
            model["log_weight"] += np.log(p if hit else 1 - p)
    state["pending"] = None

    if hit:
        state["hits"][cell] = shot["turn"]
        state["misses_in_a_row"] = 0
    else:
        state["misses"].append((cell, shot["turn"]))
        state["misses_in_a_row"] += 1
    # A miss: no unhit ship covered this cell just now. A hit: unhit ships cannot
    # overlap a damaged one either. Both rule out every placement covering the cell.
    for model in state["models"]:
        for length, belief in model["beliefs"].items():
            _zero_covering(belief, length, cell, size)
            _normalize(belief, length, size, state["hits"])
    if shot["result"] == "SUNK":
        _resolve_sunk(state, cell, shot["sunk_length"])


def _resolve_sunk(state, cell, length):
    """Mark which hit cells made up the ship we just sank, and drop it from the pool."""
    open_hits = set(state["hits"]) - state["resolved"]
    r, c = cell
    lines = [_ship_cells(r, c - k, "H", length) for k in range(length)]
    lines += [_ship_cells(r - k, c, "V", length) for k in range(length)]
    sunk = next((line for line in lines if all(x in open_hits for x in line)), [cell])
    state["resolved"].update(sunk)
    if state["pool"].get(length, 0) > 0:
        state["pool"][length] -= 1
        if state["pool"][length] == 0:
            del state["pool"][length]
            for model in state["models"]:
                del model["beliefs"][length]


def _apply_sonar(state, model, scan, correct):
    """Bayes' rule on every placement of every ship, treating the ships as independent.

    The reading counts all enemy segments in the square: damaged ones we already know
    about, plus the unknown ships. For one ship, P(reading | its placement) sums over how
    many segments the OTHER ships put in the square.
    """
    size = state["size"]
    r0, c0 = scan["center"]
    region = np.zeros((size, size))
    region[max(0, r0 - 1) : r0 + 2, max(0, c0 - 1) : c0 + 2] = 1.0
    cells = int(region.sum())
    fixed = int(sum(region[cell] for cell in state["hits"]))

    # in_region[length][o][r, c]: segments that placement puts inside the square.
    # dists[length][k]: P(this ship puts k segments inside the square).
    in_region, dists = {}, {}
    for length, belief in model["beliefs"].items():
        in_region[length] = {
            "H": sum(_shift(region, 0, k) for k in range(length)).astype(int),
            "V": sum(_shift(region, k, 0) for k in range(length)).astype(int),
        }
        dists[length] = sum(
            np.bincount(in_region[length][o].ravel(), weights=belief[o].ravel(), minlength=length + 1) for o in "HV"
        )

    ships = [length for length, count in state["pool"].items() for _ in range(count)]
    for length, belief in model["beliefs"].items():
        others = np.array([1.0])
        rest = list(ships)
        rest.remove(length)
        for other in rest:
            others = np.convolve(others, dists[other])
        likelihood = np.array([
            sum(p * _sonar_likelihood(scan["reading"], fixed + k + m, cells, correct) for m, p in enumerate(others))
            for k in range(length + 1)
        ])
        for o in "HV":
            belief[o] *= likelihood[in_region[length][o]]
        _normalize(belief, length, size, state["hits"])


def _predict(state, model, rival_actions):
    """The rival had its turn: each unhit ship may have moved one cell.

    Only one rival ship moves per turn, so with this model's move rate r each of our
    n candidate ships moves with probability r / n. Ships that were hit cannot move,
    and are not in these grids anyway.
    """
    open_hits = set(state["hits"]) - state["resolved"]
    mobile = sum(state["pool"].values()) - (1 if open_hits else 0)
    if mobile <= 0 or rival_actions <= 0 or model["rate"] <= 0:
        return
    p = min(1.0, model["rate"] / mobile)
    for length, belief in model["beliefs"].items():
        allowed = _legal_placements(length, state["size"], state["hits"])
        for _ in range(rival_actions):
            for o in "HV":
                belief[o] = _predict_grid(belief[o], allowed[o], p)
        _normalize(belief, length, state["size"], state["hits"])


def _shift(grid, dr, dc):
    """out[r, c] = grid[r + dr, c + dc], with zeros where that falls off the board."""
    size = grid.shape[0]
    out = np.zeros_like(grid)
    rows_out = slice(max(0, -dr), min(size, size - dr))
    cols_out = slice(max(0, -dc), min(size, size - dc))
    rows_in = slice(max(0, dr), min(size, size + dr))
    cols_in = slice(max(0, dc), min(size, size + dc))
    out[rows_out, cols_out] = grid[rows_in, cols_in]
    return out


def _predict_grid(grid, allowed, p):
    """The transition model of predict_ship_belief, vectorised with numpy, with one addition:
    a ship cannot move into a hit ship either, exactly like the real movement rules.

    With no hit ships on the board, allowed is "fits on the board" and this returns the same
    grid as predict_ship_belief.
    """
    options = sum(_shift(allowed, dr, dc) for dr, dc in DIRECTIONS.values())  # legal moves from each cell
    can_move = options > 0
    new = np.where(can_move, 1 - p, 1.0) * grid  # the part that stays put
    share = np.where(can_move, p * grid / np.maximum(options, 1), 0.0)  # sent to each legal neighbour
    for dr, dc in DIRECTIONS.values():
        new += allowed * _shift(share, -dr, -dc)  # cell x receives from x - direction
    return new


# Choosing an action 


def _target_cell(state):
    """After a hit, list every placement that explains it and shoot the busiest cell.

    A hit ship is frozen, so a miss fired AFTER the hit is permanent proof for that ship.
    Placements that also cover other open hits are much more likely (MULTI_HIT_BONUS),
    which makes us follow the line of a ship once we have two hits on it.
    """
    size = state["size"]
    open_hits = set(state["hits"]) - state["resolved"]
    if not open_hits:
        return None
    scores = np.zeros((size, size))
    seen = set()
    for hit in open_hits:
        hit_turn = state["hits"][hit]
        blocked = [list(c) for c, t in state["misses"] if t > hit_turn] + [list(c) for c in state["resolved"]]
        for length, count in state["pool"].items():
            for row, col, orientation in consistent_placements(length, size, blocked, [list(hit)]):
                if (length, row, col, orientation) in seen:
                    continue
                seen.add((length, row, col, orientation))
                cells = _ship_cells(row, col, orientation, length)
                weight = count * MULTI_HIT_BONUS ** (sum(c in open_hits for c in cells) - 1)
                for c in cells:
                    if c not in state["hits"]:
                        scores[c] += weight
    return _pick_best(scores) if scores.max() > 0 else None


def _pick_best(scores):
    """The highest-scoring cell, breaking ties at random."""
    choices = np.argwhere(scores >= scores.max() - 1e-12)
    r, c = random.choice(choices)
    return [int(r), int(c)]


def _sonar_center(state, occupancy):
    """The 3x3 square where we expect the most unknown enemy segments."""
    size = state["size"]
    squares = {
        (r, c): occupancy[r - 1 : r + 2, c - 1 : c + 2].sum() for r in range(1, size - 1) for c in range(1, size - 1)
    }
    return list(max(squares, key=squares.get))


def _choose_move(observation):
    """Move one unhit ship every turn, so the rival's old misses stop being true.

    If we have been hit, prefer to step away from the damage: rivals search around hits.
    """
    moves = observation["legal_moves"]
    if not moves:
        return None
    damage = [tuple(d["cell"]) for d in observation["damage_received"]]
    if not damage:
        return random.choice(moves)
    ships = {s["ship"]: s["cells"] for s in observation["my_ships"]}

    def distance(cells):
        return min(abs(r - hr) + abs(c - hc) for r, c in cells for hr, hc in damage)

    def score(move):
        dr, dc = DIRECTIONS[move["direction"]]
        before = distance(ships[move["ship"]])
        after = distance([(r + dr, c + dc) for r, c in ships[move["ship"]]])
        gain = after - before if before <= 2 else 0  # only ships close to the damage care
        return gain, random.random()

    return max(moves, key=score)


def _turn(observation):
    state = STATE if STATE.get("size") else _new_state(observation)
    time_left = observation.get("time_remaining_s")
    if time_left is not None and time_left < TIME_RESERVE_S:
        return _safe_shot(observation)

    # 1. UPDATE: fold in what our last action revealed (the belief is still "as of then").
    correct = observation.get("sonar_correct_probability", 0.8)
    for shot in observation["my_shots"][state["shots_seen"] :]:
        _apply_shot(state, shot)
    for scan in observation["my_sonar"][state["sonar_seen"] :]:
        for model in state["models"]:
            _apply_sonar(state, model, scan, correct)
    state["shots_seen"] = len(observation["my_shots"])
    state["sonar_seen"] = len(observation["my_sonar"])

    # 2. PREDICT: the rival has acted since our last action (once, or 0/1 on our first turn).
    for model in state["models"]:
        _predict(state, model, observation["turn"] - state["last_action_turn"])
    state["last_action_turn"] = observation["turn"] + 1

    # 3. DECIDE.
    move = _choose_move(observation)
    target = _target_cell(state)
    if target is not None:
        return {"move": move, "shot": target}

    per_model = [_occupancy(state, model) for model in state["models"]]
    weights = _model_weights(state)
    occupancy = sum(w * occ for w, occ in zip(weights, per_model))
    if observation["sonar_remaining"] > 0 and state["misses_in_a_row"] >= SONAR_AFTER_MISSES:
        state["misses_in_a_row"] = 0
        return {"move": move, "sonar": _sonar_center(state, occupancy)}

    shot = _pick_best(occupancy)
    # Remember what each model expected, to score the models when the result comes back.
    state["pending"] = {"cell": tuple(shot), "p_hit": [float(occ[tuple(shot)]) for occ in per_model]}
    if observation["turn"] % 40 < 2:
        rates = ", ".join(f"{m['rate']}: {w:.2f}" for m, w in zip(state["models"], weights))
        print(f"turn {observation['turn']}: belief in each rival move rate  {rates}")
    return {"move": move, "shot": shot}


def _safe_shot(observation):
    """Cheap fallback: next to a hit if there is one, else any undamaged cell."""
    size = observation["board_size"]
    hits = {tuple(s["target"]) for s in observation["my_shots"] if s["result"] in ("HIT", "SUNK")}
    near = [
        (r + dr, c + dc)
        for r, c in hits
        for dr, dc in DIRECTIONS.values()
        if _inside((r + dr, c + dc), size) and (r + dr, c + dc) not in hits
    ]
    free = near or [(r, c) for r in range(size) for c in range(size) if (r, c) not in hits]
    r, c = random.choice(free) if free else (0, 0)
    return {"shot": [r, c]}


# Required algorithms. 


def consistent_placements(length: int, board_size: int, blocked: list[list[int]], required: list[list[int]]):
    """Every position of one ship consistent with the evidence.

    blocked   cells the ship cannot occupy, as [[row, col], ...]
    required  cells the ship must cover, as [[row, col], ...]

    Returns [[row, col, "H" or "V"], ...], where [row, col] is the ship's top-left cell,
    in any order and without duplicates. Length is always at least 2.
    """
    blocked_set = {tuple(cell) for cell in blocked}
    required_set = {tuple(cell) for cell in required}
    placements = []
    for orientation in "HV":
        rows = board_size - (length - 1 if orientation == "V" else 0)
        cols = board_size - (length - 1 if orientation == "H" else 0)
        for row in range(rows):
            for col in range(cols):
                cells = set(_ship_cells(row, col, orientation, length))
                if not cells & blocked_set and required_set <= cells:
                    placements.append([row, col, orientation])
    return placements


def _sonar_likelihood(reading, true_count, region_cells, correct_probability):
    """P(reading | true count). The reading is the truth with correct_probability, else one
    lower or one higher. Readings outside 0..region_cells are impossible, so renormalise."""
    if not 0 <= true_count <= region_cells:
        return 0.0
    error = (1 - correct_probability) / 2
    options = {true_count: correct_probability, true_count - 1: error, true_count + 1: error}
    options = {r: p for r, p in options.items() if 0 <= r <= region_cells}
    return options.get(reading, 0.0) / sum(options.values())


def sonar_posterior(prior: dict, reading: int, region_cells: int, correct_probability: float):
    """Bayes' rule: P(true count | sonar reading).

    prior  {"0": 0.5, "1": 0.3, ...}, a probability for each true count (keys are strings,
           and never above region_cells)
    reading  the number the sonar returned
    region_cells  how many cells the sonar covered, which is the largest possible reading

    Sensor model: the reading equals the true count with correct_probability, and is
    one lower or one higher with the remaining probability split equally. A reading
    outside 0..region_cells is impossible, so the other outcomes are renormalised.

    Returns {"0": p0, "1": p1, ...}, the posterior over the same keys.
    """
    joint = {key: p * _sonar_likelihood(reading, int(key), region_cells, correct_probability) for key, p in prior.items()}
    total = sum(joint.values())
    return {key: value / total for key, value in joint.items()}


def predict_ship_belief(belief: list[list[float]], length: int, orientation: str, move_probability: float) -> list[list[float]]:
    """One prediction step of a belief filter for one unseen ship.

    belief[row][col] = P(the ship's top-left cell is [row, col]), on a square board.

    Transition model used here: with move_probability the ship moves one cell, chosen
    uniformly among the directions where it stays on the board. Otherwise it stays
    put. A ship with no legal direction always stays put.

    Returns the new belief, with the same shape.
    """
    size = len(belief)

    def fits(row, col):
        return all(_inside(cell, size) for cell in _ship_cells(row, col, orientation, length))

    new = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            mass = belief[row][col]
            if mass == 0:
                continue
            options = [(row + dr, col + dc) for dr, dc in DIRECTIONS.values() if fits(row + dr, col + dc)]
            if not options:
                new[row][col] += mass
                continue
            new[row][col] += mass * (1 - move_probability)  
            for r, c in options:
                new[r][c] += mass * move_probability / len(options)
    return new
