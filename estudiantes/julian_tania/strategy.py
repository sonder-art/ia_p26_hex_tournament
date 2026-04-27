from __future__ import annotations

import math
import random
import time
from collections import deque

from strategy import Strategy, GameConfig


INF = 10 ** 9
_GEOM = {}


def _geom(size: int):
    g = _GEOM.get(size)
    if g is not None:
        return g

    n = size * size
    mid = size // 2
    dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    nb = [[] for _ in range(n)]
    p1_start = [0] * n
    p1_end = [0] * n
    p2_start = [0] * n
    p2_end = [0] * n
    p1_mask = [0] * n
    p2_mask = [0] * n
    center = [0] * n
    axis_p1 = [0] * n
    axis_p2 = [0] * n
    rows = [0] * n
    cols = [0] * n

    for r in range(size):
        for c in range(size):
            i = r * size + c
            rows[i] = r
            cols[i] = c
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nb[i].append(nr * size + nc)
            p1_start[i] = 1 if r == 0 else 0
            p1_end[i] = 1 if r == size - 1 else 0
            p2_start[i] = 1 if c == 0 else 0
            p2_end[i] = 1 if c == size - 1 else 0
            p1_mask[i] = (1 if r == 0 else 0) | (2 if r == size - 1 else 0)
            p2_mask[i] = (1 if c == 0 else 0) | (2 if c == size - 1 else 0)
            dr = r - mid
            dc = c - mid
            hex_dist = (abs(dr) + abs(dc) + abs(dr + dc)) // 2
            center[i] = size - hex_dist
            axis_p1[i] = size - abs(c - mid)
            axis_p2[i] = size - abs(r - mid)

    g = {
        "n": n,
        "mid": mid,
        "nb": tuple(tuple(x) for x in nb),
        "p1_start": tuple(p1_start),
        "p1_end": tuple(p1_end),
        "p2_start": tuple(p2_start),
        "p2_end": tuple(p2_end),
        "p1_mask": tuple(p1_mask),
        "p2_mask": tuple(p2_mask),
        "center": tuple(center),
        "axis_p1": tuple(axis_p1),
        "axis_p2": tuple(axis_p2),
        "rows": tuple(rows),
        "cols": tuple(cols),
    }
    _GEOM[size] = g
    return g


class _WinChecker:
    __slots__ = ("size", "seen", "q", "gen")

    def __init__(self, size: int):
        self.size = size
        n = size * size
        self.seen = [0] * n
        self.q = [0] * n
        self.gen = 0

    def wins(self, arr, player: int) -> bool:
        g = _geom(self.size)
        nb = g["nb"]
        starts = g["p1_start"] if player == 1 else g["p2_start"]
        ends = g["p1_end"] if player == 1 else g["p2_end"]
        self.gen += 1
        stamp = self.gen
        seen = self.seen
        q = self.q
        qs = 0
        qe = 0
        for i in range(self.size * self.size):
            if starts[i] and arr[i] == player:
                seen[i] = stamp
                q[qe] = i
                qe += 1
        while qs < qe:
            x = q[qs]
            qs += 1
            if ends[x]:
                return True
            for y in nb[x]:
                if arr[y] == player and seen[y] != stamp:
                    seen[y] = stamp
                    q[qe] = y
                    qe += 1
        return False


class _DSU:
    __slots__ = ("parent", "size", "touch")

    def __init__(self, arr, player: int, size: int, side_mask):
        n = size * size
        self.parent = list(range(n))
        self.size = [0] * n
        self.touch = [0] * n
        nb = _geom(size)["nb"]
        for i in range(n):
            if arr[i] == player:
                self.size[i] = 1
                self.touch[i] = side_mask[i]
            else:
                self.parent[i] = -1
        for i in range(n):
            if arr[i] != player:
                continue
            for j in nb[i]:
                if j > i and arr[j] == player:
                    self.union(i, j)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        sa = self.size
        if sa[pa] < sa[pb]:
            pa, pb = pb, pa
        self.parent[pb] = pa
        sa[pa] += sa[pb]
        self.touch[pa] |= self.touch[pb]


def _flatten(board, size: int):
    return [board[r][c] for r in range(size) for c in range(size)]


def _empties(arr):
    return [i for i, v in enumerate(arr) if v == 0]


def _logistic(x: float) -> float:
    if x >= 10:
        return 0.999955
    if x <= -10:
        return 0.000045
    return 1.0 / (1.0 + math.exp(-x))


def _dist_maps(arr, player: int, size: int):
    opp = 3 - player
    g = _geom(size)
    n = g["n"]
    nb = g["nb"]
    starts = g["p1_start"] if player == 1 else g["p2_start"]
    ends = g["p1_end"] if player == 1 else g["p2_end"]

    def run(seeds):
        dist = [INF] * n
        dq = deque()
        for i in range(n):
            if not seeds[i]:
                continue
            v = arr[i]
            if v == opp:
                continue
            w = 0 if v == player else 1
            dist[i] = w
            if w == 0:
                dq.appendleft(i)
            else:
                dq.append(i)
        while dq:
            x = dq.popleft()
            dx = dist[x]
            for y in nb[x]:
                v = arr[y]
                if v == opp:
                    continue
                w = 0 if v == player else 1
                nd = dx + w
                if nd < dist[y]:
                    dist[y] = nd
                    if w == 0:
                        dq.appendleft(y)
                    else:
                        dq.append(y)
        return dist

    ds = run(starts)
    de = run(ends)
    best = INF
    for i in range(n):
        if ends[i] and ds[i] < best:
            best = ds[i]
    return ds, de, best


def _bridge_bonus(arr, idx: int, player: int, size: int) -> int:
    nb = _geom(size)["nb"]
    neigh = [x for x in nb[idx] if arr[x] == player]
    if len(neigh) < 2:
        return 0
    opp = 3 - player
    bonus = 0
    for i in range(len(neigh)):
        a = neigh[i]
        na = nb[a]
        for j in range(i + 1, len(neigh)):
            b = neigh[j]
            if b in na:
                continue
            other = -1
            for x in na:
                if x != idx and x in nb[b]:
                    other = x
                    break
            if other < 0:
                continue
            v = arr[other]
            if v == 0:
                bonus += 6
            elif v == player:
                bonus += 10
            else:
                bonus += 8 if v == opp else 0
    return bonus


def _local_score(arr, idx: int, player: int, size: int) -> float:
    g = _geom(size)
    nb = g["nb"]
    center = g["center"][idx]
    axis = g["axis_p1"][idx] if player == 1 else g["axis_p2"][idx]
    opp = 3 - player
    same = enemy = empty = 0
    own_touch = (g["p1_mask"] if player == 1 else g["p2_mask"])[idx]
    opp_touch = (g["p1_mask"] if opp == 1 else g["p2_mask"])[idx]
    for y in nb[idx]:
        v = arr[y]
        if v == player:
            same += 1
        elif v == opp:
            enemy += 1
        else:
            empty += 1
    s = 7.0 * same + 5.0 * enemy + 2.0 * center + 2.3 * axis + 0.4 * empty
    if own_touch:
        s += 4.0
    if opp_touch:
        s += 3.0
    s += _bridge_bonus(arr, idx, player, size)
    s += 0.65 * _bridge_bonus(arr, idx, opp, size)
    return s


def _candidate_scores(arr, empties, player: int, size: int, variant: str = "classic", hidden_prob=None):
    opp = 3 - player
    g = _geom(size)
    center = g["center"]
    axis = g["axis_p1"] if player == 1 else g["axis_p2"]
    own_mask = g["p1_mask"] if player == 1 else g["p2_mask"]
    opp_mask = g["p1_mask"] if opp == 1 else g["p2_mask"]
    nb = g["nb"]
    ds, de, my_best = _dist_maps(arr, player, size)
    ods, ode, opp_best = _dist_maps(arr, opp, size)
    dsu_self = _DSU(arr, player, size, own_mask)
    dsu_opp = _DSU(arr, opp, size, opp_mask)

    out = []
    for idx in empties:
        my_through = ds[idx] + de[idx] - 1 if ds[idx] < INF and de[idx] < INF else INF
        opp_through = ods[idx] + ode[idx] - 1 if ods[idx] < INF and ode[idx] < INF else INF
        s = 0.0
        if my_through < INF:
            s += 22.0 * (2 * size - my_through)
            s += 11.0 * max(0, my_best - (my_through - 1))
        if opp_through < INF:
            s += 19.0 * (2 * size - opp_through)
            s += 9.0 * max(0, opp_best - opp_through)

        own_roots = []
        opp_roots = []
        own_touch = own_mask[idx]
        opp_touch = opp_mask[idx]
        adj_self = 0
        adj_opp = 0
        for y in nb[idx]:
            v = arr[y]
            if v == player:
                adj_self += 1
                r = dsu_self.find(y)
                if r not in own_roots:
                    own_roots.append(r)
                    own_touch |= dsu_self.touch[r]
                    s += 2.0 * dsu_self.size[r]
            elif v == opp:
                adj_opp += 1
                r = dsu_opp.find(y)
                if r not in opp_roots:
                    opp_roots.append(r)
                    opp_touch |= dsu_opp.touch[r]
                    s += 1.6 * dsu_opp.size[r]

        s += 8.0 * adj_self + 6.0 * adj_opp
        s += 16.0 * max(0, len(own_roots) - 1)
        s += 13.0 * max(0, len(opp_roots) - 1)
        if own_touch == 3:
            s += 140.0
        elif own_touch:
            s += 15.0
        if opp_touch == 3:
            s += 95.0
        elif opp_touch:
            s += 12.0

        s += 1.7 * center[idx] + 2.5 * axis[idx]
        s += _bridge_bonus(arr, idx, player, size)
        s += 0.8 * _bridge_bonus(arr, idx, opp, size)

        if variant == "dark" and hidden_prob is not None:
            hp = hidden_prob[idx]
            if opp_through <= my_through:
                s += 8.0 * hp
            else:
                s -= 5.5 * hp

        out.append((s, idx))
    out.sort(reverse=True)
    return out, my_best, opp_best


def _eval_position(arr, player: int, size: int, checker: _WinChecker) -> float:
    opp = 3 - player
    if checker.wins(arr, player):
        return 1.0
    if checker.wins(arr, opp):
        return 0.0
    _, _, my_best = _dist_maps(arr, player, size)
    _, _, opp_best = _dist_maps(arr, opp, size)
    if my_best >= INF and opp_best >= INF:
        return 0.5
    if my_best >= INF:
        return 0.0
    if opp_best >= INF:
        return 1.0
    return _logistic(0.95 * (opp_best - my_best))


def _find_immediate(arr, candidates, player: int, checker: _WinChecker):
    for idx in candidates:
        if arr[idx] != 0:
            continue
        arr[idx] = player
        ok = checker.wins(arr, player)
        arr[idx] = 0
        if ok:
            return idx
    return -1


def _top_cap(num_empty: int) -> int:
    if num_empty > 90:
        return 12
    if num_empty > 70:
        return 14
    if num_empty > 50:
        return 16
    if num_empty > 32:
        return 18
    if num_empty > 20:
        return 22
    return 28


def _weighted_sample_no_replace(items, weights, k: int):
    if k <= 0 or not items:
        return []
    pool_i = items[:]
    pool_w = weights[:]
    out = []
    k = min(k, len(pool_i))
    for _ in range(k):
        total = sum(pool_w)
        if total <= 0:
            j = random.randrange(len(pool_i))
        else:
            r = random.random() * total
            acc = 0.0
            j = len(pool_i) - 1
            for t, w in enumerate(pool_w):
                acc += w
                if acc >= r:
                    j = t
                    break
        out.append(pool_i.pop(j))
        pool_w.pop(j)
    return out


def _reply_move(arr, empties, player: int, size: int, checker: _WinChecker) -> int:
    # Cambio 3: sin _candidate_scores (2xBFS+2xDSU) en cada iteracion del loop MCTS.
    # Chequeo tactico barato sobre muestra pequeña.
    sample = empties if len(empties) <= 12 else random.sample(empties, 12)
    win_idx = _find_immediate(arr, sample, player, checker)
    if win_idx >= 0:
        return win_idx
    # Respuesta por local_score — O(1) por celda, sin Dijkstra ni DSU.
    best_idx = empties[0]
    best_s = -1e18
    cap = min(10, len(empties))
    candidates = empties if len(empties) <= cap else random.sample(empties, cap)
    for idx in candidates:
        s = _local_score(arr, idx, player, size) + random.random() * 0.8
        if s > best_s:
            best_s = s
            best_idx = idx
    return best_idx


def _rollout_pick(arr, cells, m: int, mover: int, size: int) -> int:
    # Sample a handful of candidates and pick by bridge-aware local score.
    sample = 6 if m > 14 else m
    best_j = 0
    best_s = -1e18
    used = set()
    for _ in range(sample):
        if m <= 0:
            break
        j = random.randrange(m)
        while j in used and len(used) < m:
            j = random.randrange(m)
        used.add(j)
        idx = cells[j]
        s = _local_score(arr, idx, mover, size) + random.random() * 1.2
        if s > best_s:
            best_s = s
            best_j = j
    return best_j


def _biased_rollout(arr, empties, to_move: int, root_player: int, size: int, checker: _WinChecker):
    m = len(empties)
    if m <= 0:
        return 1.0 if checker.wins(arr, root_player) else 0.0

    cells = empties[:]
    random.shuffle(cells)
    if m > 75:
        depth = 12
    elif m > 48:
        depth = 16
    elif m > 26:
        depth = 20
    else:
        depth = m

    moves_by_root = []
    player = to_move
    steps = 0
    while m > 0 and steps < depth:
        j = _rollout_pick(arr, cells, m, player, size)
        idx = cells[j]
        m -= 1
        cells[j], cells[m] = cells[m], cells[j]
        arr[idx] = player
        if player == root_player:
            moves_by_root.append(idx)
        steps += 1
        if steps >= 6 and (steps & 1) == 0:
            if checker.wins(arr, root_player):
                return 1.0, moves_by_root
            if checker.wins(arr, 3 - root_player):
                return 0.0, moves_by_root
        player = 3 - player

    return _eval_position(arr, root_player, size, checker), moves_by_root


def _exact_negamax(arr, empties, mover: int, size: int, checker: _WinChecker, memo, alpha: int, beta: int) -> int:
    key = (tuple(arr), mover)
    got = memo.get(key)
    if got is not None:
        return got
    prev = 3 - mover
    if checker.wins(arr, prev):
        memo[key] = -1
        return -1
    if not empties:
        memo[key] = -1
        return -1

    ordered = [idx for _, idx in _candidate_scores(arr, empties, mover, size)[0]]
    for idx in ordered:
        arr[idx] = mover
        nxt = [x for x in empties if x != idx]
        val = -_exact_negamax(arr, nxt, 3 - mover, size, checker, memo, -beta, -alpha)
        arr[idx] = 0
        if val > alpha:
            alpha = val
            if alpha >= beta:
                memo[key] = alpha
                return alpha
    memo[key] = alpha
    return alpha


def _solve_endgame(arr, empties, mover: int, size: int, checker: _WinChecker):
    memo = {}
    ordered = [idx for _, idx in _candidate_scores(arr, empties, mover, size)[0]]
    for idx in ordered:
        arr[idx] = mover
        nxt = [x for x in empties if x != idx]
        val = -_exact_negamax(arr, nxt, 3 - mover, size, checker, memo, -1, 1)
        arr[idx] = 0
        if val == 1:
            return idx
    return ordered[0] if ordered else -1


def _hidden_probability(board, size: int, player: int, opponent: int, nominal_hidden: int):
    arr = _flatten(board, size)
    empt = _empties(arr)
    if not empt or nominal_hidden <= 0:
        return [0.0] * (size * size)

    g = _geom(size)
    nb = g["nb"]
    opp_scores, _, _ = _candidate_scores(arr, empt, opponent, size)
    critical = {idx: rank for rank, (_, idx) in enumerate(opp_scores[: min(18, len(opp_scores))])}
    hp = [0.0] * (size * size)
    for idx in empt:
        adj_opp = 0
        adj_me = 0
        for y in nb[idx]:
            if arr[y] == opponent:
                adj_opp += 1
            elif arr[y] == player:
                adj_me += 1
        axis = g["axis_p1"][idx] if opponent == 1 else g["axis_p2"][idx]
        center = g["center"][idx]
        score = 1.0 + 2.8 * adj_opp + 0.6 * center + 0.7 * axis - 0.55 * adj_me
        if idx in critical:
            score += max(0, 14 - critical[idx])
        hp[idx] = max(0.05, score)

    total = sum(hp[i] for i in empt)
    scale = float(nominal_hidden) / total if total > 0 else 0.0
    for i in empt:
        hp[i] *= scale
    return hp


def _determinize(board, size: int, player: int, opponent: int, hidden_prob, hidden_count: int):
    arr = _flatten(board, size)
    empt = [i for i, v in enumerate(arr) if v == 0]
    if hidden_count <= 0 or not empt:
        return arr
    weights = [hidden_prob[i] for i in empt]
    picks = _weighted_sample_no_replace(empt, weights, min(hidden_count, len(empt)))
    out = arr[:]
    for idx in picks:
        out[idx] = opponent
    return out


class _RootStats:
    __slots__ = ("idx", "prior", "visits", "wins", "amaf_visits", "amaf_wins")

    def __init__(self, idx: int, prior: float):
        self.idx = idx
        self.prior = prior
        self.visits = 0
        self.wins = 0.0
        self.amaf_visits = 0
        self.amaf_wins = 0.0

    def score(self, total: int, c: float = 1.25):
        if self.visits == 0:
            return 10.0 + 2.5 * self.prior
        q = self.wins / self.visits
        rave = self.amaf_wins / self.amaf_visits if self.amaf_visits else 0.5
        beta = self.amaf_visits / (self.amaf_visits + self.visits + 3.0)
        mean = (1.0 - beta) * q + beta * rave
        return mean + 0.08 * self.prior + c * math.sqrt(max(1e-9, math.log(total + 1.0)) / self.visits)


def _root_mc_search(board, size: int, player: int, opponent: int, variant: str, budget: float,
                    hidden_prob=None, hidden_count: int = 0):
    checker = _WinChecker(size)
    observed = _flatten(board, size)
    empt = _empties(observed)
    if not empt:
        return None
    if len(empt) == 1:
        idx = empt[0]
        return (idx // size, idx % size)

    scored, _, _ = _candidate_scores(observed, empt, player, size, variant, hidden_prob)
    cap = _top_cap(len(empt))
    root_list = [idx for _, idx in scored[: min(cap, len(scored))]]
    all_top = [idx for _, idx in scored[: min(24, len(scored))]]

    # Tactical checks.
    win_idx = _find_immediate(observed, all_top, player, checker)
    if win_idx >= 0:
        return (win_idx // size, win_idx % size)

    opp_scored, _, _ = _candidate_scores(observed, empt, opponent, size, variant, hidden_prob)
    opp_top = [idx for _, idx in opp_scored[: min(20, len(opp_scored))]]
    opp_win = _find_immediate(observed, opp_top, opponent, checker)
    if opp_win >= 0 and opp_win in root_list:
        return (opp_win // size, opp_win % size)
    if opp_win >= 0:
        root_list = [opp_win] + [idx for idx in root_list if idx != opp_win]

    if len(empt) <= 11:  # Cambio 2: ampliado de 8 a 11
        end_idx = _solve_endgame(observed, empt, player, size, checker)
        if end_idx >= 0:
            return (end_idx // size, end_idx % size)

    # Normalize heuristic priors on candidate set.
    root_scores = {idx: s for s, idx in scored[: min(cap, len(scored))]}
    mx = max(root_scores.values()) if root_scores else 1.0
    mn = min(root_scores.values()) if root_scores else 0.0
    den = max(1e-9, mx - mn)
    stats = {idx: _RootStats(idx, (root_scores.get(idx, mn) - mn) / den) for idx in root_list}
    total = 0

    # Prepare determinized worlds for dark.
    worlds = None
    if variant == "dark":
        worlds = []
        counts = [hidden_count]
        if hidden_count > 0:
            counts = sorted(set([max(0, hidden_count - 1), hidden_count, hidden_count + 1]))
        # Cambio 1: más mundos iniciales para mejor cobertura vs Tier_5
        world_budget = 8 if budget > 4.0 else 6 if budget > 2.0 else 4
        for c in counts:
            for _ in range(world_budget):
                worlds.append(_determinize(board, size, player, opponent, hidden_prob, c))
        if not worlds:
            worlds = [observed]

    deadline = time.monotonic() + budget
    rng = random.random
    while time.monotonic() < deadline:
        total += 1
        # Root selection with UCB + AMAF.
        best = None
        best_v = -1e18
        for idx in root_list:
            v = stats[idx].score(total)
            if v > best_v:
                best_v = v
                best = idx
        root_idx = best
        st = stats[root_idx]

        # Cambio 1b: rotar un mundo cada 40 iteraciones para evitar mundos estáticos
        if worlds is not None and total % 40 == 0:
            c = random.choice(counts)
            worlds[total % len(worlds)] = _determinize(board, size, player, opponent, hidden_prob, c)
        base = random.choice(worlds)[:] if worlds is not None else observed[:]
        if base[root_idx] != 0:
            # Determinization may place hidden stone here. Collision risk matters in dark.
            if variant == "dark":
                st.visits += 1
                total += 0
                continue
            else:
                continue

        base[root_idx] = player
        if checker.wins(base, player):
            st.visits += 1
            st.wins += 1.0
            st.amaf_visits += 1
            st.amaf_wins += 1.0
            continue

        sim_empt = [x for x in empt if x != root_idx and base[x] == 0]

        # Adversarial but stochastic one-ply reply.
        if sim_empt:
            opp_idx = _reply_move(base, sim_empt, opponent, size, checker)
            if base[opp_idx] == 0:
                base[opp_idx] = opponent
                if checker.wins(base, opponent):
                    result = 0.0
                    root_moves = []
                    st.visits += 1
                    st.wins += result
                    continue
                sim_empt = [x for x in sim_empt if x != opp_idx]

        result, later_root_moves = _biased_rollout(base, sim_empt, player, player, size, checker)
        st.visits += 1
        st.wins += result

        # Root-level AMAF/RAVE: if root player later used another root candidate, share signal.
        used = set(later_root_moves)
        for idx in used:
            ref = stats.get(idx)
            if ref is not None:
                ref.amaf_visits += 1
                ref.amaf_wins += result

    # Pick robustly: mostly by visits, break ties by value.
    best = max(root_list, key=lambda idx: (stats[idx].visits, stats[idx].wins / stats[idx].visits if stats[idx].visits else -1.0, stats[idx].prior))
    return (best // size, best % size)


class HexForgeStrategy(Strategy):
    @property
    def name(self) -> str:
        return "HexForge_julianTania2"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opp = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit
        self._play_budget = config.time_limit * 0.90
        self._my_attempts = 0
        self._my_success = 0
        self._my_fail = 0

    def on_move_result(self, move, success):
        self._my_attempts += 1
        if success:
            self._my_success += 1
        else:
            self._my_fail += 1

    def _opening_move(self, board):
        size = self._size
        g = _geom(size)
        empt = []
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    empt.append((r, c))
        if not empt:
            return None
        filled = size * size - len(empt)
        if filled > 1:
            return None
        mid = g["mid"]
        prefs = [(mid, mid)]
        ring = [(mid - 1, mid), (mid - 1, mid + 1), (mid, mid - 1), (mid, mid + 1), (mid + 1, mid - 1), (mid + 1, mid)]
        for rc in ring:
            prefs.append(rc)
        if self._player == 1:
            prefs.extend([(mid - 2, mid), (mid + 2, mid), (mid, mid - 1), (mid, mid + 1)])
        else:
            prefs.extend([(mid, mid - 2), (mid, mid + 2), (mid - 1, mid), (mid + 1, mid)])
        for r, c in prefs:
            if 0 <= r < size and 0 <= c < size and board[r][c] == 0:
                return (r, c)
        return None

    def _hidden_count_estimate(self, board):
        if self._variant != "dark":
            return 0
        visible_opp = 0
        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] == self._opp:
                    visible_opp += 1
        if self._player == 1:
            opp_turns_nominal = self._my_attempts
        else:
            opp_turns_nominal = self._my_attempts + 1
        target_total = max(0, opp_turns_nominal)
        raw_hidden = max(0, target_total - visible_opp)
        # Cambio 4: amortiguación — en dark no todas las jugadas del oponente son exitosas
        hidden = round(raw_hidden * 0.80)
        empties = sum(1 for r in range(self._size) for c in range(self._size) if board[r][c] == 0)
        return min(hidden, empties)

    def play(self, board, last_move):
        size = self._size
        t0 = time.monotonic()
        opening = self._opening_move(board)
        if opening is not None:
            return opening

        hidden_prob = None
        hidden_count = 0
        if self._variant == "dark":
            hidden_count = self._hidden_count_estimate(board)
            hidden_prob = _hidden_probability(board, size, self._player, self._opp, hidden_count)

        budget = self._play_budget - (time.monotonic() - t0)
        if budget < 0.05:
            # Cheap fallback.
            arr = _flatten(board, size)
            empt = _empties(arr)
            if not empt:
                return (0, 0)
            scored, _, _ = _candidate_scores(arr, empt, self._player, size, self._variant, hidden_prob)
            idx = scored[0][1]
            return (idx // size, idx % size)

        move = _root_mc_search(board, size, self._player, self._opp, self._variant, budget, hidden_prob, hidden_count)
        if move is None or board[move[0]][move[1]] != 0:
            arr = _flatten(board, size)
            empt = _empties(arr)
            if not empt:
                return (0, 0)
            scored, _, _ = _candidate_scores(arr, empt, self._player, size, self._variant, hidden_prob)
            idx = scored[0][1]
            return (idx // size, idx % size)
        return move
