"""
Estrategia MCTS + RAVE para Hex 11x11.
Autores: <gabriel_regina>

Version 7:
  Save-bridge pattern: si el oponente juega en una carrier de nuestro bridge,
  respondemos en la otra carrier preservando la conexion (patron fundamental
  de Hex — reduce varianza en partidas cerradas).
  Mantiene: V6 (root parallelization + soft eval) y todos los fixes V5.
"""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict

from strategy import Strategy, GameConfig
from hex_game import (
    get_neighbors,
    check_winner,
    shortest_path_distance,
    empty_cells,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
UCT_C         = 1.2
RAVE_K        = 400
RAVE_BLEND    = 0.8
TIME_BUDGET   = 0.92
CUTOFF_FILL   = 0.65
NEIGHBOR_P    = 0.75
DIRECTION_P   = 0.20
DIRECTION_K   = 5
EXPAND_RADIUS = 2
TRANS_CAP     = 50
NUM_WORKERS   = 3   # workers adicionales; total = NUM_WORKERS + 1 main

# Save-bridge: 6 patrones de bridge centrados en `last` (celda del oponente).
# (A_offset, B_offset, save_offset) relativo a `last`.
# Si A y B son nuestras piedras y `save` esta vacia → save preserva el bridge.
BRIDGE_PATTERNS = (
    ((-1, 0),  (0, 1),   (-1, 1)),
    ((-1, 0),  (1, -1),  (0, -1)),
    ((-1, 1),  (1, 0),   (0, 1)),
    ((-1, 1),  (0, -1),  (-1, 0)),
    ((0, -1),  (1, 0),   (1, -1)),
    ((0, 1),   (1, -1),  (1, 0)),
)

# Break-bridge: desde `last` como un extremo del bridge del oponente.
# (B_offset, c1_offset, c2_offset) relativo a `last` (extremo A).
# Si B es del oponente y c1, c2 estan vacias → jugar c1 rompe el bridge.
BRIDGE_ENDPOINTS = (
    ((-2, 1),  (-1, 0),  (-1, 1)),
    ((-1, 2),  (-1, 1),  (0, 1)),
    ((1, 1),   (0, 1),   (1, 0)),
    ((2, -1),  (1, 0),   (1, -1)),
    ((1, -2),  (1, -1),  (0, -1)),
    ((-1, -1), (0, -1),  (-1, 0)),
)


# ---------------------------------------------------------------------------
# Pool de celdas vacias — seleccion y eliminacion en O(1)
# ---------------------------------------------------------------------------
class _EmptyPool:
    __slots__ = ("cells", "pos")

    def __init__(self, board, size):
        self.cells = []
        self.pos = {}
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    self.pos[(r, c)] = len(self.cells)
                    self.cells.append((r, c))

    def random(self):
        return self.cells[random.randrange(len(self.cells))]

    def remove(self, cell):
        idx = self.pos.pop(cell)
        last = self.cells.pop()
        if idx < len(self.cells):
            self.cells[idx] = last
            self.pos[last] = idx

    def __len__(self):
        return len(self.cells)


# ---------------------------------------------------------------------------
# Nodo del arbol MCTS
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = (
        "move", "parent", "children", "visits", "wins",
        "rave_visits", "rave_wins", "untried_moves", "player_to_move",
        "_board_hash",
    )

    def __init__(self, move, parent, untried_moves, player_to_move):
        self.move           = move
        self.parent         = parent
        self.children       = []
        self.visits         = 0
        self.wins           = 0.0
        self.rave_visits    = defaultdict(int)
        self.rave_wins      = defaultdict(float)
        self.untried_moves  = untried_moves
        self.player_to_move = player_to_move
        self._board_hash    = None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def uct_rave_score(self, parent_visits):
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = UCT_C * math.sqrt(math.log(parent_visits) / self.visits)
        uct_val = exploit + explore

        rv = self.parent.rave_visits.get(self.move, 0) if self.parent else 0
        if rv > 0:
            rave_val = self.parent.rave_wins.get(self.move, 0) / rv
            beta = math.sqrt(RAVE_K / (3 * self.visits + RAVE_K))
            beta = min(beta, RAVE_BLEND)
            return (1 - beta) * uct_val + beta * rave_val
        return uct_val


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------

def _board_to_lists(board):
    return [list(row) for row in board]


def _soft_eval(board, size, root_player, next_to_move):
    """Eval continuo [0,1]: sigmoid de diferencia de distancias + tempo."""
    my_dist  = shortest_path_distance(board, size, root_player)
    opp_dist = shortest_path_distance(board, size, 3 - root_player)
    INF = size * size + 1
    if my_dist >= INF:
        return 0.0
    if opp_dist >= INF:
        return 1.0
    if next_to_move != root_player:
        opp_dist = max(0, opp_dist - 1)
    diff = opp_dist - my_dist
    return 1.0 / (1.0 + math.exp(-diff * 0.8))


def _neighborhood_empties(board, size, empties, radius=EXPAND_RADIUS):
    pieces_exist = False
    in_nbhd = [[False] * size for _ in range(size)]
    for r in range(size):
        row = board[r]
        for c in range(size):
            if row[c] != 0:
                pieces_exist = True
                rmin = max(0, r - radius)
                rmax = min(size - 1, r + radius)
                cmin = max(0, c - radius)
                cmax = min(size - 1, c + radius)
                for nr in range(rmin, rmax + 1):
                    nrow = in_nbhd[nr]
                    for nc in range(cmin, cmax + 1):
                        nrow[nc] = True
    if not pieces_exist:
        return None
    return [m for m in empties if in_nbhd[m[0]][m[1]]]


def _candidates(board, size, empties):
    nbhd = _neighborhood_empties(board, size, empties)
    if nbhd and len(nbhd) >= 5:
        return nbhd
    if nbhd is None:
        center = size // 2
        radius = size // 3
        return [(r, c) for r, c in empties
                if abs(r - center) <= radius and abs(c - center) <= radius]
    return list(empties)


# ---------------------------------------------------------------------------
# FPU: Dijkstra bidireccional para ordenar candidatos en la raiz
# ---------------------------------------------------------------------------

def _full_dijkstra(board, size, player, from_start):
    INF = float("inf")
    opp = 3 - player
    dist = {}
    heap = []

    if player == 1:
        edge_rc = [(0, c) for c in range(size)] if from_start \
                  else [(size - 1, c) for c in range(size)]
    else:
        edge_rc = [(r, 0) for r in range(size)] if from_start \
                  else [(r, size - 1) for r in range(size)]

    for r, c in edge_rc:
        if board[r][c] == opp:
            continue
        d = 0 if board[r][c] == player else 1
        if d < dist.get((r, c), INF):
            dist[(r, c)] = d
            heapq.heappush(heap, (d, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist.get((r, c), INF):
            continue
        for nr, nc in get_neighbors(r, c, size):
            if board[nr][nc] == opp:
                continue
            add = 0 if board[nr][nc] == player else 1
            nd = d + add
            if nd < dist.get((nr, nc), INF):
                dist[(nr, nc)] = nd
                heapq.heappush(heap, (nd, nr, nc))

    return dist


def _fpu_order(board, size, candidates, player):
    """Celdas en camino Dijkstra minimo van al final (pop() las extrae primero)."""
    if not candidates:
        return candidates

    INF = float("inf")
    fwd = _full_dijkstra(board, size, player, from_start=True)
    bwd = _full_dijkstra(board, size, player, from_start=False)

    if player == 1:
        total = min(fwd.get((size - 1, c), INF) for c in range(size))
    else:
        total = min(fwd.get((r, size - 1), INF) for r in range(size))

    if total == INF:
        random.shuffle(candidates)
        return candidates

    on_path = set()
    for r, c in candidates:
        f = fwd.get((r, c), INF)
        b_d = bwd.get((r, c), INF)
        if f != INF and b_d != INF and f + b_d == total + 1:
            on_path.add((r, c))

    path_cells  = [m for m in candidates if m in on_path]
    other_cells = [m for m in candidates if m not in on_path]
    random.shuffle(other_cells)
    random.shuffle(path_cells)
    return other_cells + path_cells


# ---------------------------------------------------------------------------
# Save-bridge: detectar si `last` rompio uno de nuestros bridges
# ---------------------------------------------------------------------------

def _check_save_bridge(b, size, last, current):
    """Si `last` es carrier de un bridge de `current`, retorna la otra carrier."""
    lr, lc = last
    for (dar, dac), (dbr, dbc), (dsr, dsc) in BRIDGE_PATTERNS:
        ar, ac = lr + dar, lc + dac
        if not (0 <= ar < size and 0 <= ac < size):
            continue
        if b[ar][ac] != current:
            continue
        br, bc = lr + dbr, lc + dbc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != current:
            continue
        sr, sc = lr + dsr, lc + dsc
        if not (0 <= sr < size and 0 <= sc < size):
            continue
        if b[sr][sc] != 0:
            continue
        return (sr, sc)
    return None


def _check_break_bridge(b, size, last, current):
    """Si `last` (piedra del oponente) forma un bridge con otra piedra suya, retorna
    una carrier para romperlo. `last` se trata como extremo A del bridge."""
    opp = 3 - current
    lr, lc = last
    for (dbr, dbc), (dc1r, dc1c), (dc2r, dc2c) in BRIDGE_ENDPOINTS:
        br, bc = lr + dbr, lc + dbc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != opp:
            continue
        c1r, c1c = lr + dc1r, lc + dc1c
        if not (0 <= c1r < size and 0 <= c1c < size):
            continue
        c2r, c2c = lr + dc2r, lc + dc2c
        if not (0 <= c2r < size and 0 <= c2c < size):
            continue
        if b[c1r][c1c] == 0 and b[c2r][c2c] == 0:
            return (c1r, c1c)
    return None


# ---------------------------------------------------------------------------
# Rollout rapido con save-bridge + bias direccional
# ---------------------------------------------------------------------------

def _fast_rollout(b, size, player_to_move, root_player, pool, filled):
    current = player_to_move
    cutoff = int(CUTOFF_FILL * size * size)
    moves_played = []
    last = None

    p1_top   = any(b[0][c] == 1 for c in range(size))
    p1_bot   = any(b[size - 1][c] == 1 for c in range(size))
    p2_left  = any(b[r][0] == 2 for r in range(size))
    p2_right = any(b[r][size - 1] == 2 for r in range(size))

    while filled < cutoff and len(pool) > 0:
        chosen = None

        # Prioridad 0: save-bridge (salvar nuestra conexion)
        if last is not None:
            save = _check_save_bridge(b, size, last, current)
            if save is not None:
                chosen = save

        # Prioridad 1: break-bridge (atacar conexion del oponente)
        if chosen is None and last is not None:
            brk = _check_break_bridge(b, size, last, current)
            if brk is not None:
                chosen = brk

        if chosen is None and last is not None and random.random() < NEIGHBOR_P:
            r, c = last
            nbrs = [(nr, nc) for nr, nc in get_neighbors(r, c, size)
                    if b[nr][nc] == 0]
            if nbrs:
                chosen = nbrs[random.randrange(len(nbrs))]

        if chosen is None and len(pool) >= DIRECTION_K \
                and random.random() < DIRECTION_P:
            samples = [pool.cells[random.randrange(len(pool.cells))]
                       for _ in range(DIRECTION_K)]
            if current == 1:
                chosen = max(samples, key=lambda rc: rc[0])
            else:
                chosen = max(samples, key=lambda rc: rc[1])

        if chosen is None:
            chosen = pool.random()

        cr, cc = chosen
        b[cr][cc] = current
        pool.remove(chosen)
        filled += 1
        moves_played.append((chosen, current))

        if current == 1:
            if cr == 0:
                p1_top = True
            elif cr == size - 1:
                p1_bot = True
            if p1_top and p1_bot and check_winner(b, size) == 1:
                return (1.0 if root_player == 1 else 0.0), moves_played
        else:
            if cc == 0:
                p2_left = True
            elif cc == size - 1:
                p2_right = True
            if p2_left and p2_right and check_winner(b, size) == 2:
                return (1.0 if root_player == 2 else 0.0), moves_played

        last = chosen
        current = 3 - current

    return _soft_eval(b, size, root_player, current), moves_played


# ---------------------------------------------------------------------------
# Funciones MCTS standalone (usadas por main y workers)
# ---------------------------------------------------------------------------

def _mcts_select(node, board):
    b = _board_to_lists(board)
    while node.is_fully_expanded() and node.children:
        best_child = max(node.children, key=lambda c: c.uct_rave_score(node.visits))
        node = best_child
        b[node.move[0]][node.move[1]] = 3 - node.player_to_move
    return node, b


def _mcts_expand(node, board, size, trans_table=None):
    move = node.untried_moves.pop()
    b = [list(row) for row in board]
    b[move[0]][move[1]] = node.player_to_move
    next_player = 3 - node.player_to_move
    child_empties = [(r, c) for r in range(size) for c in range(size) if b[r][c] == 0]
    cands = _candidates(b, size, child_empties)
    random.shuffle(cands)
    child = _Node(move=move, parent=node, untried_moves=cands, player_to_move=next_player)
    if trans_table is not None:
        bkey = hash(tuple(tuple(row) for row in b))
        child._board_hash = bkey
        if bkey in trans_table:
            prior_v, prior_w = trans_table[bkey]
            if prior_v > TRANS_CAP:
                prior_w = prior_w * TRANS_CAP / prior_v
                prior_v = TRANS_CAP
            child.visits += prior_v
            child.wins   += prior_w
    node.children.append(child)
    return child, b


def _mcts_backpropagate(node, result, sim_moves, trans_table=None):
    amaf = defaultdict(set)
    for (move, player) in sim_moves:
        amaf[player].add(move)
    current = node
    while current is not None:
        current.visits += 1
        current.wins   += result
        if trans_table is not None and current._board_hash is not None:
            trans_table[current._board_hash] = (current.visits, current.wins)
        if current.parent is not None:
            player_moved = current.parent.player_to_move
            for m in amaf[player_moved]:
                current.parent.rave_visits[m] += 1
                current.parent.rave_wins[m]   += result
        current = current.parent


def _build_root(board, size, player, empties):
    """Construye nodo raiz con candidatos ordenados por FPU."""
    cands = _candidates(board, size, empties)
    if not cands:
        cands = list(empties)
    n_empty = len(empties)
    if n_empty == size * size:
        center = (size // 2, size // 2)
        rest = [m for m in cands if m != center]
        random.shuffle(rest)
        cands = rest + ([center] if center in cands else rest[-1:])
    else:
        cands = _fpu_order(board, size, cands, player)
    return _Node(move=None, parent=None, untried_moves=cands, player_to_move=player)


def _worker_run(args):
    """Worker independiente: MCTS por `duration` segundos. Retorna {move: visits}."""
    board_tuple, size, player, duration, seed = args
    random.seed(seed)
    t0 = time.monotonic()
    deadline = t0 + duration

    empties = [(r, c) for r in range(size) for c in range(size)
               if board_tuple[r][c] == 0]
    if not empties:
        return {}

    root = _build_root(board_tuple, size, player, empties)

    while time.monotonic() < deadline:
        node, b_sim = _mcts_select(root, board_tuple)
        if node.untried_moves:
            node, b_sim = _mcts_expand(node, b_sim, size, None)
        pool = _EmptyPool(b_sim, size)
        filled = size * size - len(pool)
        result, sim_moves = _fast_rollout(
            b_sim, size, node.player_to_move, player, pool, filled
        )
        _mcts_backpropagate(node, result, sim_moves, None)

    return {child.move: child.visits for child in root.children}


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------

class MiEstrategia(Strategy):

    @property
    def name(self) -> str:
        return "gabriel_regina"

    def begin_game(self, config: GameConfig) -> None:
        self._size       = config.board_size
        self._player     = config.player
        self._opponent   = config.opponent
        self._time_limit = config.time_limit
        self._variant    = config.variant

        self._hidden_opp   = set()
        self._my_moves     = set()
        self._failed_moves = set()

        self._root         = None
        self._last_my_move = None
        self._trans_table  = {}
        self._move_count   = 0

        # Terminar pool anterior y crear nuevo con fork
        old_pool = getattr(self, '_pool', None)
        if old_pool is not None:
            try:
                old_pool.terminate()
                old_pool.join()
            except Exception:
                pass
        try:
            ctx = mp.get_context('fork')
            self._pool = ctx.Pool(NUM_WORKERS)
        except Exception:
            self._pool = None

    def on_move_result(self, move, success):
        if success:
            self._my_moves.add(move)
        else:
            self._failed_moves.add(move)
            self._hidden_opp.add(move)

    def play(self, board, last_move):
        t0 = time.monotonic()
        size = self._size

        self._move_count += 1
        budget = min(0.97, TIME_BUDGET + 0.06 / (1.0 + self._move_count * 0.25))
        duration = self._time_limit * budget
        deadline = t0 + duration

        if self._variant == "dark":
            board = self._determinize(board)
            self._root = None

        empties = empty_cells(board, size)
        if len(empties) == 1:
            self._reset_tree(empties[0])
            return empties[0]

        # Victoria inmediata
        for m in empties:
            brd = _board_to_lists(board)
            brd[m[0]][m[1]] = self._player
            if check_winner(brd, size) == self._player:
                self._reset_tree(m)
                return m

        # Bloqueo de victoria del oponente
        for m in empties:
            brd = _board_to_lists(board)
            brd[m[0]][m[1]] = self._opponent
            if check_winner(brd, size) == self._opponent:
                self._reset_tree(m)
                return m

        # Enviar workers paralelos
        board_tuple = (board if isinstance(board[0], tuple)
                       else tuple(tuple(r) for r in board))
        worker_duration = max(0.1, duration - 0.15)
        async_result = None
        if self._pool is not None:
            try:
                args_list = [
                    (board_tuple, size, self._player,
                     worker_duration, random.randint(0, 2**31))
                    for _ in range(NUM_WORKERS)
                ]
                async_result = self._pool.map_async(_worker_run, args_list)
            except Exception:
                async_result = None

        # MCTS en proceso principal (con tree reuse y tabla de transposicion)
        root = None
        if self._variant == "classic":
            root = self._descend_root(last_move)
        if root is None:
            root = _build_root(board, size, self._player, empties)

        while time.monotonic() < deadline:
            node, b_sim = _mcts_select(root, board)
            if node.untried_moves:
                node, b_sim = _mcts_expand(node, b_sim, size, self._trans_table)
            pool_obj = _EmptyPool(b_sim, size)
            filled = size * size - len(pool_obj)
            result, sim_moves = _fast_rollout(
                b_sim, size, node.player_to_move, self._player, pool_obj, filled
            )
            _mcts_backpropagate(node, result, sim_moves, self._trans_table)

        # Agregar votos de workers
        vote_counts: dict = defaultdict(int)
        for child in root.children:
            vote_counts[child.move] += child.visits

        if async_result is not None:
            try:
                worker_results = async_result.get(timeout=2.0)
                for worker_votes in worker_results:
                    for move, v in worker_votes.items():
                        vote_counts[move] += v
            except Exception:
                pass

        if not vote_counts:
            move = random.choice(empties)
            self._reset_tree(move)
            return move

        best = max(vote_counts, key=vote_counts.get)
        self._root = root
        self._last_my_move = best
        return best

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------
    def _reset_tree(self, my_move):
        self._root = None
        self._last_my_move = my_move

    def _descend_root(self, opp_last_move):
        if self._root is None or self._last_my_move is None or opp_last_move is None:
            return None
        my_child = None
        for c in self._root.children:
            if c.move == self._last_my_move:
                my_child = c
                break
        if my_child is None:
            return None
        opp_child = None
        for c in my_child.children:
            if c.move == opp_last_move:
                opp_child = c
                break
        if opp_child is None:
            return None
        opp_child.parent = None
        return opp_child

    # ------------------------------------------------------------------
    # Dark mode: determinizacion
    # ------------------------------------------------------------------
    def _determinize(self, board):
        size = self._size
        known_opp = self._hidden_opp
        b = _board_to_lists(board)

        for (r, c) in known_opp:
            if b[r][c] == 0:
                b[r][c] = self._opponent

        my_count    = len(self._my_moves)
        known_count = len(known_opp)
        estimated_hidden = max(0, my_count - known_count - 1)

        available = [
            (r, c) for r in range(size) for c in range(size)
            if b[r][c] == 0 and (r, c) not in self._failed_moves
        ]

        center = size / 2
        def weight(rc):
            r, c = rc
            dist = abs(r - center) + abs(c - center)
            return max(1, size - dist)

        n_place = min(estimated_hidden, len(available))
        if n_place > 0:
            avail_copy = list(available)
            chosen = []
            for _ in range(n_place):
                if not avail_copy:
                    break
                total_w = sum(weight(rc) for rc in avail_copy)
                r_val = random.random()
                cumul = 0.0
                idx = 0
                for i, rc in enumerate(avail_copy):
                    cumul += weight(rc) / total_w
                    if r_val <= cumul:
                        idx = i
                        break
                chosen.append(avail_copy[idx])
                avail_copy.pop(idx)

            for (r, c) in chosen:
                b[r][c] = self._opponent

        return tuple(tuple(row) for row in b)
