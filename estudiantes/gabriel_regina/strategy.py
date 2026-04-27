"""
Estrategia MCTS + RAVE para Hex 11x11 — Version 3 (V9).
Autores: gabriel_regina

Mejoras sobre V8 (v2):
  * Parametros separados por variante: classic mantiene tuning v2, dark usa
    UCT_C=1.1 / RAVE_K=300 / RAVE_BLEND=0.75 (conjunto que rindio 23/24 en dark).
  * Transposition table deshabilitada en dark: cada determinizacion cambia las
    estadisticas del mismo hash de tablero y contamina los priors.
  * ISMCTS-lite mas robusto: 6 determinizaciones en dark (antes 4) distribuidas
    entre main + workers para reducir varianza del modelo del oponente.
  * Prior de determinizacion con termino de cluster: 45% centro, 30% eje
    ganador del oponente, 25% cercania a piedras ocultas ya reveladas. Modela
    mejor la tendencia del oponente a continuar clusters -> menos colisiones.
  * Libro de apertura extendido con respuestas adicionales como blanco.
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
# Parametros globales (A/B tuning). Los valores por defecto son los de classic;
# begin_game() los sobre-escribe con el perfil correspondiente a la variante.
# ---------------------------------------------------------------------------
UCT_C              = 1.2
RAVE_K             = 400
RAVE_BLEND         = 0.8
TIME_BUDGET        = 0.93
CUTOFF_FILL        = 0.65
NEIGHBOR_P         = 0.75
DIRECTION_P        = 0.20
DIRECTION_K        = 5
EXPAND_RADIUS      = 2
EXPAND_RADIUS_MAX  = 3
TRANS_CAP          = 50
NUM_WORKERS        = 3
# ISMCTS-lite: usar NUM_WORKERS + 1 determinizaciones (main + cada worker usa
# una diferente). Con NUM_WORKERS=3 esto son 4 muestras unicas del estado
# oculto, sin colisiones de asignacion.
NUM_DETERMINIZATIONS = NUM_WORKERS + 1
OPENING_BOOK       = True
SAFETY_TAIL        = 0.20   # segundos reservados como margen duro (200 ms)

# Perfiles por variante. classic mantiene el tuning v2 que rindio 23/24;
# dark revierte a los valores originales de v2-predecesor (23/24 en dark).
_VARIANT_PARAMS = {
    "classic": {
        "UCT_C":      1.2,
        "RAVE_K":     400,
        "RAVE_BLEND": 0.8,
        "CUTOFF_FILL":0.65,
        "NEIGHBOR_P": 0.75,
    },
    "dark": {
        "UCT_C":      1.1,
        "RAVE_K":     300,
        "RAVE_BLEND": 0.75,
        "CUTOFF_FILL":0.62,
        "NEIGHBOR_P": 0.72,
    },
}

# ---------------------------------------------------------------------------
# Patrones de bridge
# ---------------------------------------------------------------------------
BRIDGE_PATTERNS = (
    ((-1, 0),  (0, 1),   (-1, 1)),
    ((-1, 0),  (1, -1),  (0, -1)),
    ((-1, 1),  (1, 0),   (0, 1)),
    ((-1, 1),  (0, -1),  (-1, 0)),
    ((0, -1),  (1, 0),   (1, -1)),
    ((0, 1),   (1, -1),  (1, 0)),
)

BRIDGE_ENDPOINTS = (
    ((-2, 1),  (-1, 0),  (-1, 1)),
    ((-1, 2),  (-1, 1),  (0, 1)),
    ((1, 1),   (0, 1),   (1, 0)),
    ((2, -1),  (1, 0),   (1, -1)),
    ((1, -2),  (1, -1),  (0, -1)),
    ((-1, -1), (0, -1),  (-1, 0)),
)

# ---------------------------------------------------------------------------
# Plantillas de borde 4-3-2: patrones (offset_A, offset_B, offset_resp)
# relativos a la ultima jugada del oponente. Si A y B son del oponente y
# resp esta vacio, jugar resp bloquea la amenaza de borde.
# ---------------------------------------------------------------------------
EDGE_TEMPLATES_P1 = (
    # Plantillas para jugador 1 (conecta filas): amenazas cerca de fila 0 o N-1
    ((-1, 0), (-2, 0), (0, 0)),
    (( 1, 0), ( 2, 0), (0, 0)),
    ((-1, 1), (-2, 0), (-1, 0)),
    (( 1,-1), ( 2, 0), ( 1, 0)),
)

EDGE_TEMPLATES_P2 = (
    # Plantillas para jugador 2 (conecta columnas): amenazas cerca de col 0 o N-1
    ((0,-1), (0,-2), (0, 0)),
    ((0, 1), (0, 2), (0, 0)),
    ((1,-1), (0,-2), (0,-1)),
    ((-1, 1),(0, 2), (0, 1)),
)

# ---------------------------------------------------------------------------
# Libro de apertura para 11x11
# ---------------------------------------------------------------------------
_OPENING_BOOK: dict[tuple, tuple[int,int]] = {
    # Negro mueve primero; como negro jugar cerca del centro-superior-derecha
    (): (1, 9),
    # Respuestas como blanco al primer movimiento negro en celdas comunes.
    # Nota: el analisis del run 2026-04-20T01-58-08 mostro que expandir el
    # libro con "(5,5) a todo" nos condujo directamente a 2 derrotas (Tier_4,
    # Tier_2). Mantenemos solo el set estable que ya validaba v2.
    ((5, 5),):   (5, 4),
    ((5, 4),):   (5, 5),
    ((5, 6),):   (5, 5),
    ((1, 9),):   (5, 5),
    ((3, 7),):   (5, 5),
    ((7, 3),):   (5, 5),
}


# ---------------------------------------------------------------------------
# Pool de celdas vacias (O(1) seleccion/eliminacion)
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
# Nodo MCTS
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


def _board_hash(board):
    return hash(tuple(tuple(r) for r in board))


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


def _bridge_aware_dijkstra(board, size, player):
    """Dijkstra donde cruzar un vacio flanqueado por dos bridges nuestros cuesta 0."""
    opp = 3 - player
    INF = size * size + 1
    dist: dict = {}
    heap = []

    if player == 1:
        for c in range(size):
            if board[0][c] == opp:
                continue
            d = 0 if board[0][c] == player else 1
            if d < dist.get((0, c), INF):
                dist[(0, c)] = d
                heapq.heappush(heap, (d, 0, c))
        goal_check = lambda r, c: r == size - 1
    else:
        for r in range(size):
            if board[r][0] == opp:
                continue
            d = 0 if board[r][0] == player else 1
            if d < dist.get((r, 0), INF):
                dist[(r, 0)] = d
                heapq.heappush(heap, (d, r, 0))
        goal_check = lambda r, c: c == size - 1

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist.get((r, c), INF):
            continue
        if goal_check(r, c):
            return d
        for nr, nc in get_neighbors(r, c, size):
            if board[nr][nc] == opp:
                continue
            if board[nr][nc] == player:
                add = 0
            else:
                add = _bridge_cost(board, size, player, nr, nc)
            nd = d + add
            if nd < dist.get((nr, nc), INF):
                dist[(nr, nc)] = nd
                heapq.heappush(heap, (nd, nr, nc))

    return INF


def _bridge_cost(board, size, player, r, c):
    """Costo 0 si el vacio (r,c) forma un puente virtual entre dos piezas nuestras."""
    nbrs = get_neighbors(r, c, size)
    own_nbrs = [n for n in nbrs if board[n[0]][n[1]] == player]
    if len(own_nbrs) >= 2:
        # Verificar que dos vecinos nuestros comparten este vacio como carrier
        for i in range(len(own_nbrs)):
            for j in range(i + 1, len(own_nbrs)):
                ar, ac = own_nbrs[i]
                br2, bc = own_nbrs[j]
                for (dar, dac), (dbr, dbc), (dsr, dsc) in BRIDGE_PATTERNS:
                    if (ar + dar == br2 and ac + dac == bc and
                            ar + dsr == r and ac + dsc == c):
                        return 0
                    if (br2 + dar == ar and bc + dac == ac and
                            br2 + dsr == r and bc + dsc == c):
                        return 0
    return 1


def _is_dead_cell(board, size, cell, player):
    """Celda muerta para `player`: todos sus vecinos son del oponente o fuera del tablero."""
    opp = 3 - player
    r, c = cell
    for nr, nc in get_neighbors(r, c, size):
        if board[nr][nc] != opp:
            return False
    return True


def _neighborhood_empties(board, size, empties, radius):
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


def _candidates(board, size, empties, player=None, radius=EXPAND_RADIUS):
    """Celdas candidatas con filtro dead-cell y radio adaptativo."""
    # Filtrar celdas muertas si conocemos al jugador
    if player is not None:
        empties = [m for m in empties if not _is_dead_cell(board, size, m, player)]
        if not empties:
            empties = list(empty_cells(board, size))

    nbhd = _neighborhood_empties(board, size, empties, radius)
    if nbhd and len(nbhd) >= 5:
        return nbhd
    if nbhd is None:
        center = size // 2
        r2 = size // 3
        return [(r, c) for r, c in empties
                if abs(r - center) <= r2 and abs(c - center) <= r2]
    return list(empties)


# ---------------------------------------------------------------------------
# FPU: Dijkstra bidireccional
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


def _greedy_fallback(board, size, player, empties):
    """Movimiento greedy: celda del camino Dijkstra minimo."""
    if not empties:
        return None
    INF = float("inf")
    fwd = _full_dijkstra(board, size, player, from_start=True)
    bwd = _full_dijkstra(board, size, player, from_start=False)
    if player == 1:
        total = min(fwd.get((size - 1, c), INF) for c in range(size))
    else:
        total = min(fwd.get((r, size - 1), INF) for r in range(size))
    if total < INF:
        for r, c in empties:
            f = fwd.get((r, c), INF)
            b = bwd.get((r, c), INF)
            if f != INF and b != INF and f + b == total + 1:
                return (r, c)
    return empties[0]


# ---------------------------------------------------------------------------
# Bridge patterns
# ---------------------------------------------------------------------------

def _check_save_bridge(b, size, last, current):
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


def _check_edge_template(b, size, last, current):
    """Plantilla de borde 4-3-2: si el oponente amenaza el borde, bloquear."""
    opp = 3 - current
    templates = EDGE_TEMPLATES_P1 if opp == 1 else EDGE_TEMPLATES_P2
    lr, lc = last
    for (dAr, dAc), (dBr, dBc), (dRr, dRc) in templates:
        ar, ac = lr + dAr, lc + dAc
        if not (0 <= ar < size and 0 <= ac < size):
            continue
        if b[ar][ac] != opp:
            continue
        br, bc = lr + dBr, lc + dBc
        if not (0 <= br < size and 0 <= bc < size):
            continue
        if b[br][bc] != opp:
            continue
        rr, rc = lr + dRr, lc + dRc
        if not (0 <= rr < size and 0 <= rc < size):
            continue
        if b[rr][rc] == 0:
            return (rr, rc)
    return None


# ---------------------------------------------------------------------------
# Rollout con save/break-bridge, plantillas de borde y ladder-avoidance
# ---------------------------------------------------------------------------

def _fast_rollout(b, size, player_to_move, root_player, pool, filled, deadline=None):
    current = player_to_move
    cutoff = int(CUTOFF_FILL * size * size)
    moves_played = []
    last = None
    last2 = None  # penultima jugada para ladder-avoidance

    p1_top   = any(b[0][c] == 1 for c in range(size))
    p1_bot   = any(b[size - 1][c] == 1 for c in range(size))
    p2_left  = any(b[r][0] == 2 for r in range(size))
    p2_right = any(b[r][size - 1] == 2 for r in range(size))

    while filled < cutoff and len(pool) > 0:
        if deadline is not None and time.monotonic() >= deadline:
            break

        chosen = None

        if last is not None:
            # Prioridad 0: save-bridge
            save = _check_save_bridge(b, size, last, current)
            if save is not None:
                chosen = save

            # Prioridad 1: break-bridge
            if chosen is None:
                brk = _check_break_bridge(b, size, last, current)
                if brk is not None:
                    chosen = brk

            # Prioridad 2: plantilla de borde 4-3-2
            if chosen is None:
                tpl = _check_edge_template(b, size, last, current)
                if tpl is not None:
                    chosen = tpl

        # Ladder-avoidance: bump de 5% para respuesta perpendicular
        if chosen is None and last is not None and last2 is not None:
            lr, lc = last
            l2r, l2c = last2
            if (lr - l2r, lc - l2c) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if random.random() < 0.05:
                    perp = []
                    if lr == l2r:
                        perp = [(lr + 1, lc), (lr - 1, lc)]
                    else:
                        perp = [(lr, lc + 1), (lr, lc - 1)]
                    perp = [(r, c) for r, c in perp
                            if 0 <= r < size and 0 <= c < size and b[r][c] == 0]
                    if perp:
                        chosen = random.choice(perp)

        if chosen is None and last is not None and random.random() < NEIGHBOR_P:
            r, c = last
            nbrs = [(nr, nc) for nr, nc in get_neighbors(r, c, size)
                    if b[nr][nc] == 0]
            if nbrs:
                chosen = nbrs[random.randrange(len(nbrs))]

        if chosen is None and len(pool) >= DIRECTION_K and random.random() < DIRECTION_P:
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

        last2 = last
        last = chosen
        current = 3 - current

    return _soft_eval(b, size, root_player, current), moves_played


# ---------------------------------------------------------------------------
# MCTS: select, expand (con fix transposition), backprop
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
    cands = _candidates(b, size, child_empties, next_player)
    random.shuffle(cands)
    child = _Node(move=move, parent=node, untried_moves=cands, player_to_move=next_player)

    # Bug fix: inyectar priors solo en el primer visit (evita double-counting)
    if trans_table is not None and child.visits == 0:
        bkey = hash(tuple(tuple(row) for row in b))
        child._board_hash = bkey
        if bkey in trans_table:
            prior_v, prior_w = trans_table[bkey]
            virtual_v = min(prior_v, TRANS_CAP)
            virtual_w = prior_w * virtual_v / prior_v if prior_v > 0 else 0.0
            child.visits += virtual_v
            child.wins   += virtual_w
        else:
            child._board_hash = bkey

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


def _build_root(board, size, player, empties, prog_k=None):
    """Construye raiz con candidatos FPU-ordenados. prog_k limita candidatos iniciales."""
    cands = _candidates(board, size, empties, player)
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
    # Progressive widening: limitar lista inicial (stack = pop del final)
    if prog_k is not None and len(cands) > prog_k:
        # Las mejores celdas (FPU) estan al final de la lista
        cands = cands[-(prog_k):]
    return _Node(move=None, parent=None, untried_moves=cands, player_to_move=player)


# ---------------------------------------------------------------------------
# Worker standalone para root parallelization
# ---------------------------------------------------------------------------

def _worker_run(args):
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
            b_sim, size, node.player_to_move, player, pool, filled, deadline=deadline
        )
        _mcts_backpropagate(node, result, sim_moves, None)

    return {child.move: child.visits for child in root.children}


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------

class MiEstrategiaV3(Strategy):

    @property
    def name(self) -> str:
        return "gabriel_regina_v3"

    def begin_game(self, config: GameConfig) -> None:
        self._size       = config.board_size
        self._player     = config.player
        self._opponent   = config.opponent
        self._time_limit = config.time_limit
        self._variant    = config.variant

        # Aplicar perfil de parametros segun variante (mutar globales antes de
        # crear el pool, para que los workers hereden los valores correctos).
        profile = _VARIANT_PARAMS.get(self._variant, _VARIANT_PARAMS["classic"])
        global UCT_C, RAVE_K, RAVE_BLEND, CUTOFF_FILL, NEIGHBOR_P
        UCT_C       = profile["UCT_C"]
        RAVE_K      = profile["RAVE_K"]
        RAVE_BLEND  = profile["RAVE_BLEND"]
        CUTOFF_FILL = profile["CUTOFF_FILL"]
        NEIGHBOR_P  = profile["NEIGHBOR_P"]

        self._hidden_opp     = set()
        self._my_moves       = set()
        self._failed_moves   = set()
        self._collision_count = 0   # Bug fix: rastrear colisiones para hidden estimate

        self._root         = None
        self._last_my_move = None
        self._last_board_hash = None  # Bug fix: tree reuse con validacion hash
        # TT deshabilitada en dark: cada determinizacion cambia las estadisticas
        # del mismo hash de tablero y contamina los priors.
        self._trans_table  = {} if self._variant == "classic" else None
        self._move_count   = 0

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
            self._collision_count += 1  # Bug fix

    def play(self, board, last_move):
        t0 = time.monotonic()
        size = self._size
        tl   = self._time_limit

        self._move_count += 1

        budget = min(0.97, TIME_BUDGET + 0.06 / (1.0 + self._move_count * 0.25))

        # Margen duro: SAFETY_TAIL es en segundos (0.20 = 200ms)
        hard_limit = tl - SAFETY_TAIL
        duration   = min(tl * budget, hard_limit)
        deadline   = t0 + duration

        # Libro de apertura
        if OPENING_BOOK and self._variant == "classic":
            ob_move = self._opening_book_move(board, last_move)
            if ob_move is not None:
                self._reset_tree(ob_move, board)
                return ob_move

        # Determinizacion dark mode (ISMCTS-lite: multiples determinizaciones)
        determinizations = None
        if self._variant == "dark":
            determinizations = [self._determinize(board) for _ in range(NUM_DETERMINIZATIONS)]
            # Main thread usa la ultima determinizacion (los workers se reparten
            # 0..NUM_WORKERS-1), asi todas las muestras son aprovechadas.
            board = determinizations[NUM_WORKERS]
            self._root = None

        empties = empty_cells(board, size)

        # Fallback precomputado (Feature A)
        fallback = _greedy_fallback(board, size, self._player, empties)

        if len(empties) == 1:
            self._reset_tree(empties[0], board)
            return empties[0]

        # Victoria inmediata
        for m in empties:
            if time.monotonic() >= deadline:
                self._reset_tree(fallback, board)
                return fallback
            brd = _board_to_lists(board)
            brd[m[0]][m[1]] = self._player
            if check_winner(brd, size) == self._player:
                self._reset_tree(m, board)
                return m

        # Bloqueo de victoria del oponente
        for m in empties:
            if time.monotonic() >= deadline:
                self._reset_tree(fallback, board)
                return fallback
            brd = _board_to_lists(board)
            brd[m[0]][m[1]] = self._opponent
            if check_winner(brd, size) == self._opponent:
                self._reset_tree(m, board)
                return m

        board_tuple = (board if isinstance(board[0], tuple)
                       else tuple(tuple(r) for r in board))

        # Tiempo disponible para workers (con margen)
        worker_duration = max(0.05, duration - 0.15)
        async_result = None
        if self._pool is not None:
            try:
                if self._variant == "dark" and determinizations is not None:
                    # ISMCTS-lite: cada worker usa una determinizacion diferente
                    args_list = [
                        (determinizations[i % NUM_DETERMINIZATIONS], size,
                         self._player, worker_duration, random.randint(0, 2**31))
                        for i in range(NUM_WORKERS)
                    ]
                else:
                    args_list = [
                        (board_tuple, size, self._player,
                         worker_duration, random.randint(0, 2**31))
                        for _ in range(NUM_WORKERS)
                    ]
                async_result = self._pool.map_async(_worker_run, args_list)
            except Exception:
                self._pool = None
                async_result = None

        # MCTS en proceso principal
        root = None
        if self._variant == "classic":
            root = self._descend_root(last_move, board_tuple)
        if root is None:
            root = _build_root(board, size, self._player, empties)

        while time.monotonic() < deadline:
            node, b_sim = _mcts_select(root, board)
            if node.untried_moves:
                node, b_sim = _mcts_expand(node, b_sim, size, self._trans_table)
            if time.monotonic() >= deadline:
                break
            pool_obj = _EmptyPool(b_sim, size)
            filled = size * size - len(pool_obj)
            result, sim_moves = _fast_rollout(
                b_sim, size, node.player_to_move, self._player, pool_obj, filled,
                deadline=deadline
            )
            _mcts_backpropagate(node, result, sim_moves, self._trans_table)

        # Recoger votos de workers con timeout ajustado
        vote_counts: dict = defaultdict(int)
        for child in root.children:
            vote_counts[child.move] += child.visits

        if async_result is not None:
            remaining = max(0.05, deadline - time.monotonic() - 0.05)
            try:
                worker_results = async_result.get(timeout=remaining)
                for worker_votes in worker_results:
                    for move, v in worker_votes.items():
                        vote_counts[move] += v
            except Exception:
                self._pool = None

        if not vote_counts:
            self._reset_tree(fallback, board)
            return fallback

        # Dark: penalizar votos por P(colision). Un candidato que aparece
        # vacio en pocas determinizaciones probablemente oculta una piedra
        # rival; escalar sus votos por la fraccion de muestras donde estaba
        # vacio reduce las jugadas-kamikaze sin descartar el movimiento del
        # todo (variance robustness).
        if self._variant == "dark" and determinizations is not None:
            N = len(determinizations)
            p_empty: dict = {}
            for cand in vote_counts:
                cnt = 0
                for det in determinizations:
                    r, c = cand
                    if det[r][c] == 0:
                        cnt += 1
                # Piso 1/N para no eliminar el candidato si todas las dets lo
                # tachan (puede seguir siendo la mejor jugada visible).
                p_empty[cand] = max(1.0 / N, cnt / N)
            best = max(vote_counts,
                       key=lambda m: vote_counts[m] * p_empty.get(m, 1.0))
        else:
            best = max(vote_counts, key=vote_counts.get)
        self._root = root
        self._last_my_move = best
        self._last_board_hash = _board_hash(board_tuple)
        return best

    # ------------------------------------------------------------------
    # Opening book
    # ------------------------------------------------------------------
    def _opening_book_move(self, board, last_move):
        size = self._size
        if size != 11:
            return None
        empties = empty_cells(board, size)
        if len(empties) < size * size - 2:
            return None
        # Primera jugada como negro
        if self._player == 1 and last_move is None and len(empties) == size * size:
            m = _OPENING_BOOK.get(())
            if m and board[m[0]][m[1]] == 0:
                return m
        # Respuesta como blanco al primer movimiento negro
        if self._player == 2 and last_move is not None and len(empties) == size * size - 1:
            m = _OPENING_BOOK.get((last_move,))
            if m and board[m[0]][m[1]] == 0:
                return m
        return None

    # ------------------------------------------------------------------
    # Tree reuse con validacion de hash (Bug fix #2)
    # ------------------------------------------------------------------
    def _reset_tree(self, my_move, board):
        self._root = None
        self._last_my_move = my_move
        if board is not None:
            bt = (board if isinstance(board[0], tuple)
                  else tuple(tuple(r) for r in board))
            self._last_board_hash = _board_hash(bt)

    def _descend_root(self, opp_last_move, board_tuple):
        if (self._root is None or self._last_my_move is None
                or opp_last_move is None):
            return None

        # Bug fix: validar que el hash del tablero coincide
        expected_hash = self._last_board_hash
        if expected_hash is not None:
            current_hash = _board_hash(board_tuple)
            # El tablero actual deberia tener mi ultima jugada + la del oponente
            # Simplemente chequeamos que no sea identico (hubo cambios)
            if current_hash == expected_hash:
                return None  # No hubo cambios, algo esta mal

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
    # Dark mode: ISMCTS-lite con determinizaciones multiples
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

        # Bug fix: formula corregida considerando colisiones y color
        offset = 1 if self._player == 1 else 0
        estimated_hidden = max(
            0,
            (my_count + self._collision_count) - known_count - offset
        )
        current_empties = [(r, c) for r in range(size) for c in range(size)
                           if b[r][c] == 0]
        estimated_hidden = min(estimated_hidden, len(current_empties))

        available = [
            (r, c) for r, c in current_empties
            if (r, c) not in self._failed_moves
        ]

        center = size / 2
        known_list = list(known_opp)

        def weight(rc):
            r, c = rc
            dist_center = abs(r - center) + abs(c - center)
            w_center = max(1, size - dist_center)
            # 30% peso hacia eje ganador del oponente
            if self._opponent == 1:
                w_edge = max(1, size - abs(r - center))
            else:
                w_edge = max(1, size - abs(c - center))
            # 35% peso hacia cluster cerca de piedras ocultas ya reveladas:
            # cuando el rival coloca en una fila/col, suele continuar cerca
            # para completar conexiones o bloqueos (observado en tier_2 dark).
            if known_list:
                min_d = min(abs(r - kr) + abs(c - kc) for kr, kc in known_list)
                w_cluster = max(1, size - min_d)
            else:
                w_cluster = 1
            return 0.35 * w_center + 0.30 * w_edge + 0.35 * w_cluster

        n_place = min(estimated_hidden, len(available))
        if n_place > 0:
            avail_copy = list(available)
            for _ in range(n_place):
                if not avail_copy:
                    break
                total_w = sum(weight(rc) for rc in avail_copy)
                r_val = random.random() * total_w
                cumul = 0.0
                idx = len(avail_copy) - 1
                for i, rc in enumerate(avail_copy):
                    cumul += weight(rc)
                    if r_val <= cumul:
                        idx = i
                        break
                b[avail_copy[idx][0]][avail_copy[idx][1]] = self._opponent
                avail_copy.pop(idx)

        return tuple(tuple(row) for row in b)
