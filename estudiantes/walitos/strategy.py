"""
Estrategia MCTS + PUCT para Hex. Solo depende de numpy.

CLASSIC: MCTS + PUCT con prior de red de resistencia + VC solver + endgame.
DARK:    Estrategia paranoid completamente distinta:
         - Tablero paranoid: asume que el rival ocupa sus celdas más valiosas
         - Two-path robustness: busca movidas que pertenecen a múltiples caminos
         - Bridges integrado en la selección de movidas
         - IS-MCTS corto sobre tablero paranoid para refinar
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from strategy import Strategy, GameConfig

# ---------------------------------------------------------------------------
# Tablas precomputadas
# ---------------------------------------------------------------------------

_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

_BOOK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "opening_book.json"
)

_NB11: list[list[list[tuple[int, int]]]] = [
    [
        [
            (r + dr, c + dc)
            for dr, dc in _NEIGHBORS
            if 0 <= r + dr < 11 and 0 <= c + dc < 11
        ]
        for c in range(11)
    ]
    for r in range(11)
]

_ENDGAME_THRESHOLD = 20
_C_PUCT = 1.2
_FPU_REDUCTION = 0.20


def _get_neighbors(r: int, c: int, size: int) -> list[tuple[int, int]]:
    if size == 11:
        return _NB11[r][c]
    return [
        (r + dr, c + dc)
        for dr, dc in _NEIGHBORS
        if 0 <= r + dr < size and 0 <= c + dc < size
    ]


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class _UF:
    __slots__ = ["p", "rk"]

    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.rk = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rk[px] < self.rk[py]:
            px, py = py, px
        self.p[py] = px
        if self.rk[px] == self.rk[py]:
            self.rk[px] += 1

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


def _make_rollout_uf(board, size):
    N = size * size
    SRC1 = N
    SNK1 = N + 1
    SRC2 = N + 2
    SNK2 = N + 3
    uf1 = _UF(N + 2)
    uf2 = _UF(N + 4)
    for r in range(size):
        for c in range(size):
            v = board[r][c]
            if v == 0:
                continue
            i = r * size + c
            if v == 1:
                for nr, nc in _get_neighbors(r, c, size):
                    if board[nr][nc] == 1:
                        uf1.union(i, nr * size + nc)
                if r == 0:
                    uf1.union(i, SRC1)
                if r == size - 1:
                    uf1.union(i, SNK1)
            else:
                for nr, nc in _get_neighbors(r, c, size):
                    if board[nr][nc] == 2:
                        uf2.union(i, nr * size + nc)
                if c == 0:
                    uf2.union(i, SRC2)
                if c == size - 1:
                    uf2.union(i, SNK2)
    return uf1, uf2, SRC1, SNK1, SRC2, SNK2


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------


def _check_winner_local(board, size: int) -> int:
    for player in (1, 2):
        visited: set = set()
        stack = []
        if player == 1:
            starts = [(0, c) for c in range(size) if board[0][c] == 1]
            goal = lambda r, c: r == size - 1
        else:
            starts = [(r, 0) for r in range(size) if board[r][0] == 2]
            goal = lambda r, c: c == size - 1
        for cell in starts:
            if cell not in visited:
                visited.add(cell)
                stack.append(cell)
        while stack:
            r, c = stack.pop()
            if goal(r, c):
                return player
            for nr, nc in _get_neighbors(r, c, size):
                if (nr, nc) not in visited and board[nr][nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
    return 0


def _board_key(board, size: int) -> str:
    return ";".join(
        sorted(
            f"{board[r][c]}:{r},{c}"
            for r in range(size)
            for c in range(size)
            if board[r][c] != 0
        )
    )


def _dead_cells(board, size: int) -> set:
    dead: set = set()
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                continue
            nbs = _get_neighbors(r, c, size)
            if not nbs:
                continue
            if any(board[nr][nc] == 0 for nr, nc in nbs):
                continue
            if len({board[nr][nc] for nr, nc in nbs}) == 1:
                dead.add((r, c))
    return dead


# ---------------------------------------------------------------------------
# Red de resistencia eléctrica
# ---------------------------------------------------------------------------


def _resistance_prior(board, size: int, player: int) -> np.ndarray:
    """
    Prior P[r][c] via red de resistencia (Kirchhoff). Solo numpy.
    Conductancias: propia=1e6, vacío=1.0, rival=0.0.
    """
    N = size * size
    SRC = N
    SNK = N + 1
    opponent = 3 - player
    A = np.zeros((N + 2, N + 2), dtype=np.float64)
    b = np.zeros(N + 2, dtype=np.float64)

    def idx(r, c):
        return r * size + c

    def cond(r, c):
        v = board[r][c]
        if v == player:
            return 1e6
        if v == opponent:
            return 0.0
        return 1.0

    for r in range(size):
        for c in range(size):
            g_rc = cond(r, c)
            if g_rc == 0.0:
                continue
            i = idx(r, c)
            for nr, nc in _get_neighbors(r, c, size):
                g = min(g_rc, cond(nr, nc))
                if g == 0.0:
                    continue
                j = idx(nr, nc)
                A[i, i] += g
                A[i, j] -= g

    es = [(0, c) for c in range(size)] if player == 1 else [(r, 0) for r in range(size)]
    ee = (
        [(size - 1, c) for c in range(size)]
        if player == 1
        else [(r, size - 1) for r in range(size)]
    )
    for r, c in es:
        g = cond(r, c)
        if g == 0.0:
            continue
        i = idx(r, c)
        A[SRC, SRC] += g
        A[SRC, i] -= g
        A[i, i] += g
        A[i, SRC] -= g
    for r, c in ee:
        g = cond(r, c)
        if g == 0.0:
            continue
        i = idx(r, c)
        A[SNK, SNK] += g
        A[SNK, i] -= g
        A[i, i] += g
        A[i, SNK] -= g

    A[SRC, :] = 0.0
    A[SRC, SRC] = 1.0
    b[SRC] = 1.0
    A[SNK, :] = 0.0
    A[SNK, SNK] = 1.0
    b[SNK] = 0.0

    try:
        V = np.linalg.lstsq(A, b, rcond=None)[0]
    except Exception:
        prior = np.zeros((size, size))
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    prior[r][c] = 1.0
        t = prior.sum()
        return prior / t if t > 0 else prior

    prior = np.zeros((size, size))
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                continue
            i = idx(r, c)
            cur = 0.0
            for nr, nc in _get_neighbors(r, c, size):
                g = min(cond(r, c), cond(nr, nc))
                if g > 0:
                    cur += abs(g * (V[idx(nr, nc)] - V[i]))
            prior[r][c] = cur
    t = prior.sum()
    if t > 0:
        prior /= t
    return prior


# ---------------------------------------------------------------------------
# Tablero paranoid (dark mode)
# ---------------------------------------------------------------------------


def _bfs_distance_to_edge(
    board: list[list[int]],
    size: int,
    player: int,
    to_start_edge: bool,
) -> list[list[int]]:
    """
    Distancia BFS desde cada celda hasta el borde objetivo del jugador.
    - Piedras propias: costo 0 (las atravesamos gratis)
    - Celdas vacías:   costo 1
    - Piedras rivales: bloqueadas (infinito)

    Player 1: SRC=fila 0, SNK=fila size-1
    Player 2: SRC=col 0,  SNK=col size-1
    """
    import heapq

    INF = size * size + 1
    opponent = 3 - player
    dist = [[INF] * size for _ in range(size)]

    if player == 1:
        edge_cells = (
            [(0, c) for c in range(size)]
            if to_start_edge
            else [(size - 1, c) for c in range(size)]
        )
    else:
        edge_cells = (
            [(r, 0) for r in range(size)]
            if to_start_edge
            else [(r, size - 1) for r in range(size)]
        )

    heap = []
    for r, c in edge_cells:
        if board[r][c] == opponent:
            continue
        cost = 0 if board[r][c] == player else 1
        dist[r][c] = cost
        heapq.heappush(heap, (cost, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for nr, nc in _get_neighbors(r, c, size):
            if board[nr][nc] == opponent:
                continue
            step = 0 if board[nr][nc] == player else 1
            nd = d + step
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))

    return dist


def _two_distance_prior(
    board: list[list[int]],
    size: int,
    player: int,
) -> np.ndarray:
    """
    Heurística two-distance de Anshelevich (clásica para Hex AI).

    Score = -(max(d_inicio, d_fin) + 0.1 * min(d_inicio, d_fin))

    El término dominante es MAX, no SUM. Esto fuerza al algoritmo a
    minimizar la distancia al borde MÁS LEJANO. Sin esto, el algoritmo
    termina consolidando un cluster cerca de un borde mientras ignora
    el otro.

    Ejemplo (player 1 con cluster en filas 1-5):
    - Celda (0,7): d_top=1, d_bot=7. max=7. Score=-7.7
    - Celda (6,5): d_top=3, d_bot=5. max=5. Score=-5.3
    - (6,5) es preferida (extiende hacia abajo, edge más lejano).

    Con la versión de sum=12 ambas, la elección caía en el bonus de
    conexión, que prefiere consolidar el cluster existente.
    """
    d_start = _bfs_distance_to_edge(board, size, player, to_start_edge=True)
    d_end = _bfs_distance_to_edge(board, size, player, to_start_edge=False)
    INF = size * size + 1

    raw = np.full((size, size), -np.inf)
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                continue
            ds, de = d_start[r][c], d_end[r][c]
            if ds >= INF or de >= INF:
                continue
            # Anshelevich potential: minimizar max(d_to_each_edge)
            # con tiebreak suave por suma para preferir conexiones cortas
            raw[r][c] = -(max(ds, de) + 0.1 * min(ds, de))

    valid_mask = raw > -np.inf
    if not valid_mask.any():
        return np.zeros((size, size))

    valid = raw[valid_mask]
    s_min, s_max = valid.min(), valid.max()
    out = np.zeros((size, size))
    if s_max > s_min:
        for r in range(size):
            for c in range(size):
                if valid_mask[r, c]:
                    out[r, c] = (raw[r, c] - s_min) / (s_max - s_min)
    else:
        out[valid_mask] = 0.5

    return out


def _build_paranoid_board(
    board,
    size: int,
    player: int,
    n_hidden: int,
    opp_prior: np.ndarray,
) -> list[list[int]]:
    """
    Tablero paranoid: coloca las n_hidden piedras rivales en las celdas
    de MAYOR prior del rival. Representa el peor caso plausible.

    Diferencia clave vs determinización aleatoria:
    - La determinización aleatoria promedia escenarios → puede optimizar
      rutas que solo existen en algunos mundos.
    - El tablero paranoid asume el PEOR CASO → la movida elegida es buena
      en todos los escenarios, no solo en el promedio.
    """
    if n_hidden <= 0:
        return [list(row) for row in board]

    sim = [list(row) for row in board]
    opponent = 3 - player

    # Ordenar celdas vacías por prior del rival (descendente)
    empties = [(r, c) for r in range(size) for c in range(size) if sim[r][c] == 0]
    empties.sort(key=lambda rc: -float(opp_prior[rc[0]][rc[1]]))

    # Colocar en las top n_hidden posiciones
    for r, c in empties[:n_hidden]:
        sim[r][c] = opponent

    return sim


def _sample_determinized_board(
    board,
    size: int,
    player: int,
    n_hidden: int,
    opp_prior: np.ndarray,
) -> list[list[int]]:
    """
    Determinización aleatoria ponderada por prior del rival.
    Usada en IS-MCTS para diversidad de escenarios.
    """
    if n_hidden <= 0:
        return [list(row) for row in board]

    sim = [list(row) for row in board]
    opponent = 3 - player
    empties = [(r, c) for r in range(size) for c in range(size) if sim[r][c] == 0]
    if not empties:
        return sim

    weights = [float(opp_prior[r][c]) + 1e-4 for r, c in empties]
    n_place = min(n_hidden, len(empties))
    chosen: set = set()

    for _ in range(n_place):
        avail = [(cell, w) for cell, w in zip(empties, weights) if cell not in chosen]
        if not avail:
            break
        cells_a, wts_a = zip(*avail)
        total = sum(wts_a)
        rv = random.random() * total
        cum = 0.0
        sel = cells_a[-1]
        for cell, w in zip(cells_a, wts_a):
            cum += w
            if cum >= rv:
                sel = cell
                break
        chosen.add(sel)
        sim[sel[0]][sel[1]] = opponent

    return sim


# ---------------------------------------------------------------------------
# Two-path robustness (dark mode)
# ---------------------------------------------------------------------------


def _two_path_score(
    board,
    size: int,
    player: int,
    r0: int,
    c0: int,
) -> float:
    """
    Mide la robustez de jugar en (r0,c0): calcula la resistencia eléctrica
    con esa celda ocupada por nosotros. Una celda con alta corriente en DOS
    configuraciones distintas del rival es una celda robusta.

    Proxy simplificado: sum of prior values weighted by neighbor connectivity.
    """
    sim = [list(row) for row in board]
    sim[r0][c0] = player
    prior = _resistance_prior(sim, size, player)
    score = float(prior[r0][c0])
    # Bonus por vecinos propios (conectividad)
    for nr, nc in _get_neighbors(r0, c0, size):
        if sim[nr][nc] == player:
            score += 0.1
    return score


# ---------------------------------------------------------------------------
# Bridge detection (integrado en selección)
# ---------------------------------------------------------------------------

_BRIDGE_PATTERNS = [
    ((-1, 0), (0, 1), (-1, 1)),
    ((-1, 1), (0, -1), (-1, 0)),
    ((0, -1), (1, 0), (1, -1)),
    ((0, 1), (1, -1), (1, 0)),
    ((1, 0), (-1, 1), (0, 1)),
    ((1, -1), (-1, 0), (0, -1)),
]


def _bridge_defense_moves(board, size: int, player: int) -> list[tuple[int, int]]:
    """
    Devuelve celdas que defienden puentes propios amenazados por el rival.
    Si el rival ocupa una celda del puente, jugar la otra lo preserva.
    """
    moves: set = set()
    opponent = 3 - player
    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            for (dr1, dc1), (dr2, dc2), (drp, dcp) in _BRIDGE_PATTERNS:
                r1, c1 = r + dr1, c + dc1
                r2, c2 = r + dr2, c + dc2
                rp, cp = r + drp, c + dcp
                if not (
                    0 <= r1 < size
                    and 0 <= c1 < size
                    and 0 <= r2 < size
                    and 0 <= c2 < size
                    and 0 <= rp < size
                    and 0 <= cp < size
                ):
                    continue
                if board[rp][cp] != player:
                    continue
                if board[r1][c1] == opponent and board[r2][c2] == 0:
                    moves.add((r2, c2))
                if board[r2][c2] == opponent and board[r1][c1] == 0:
                    moves.add((r1, c1))
    return list(moves)


def _bridge_build_moves(board, size: int, player: int) -> list[tuple[int, int]]:
    """
    Devuelve celdas que CREAN nuevos puentes conectando grupos separados.
    Construir puentes activamente es más fuerte que esperar a defenderlos.
    """
    moves: set = set()
    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            for (dr1, dc1), (dr2, dc2), (drp, dcp) in _BRIDGE_PATTERNS:
                r1, c1 = r + dr1, c + dc1
                r2, c2 = r + dr2, c + dc2
                rp, cp = r + drp, c + dcp
                if not (
                    0 <= r1 < size
                    and 0 <= c1 < size
                    and 0 <= r2 < size
                    and 0 <= c2 < size
                    and 0 <= rp < size
                    and 0 <= cp < size
                ):
                    continue
                if board[rp][cp] != player:
                    continue
                if board[r1][c1] == 0 and board[r2][c2] == 0:
                    # Puente completo disponible: ambas celdas vacías
                    moves.add((r1, c1))
                    moves.add((r2, c2))
    return list(moves)


# ---------------------------------------------------------------------------
# VC Chain solver
# ---------------------------------------------------------------------------

_VC_PATTERNS = _BRIDGE_PATTERNS


def _find_vc_chain(board, size: int, player: int) -> Optional[list]:
    """Detecta cadena de VCs de borde a borde. Retorna carrier cells o None."""
    parent: dict = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            cell = (r, c)
            for nr, nc in _get_neighbors(r, c, size):
                if board[nr][nc] == player:
                    union(cell, (nr, nc))
            if player == 1:
                if r == 0:
                    union(cell, "SRC")
                if r == size - 1:
                    union(cell, "SNK")
            else:
                if c == 0:
                    union(cell, "SRC")
                if c == size - 1:
                    union(cell, "SNK")

    src_root = find("SRC")
    snk_root = find("SNK")
    if src_root == snk_root:
        return []

    vc_graph: dict = defaultdict(list)
    seen: set = set()

    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            g1 = find((r, c))
            for (dr1, dc1), (dr2, dc2), (drp, dcp) in _VC_PATTERNS:
                r1, c1 = r + dr1, c + dc1
                r2, c2 = r + dr2, c + dc2
                rp, cp = r + drp, c + dcp
                if not (
                    0 <= r1 < size
                    and 0 <= c1 < size
                    and 0 <= r2 < size
                    and 0 <= c2 < size
                    and 0 <= rp < size
                    and 0 <= cp < size
                ):
                    continue
                if board[rp][cp] != player:
                    continue
                if board[r1][c1] != 0 or board[r2][c2] != 0:
                    continue
                g2 = find((rp, cp))
                if g1 == g2:
                    continue
                key = (min(r, rp, r1, r2), min(c, cp, c1, c2), r1, c1, r2, c2)
                if key in seen:
                    continue
                seen.add(key)
                carrier = ((r1, c1), (r2, c2))
                vc_graph[g1].append((g2, carrier))
                vc_graph[g2].append((g1, carrier))

    queue = deque([(src_root, [])])
    visited = {src_root}
    while queue:
        node, path = queue.popleft()
        for neighbor, carrier in vc_graph.get(node, []):
            if neighbor in visited:
                continue
            new_path = path + list(carrier)
            if neighbor == snk_root:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return None


# ---------------------------------------------------------------------------
# Heurísticas de movida
# ---------------------------------------------------------------------------


def _find_winning_move(board, size: int, player: int) -> Optional[tuple]:
    for r in range(size):
        for c in range(size):
            if board[r][c] == 0:
                board[r][c] = player
                won = _check_winner_local(board, size) == player
                board[r][c] = 0
                if won:
                    return (r, c)
    return None


# ---------------------------------------------------------------------------
# Endgame alpha-beta
# ---------------------------------------------------------------------------


def _endgame_solve(board, size: int, player: int, empties: list) -> Optional[tuple]:
    if len(empties) > _ENDGAME_THRESHOLD:
        return None

    center = size // 2
    empties_s = sorted(
        empties, key=lambda rc: abs(rc[0] - center) + abs(rc[1] - center)
    )

    def ab(sim, cur, alpha, beta):
        w = _check_winner_local(sim, size)
        if w == player:
            return 1
        if w == 3 - player:
            return -1
        moves = [(r, c) for r in range(size) for c in range(size) if sim[r][c] == 0]
        if not moves:
            return 0
        moves.sort(key=lambda rc: abs(rc[0] - center) + abs(rc[1] - center))
        best = -2
        for r, c in moves:
            sim[r][c] = cur
            val = -ab(
                sim,
                3 - cur,
                -beta,
                -alpha,
            )
            sim[r][c] = 0
            if val > best:
                best = val
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best

    sim = [list(row) for row in board]
    best_move = None
    best_val = -2
    for r, c in empties_s:
        if sim[r][c] != 0:
            continue
        sim[r][c] = player
        val = -ab(sim, 3 - player, -2, -best_val)
        sim[r][c] = 0
        if val > best_val:
            best_val = val
            best_move = (r, c)
        if best_val == 1:
            break

    return best_move if best_val == 1 else None


# ---------------------------------------------------------------------------
# DARK MODE STRATEGY
# ---------------------------------------------------------------------------


def _dark_play(
    board_list: list[list[int]],
    size: int,
    player: int,
    opponent: int,
    legal: list,
    time_limit: float,
    n_hidden: int,
    opp_prior: np.ndarray,
    prior_matrix: np.ndarray,
    root: Optional["Node"],
    our_last_move: Optional[tuple],
) -> tuple[int, int]:
    """
    Estrategia dark completamente separada.

    Lógica en capas (más rápido → más profundo):
    1. Defensa de puentes amenazados (O(n))
    2. Resistencia sobre tablero PARANOID como selección greedy (O(n²))
    3. IS-MCTS corto sobre tableros muestreados para refinamiento (si sobra tiempo)

    El tablero paranoid es clave: asumimos que el rival ya ocupó sus mejores
    celdas. Si nuestra movida es buena en ese escenario, es robusta a cualquier
    realidad.
    """
    deadline = time.monotonic() + time_limit - 0.5
    legal_set = set(legal)

    # 1. Defensa de puentes amenazados (urgente — O(n))
    bridge_def = [
        m for m in _bridge_defense_moves(board_list, size, player) if m in legal_set
    ]
    if bridge_def:
        return bridge_def[0]

    # 2. Construir tablero paranoid
    paranoid = _build_paranoid_board(board_list, size, player, n_hidden, opp_prior)

    # 3. Resistencia en tablero paranoid — movida greedy robusta
    paranoid_prior = _resistance_prior(paranoid, size, player)

    # Score combinado: prior paranoid + bonus por construir puentes
    bridge_build = set(_bridge_build_moves(board_list, size, player))
    legal_empties = [(r, c) for r, c in legal if paranoid[r][c] == 0]

    if not legal_empties:
        # Todos los legales están bloqueados en el paranoid → usar prior original
        legal_empties = legal

    scores = {}
    for r, c in legal_empties:
        s = float(paranoid_prior[r][c])
        if (r, c) in bridge_build:
            s += 0.15  # bonus por crear puente
        scores[(r, c)] = s

    # Movida greedy sobre el paranoid
    if scores:
        greedy_move = max(scores, key=scores.get)
    else:
        greedy_move = random.choice(legal)

    # 4. IS-MCTS corto para refinar (si queda tiempo suficiente)
    # Usamos determinizaciones ALEATORIAS (no paranoid) para diversidad
    time_remaining = deadline - time.monotonic()
    if time_remaining > 1.0 and root is not None:
        # Correr IS-MCTS sobre tableros muestreados distintos
        move_votes: dict = defaultdict(int)
        move_votes[greedy_move] += 5  # Prior fuerte al paranoid greedy

        iter_count = 0
        while time.monotonic() < deadline - 0.1:
            # Cada iteración usa un tablero determinizado distinto
            det = _sample_determinized_board(
                board_list, size, player, n_hidden, opp_prior
            )
            node, sim_board = _mcts_select(root, [list(r) for r in det])
            winner = _check_winner_local(sim_board, size)
            if winner == 0 and not node.is_fully_expanded():
                node = _mcts_expand(node, sim_board, paranoid_prior, player, opponent)
                winner = _check_winner_local(sim_board, size)
            if winner == 0:
                winner = _simulate_uf(sim_board, size)
            _backprop(node, winner)
            iter_count += 1

        if root.children:
            # Combinar votos del greedy paranoid con visitas del MCTS
            for ch in root.children:
                move_votes[ch.move] += ch.N
            refined = max(move_votes, key=move_votes.get)
            if refined in legal_set:
                return refined

    return greedy_move if greedy_move in legal_set else random.choice(legal)


# ---------------------------------------------------------------------------
# Nodo MCTS
# ---------------------------------------------------------------------------


class Node:
    __slots__ = [
        "move",
        "parent",
        "children",
        "untried",
        "N",
        "W",
        "player_who_moved",
        "prior",
    ]

    def __init__(self, move=None, parent=None, player_who_moved=None, prior=0.0):
        self.move = move
        self.parent = parent
        self.children: list = []
        self.untried: Optional[list] = None
        self.N: int = 0
        self.W: int = 0
        self.player_who_moved = player_who_moved
        self.prior = prior

    def puct_score(self, parent_N: int, C: float, fpu: float) -> float:
        u = C * self.prior * math.sqrt(parent_N)
        if self.N == 0:
            return fpu + u
        return self.W / self.N + u / (1 + self.N)

    def best_child(self, C: float = _C_PUCT) -> "Node":
        parent_q = (self.W / self.N) if self.N > 0 else 0.5
        fpu = parent_q - _FPU_REDUCTION
        pN = self.N
        return max(self.children, key=lambda ch: ch.puct_score(pN, C, fpu))

    def is_fully_expanded(self) -> bool:
        return self.untried is not None and len(self.untried) == 0


def _legal_from_board(board, size: int, player: int, revealed_opponent: set) -> list:
    dead = _dead_cells(board, size)
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] == 0 and (r, c) not in revealed_opponent and (r, c) not in dead
    ]


def _mcts_select(node: Node, board):
    size = len(board)
    while (
        not _check_winner_local(board, size)
        and node.is_fully_expanded()
        and node.children
    ):
        node = node.best_child()
        board[node.move[0]][node.move[1]] = node.player_who_moved
    return node, board


def _mcts_expand(node: Node, board, prior_matrix, player, opponent) -> Node:
    size = len(board)
    if node.untried is None:
        node.untried = _legal_from_board(board, size, player, set())
    if not node.untried:
        return node

    priors = [float(prior_matrix[r][c]) for r, c in node.untried]
    total = sum(priors)
    if total > 0:
        rv = random.random() * total
        cum = 0.0
        chosen_idx = len(node.untried) - 1
        for i, p in enumerate(priors):
            cum += p
            if cum >= rv:
                chosen_idx = i
                break
    else:
        chosen_idx = random.randrange(len(node.untried))

    move = node.untried.pop(chosen_idx)
    next_player = 3 - (node.player_who_moved if node.player_who_moved else opponent)
    board[move[0]][move[1]] = next_player

    child = Node(
        move=move,
        parent=node,
        player_who_moved=next_player,
        prior=float(prior_matrix[move[0]][move[1]]),
    )
    child.untried = _legal_from_board(board, len(board), player, set())
    node.children.append(child)
    return child


def _simulate_uf(board, size: int) -> int:
    """Rollout con Union-Find O(α·n)."""
    sim = [list(row) for row in board]
    ones = twos = 0
    empties = []
    for r in range(size):
        for c in range(size):
            v = sim[r][c]
            if v == 1:
                ones += 1
            elif v == 2:
                twos += 1
            else:
                empties.append((r, c))
    if not empties:
        return _check_winner_local(sim, size)

    player = 1 if ones <= twos else 2
    uf1, uf2, SRC1, SNK1, SRC2, SNK2 = _make_rollout_uf(sim, size)
    random.shuffle(empties)
    last_r = last_c = -1
    move_idx = 0

    while move_idx < len(empties):
        move = None
        if last_r >= 0 and random.random() < 0.30:
            for nr, nc in _get_neighbors(last_r, last_c, size):
                if sim[nr][nc] == 0:
                    move = (nr, nc)
                    break
        if move is None:
            while (
                move_idx < len(empties)
                and sim[empties[move_idx][0]][empties[move_idx][1]] != 0
            ):
                move_idx += 1
            if move_idx >= len(empties):
                break
            move = empties[move_idx]
            move_idx += 1

        r, c = move
        sim[r][c] = player
        last_r, last_c = r, c
        i = r * size + c
        if player == 1:
            if r == 0:
                uf1.union(i, SRC1)
            if r == size - 1:
                uf1.union(i, SNK1)
            for nr, nc in _get_neighbors(r, c, size):
                if sim[nr][nc] == 1:
                    uf1.union(i, nr * size + nc)
            if uf1.connected(SRC1, SNK1):
                return 1
        else:
            if c == 0:
                uf2.union(i, SRC2)
            if c == size - 1:
                uf2.union(i, SNK2)
            for nr, nc in _get_neighbors(r, c, size):
                if sim[nr][nc] == 2:
                    uf2.union(i, nr * size + nc)
            if uf2.connected(SRC2, SNK2):
                return 2
        player = 3 - player

    return _check_winner_local(sim, size)


def _backprop(node: Node, result: int) -> None:
    cur: Optional[Node] = node
    while cur is not None:
        cur.N += 1
        if result == cur.player_who_moved:
            cur.W += 1
        cur = cur.parent


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------


class HexMCTSPUCT_walitos(Strategy):

    @property
    def name(self) -> str:
        return "HexMCTSPUCT_walitos"

    def begin_game(self, config: GameConfig) -> None:
        self.config = config
        self.size = config.board_size
        self.player = config.player
        self.opponent = config.opponent
        self.is_dark = config.variant == "dark"
        self.time_limit = config.time_limit

        self.root: Optional[Node] = None
        self.revealed_opponent: set = set()
        self._our_last_move: Optional[tuple] = None
        self._prior_cache: Optional[np.ndarray] = None
        self._prior_hash: Optional[int] = None
        self._opp_prior_cache: Optional[np.ndarray] = None
        self._opp_prior_hash: Optional[int] = None
        self._our_turn_count: int = 0
        self._n_hidden: int = 0

        self._opening_book: dict = {}
        self._book_depth = 16
        _load_opening_book(self._opening_book, config.variant)

    def on_move_result(self, move: tuple, success: bool) -> None:
        self._our_turn_count += 1
        if not success:
            self.revealed_opponent.add(move)

    def play(self, board, last_move) -> tuple:
        deadline = time.monotonic() + self.time_limit - 0.5
        board_list = [list(row) for row in board]
        legal = _legal_from_board(
            board_list, self.size, self.player, self.revealed_opponent
        )

        if not legal:
            return (0, 0)
        if len(legal) == 1:
            self._our_last_move = legal[0]
            return legal[0]

        # ── Movidas urgentes (ambas variantes) ──────────────────────────
        win = _find_winning_move(board_list, self.size, self.player)
        if win:
            self._our_last_move = win
            return win

        block = _find_winning_move(board_list, self.size, self.opponent)
        if block:
            self._our_last_move = block
            return block

        # ── Priors propios (cacheados) ───────────────────────────────────
        h = hash(tuple(board[r][c] for r in range(self.size) for c in range(self.size)))
        if h != self._prior_hash:
            self._prior_cache = _resistance_prior(board_list, self.size, self.player)
            self._prior_hash = h
        prior_matrix = self._prior_cache

        # ── DISPATCH: dark vs classic ────────────────────────────────────
        if self.is_dark:
            move = self._play_dark(board_list, board, legal, prior_matrix, deadline)
        else:
            move = self._play_classic(
                board_list, board, legal, prior_matrix, deadline, last_move
            )

        # SAFETY: garantizar que nunca retornamos una movida inválida
        if move not in set(legal):
            move = random.choice(legal) if legal else (0, 0)

        self._our_last_move = move
        return move

    # ------------------------------------------------------------------
    # DARK MODE
    # ------------------------------------------------------------------

    def _play_dark(self, board_list, board, legal, prior_matrix, deadline) -> tuple:
        """
        Estrategia dark:
        1. Defensa de puentes
        2. RESTRICCIÓN ESTRUCTURAL: una vez que tenemos piedras, MCTS solo
           considera movidas que extienden la cadena (adyacentes o forman puente).
           Esto suple la falta de "estructura visible del rival" que en classic
           guía a MCTS hacia cadenas focalizadas.
        3. MCTS classic sobre el legal restrictado

        Sin la restricción, MCTS en dark juega disperso: en classic los muros
        del rival visibles bloquean opciones malas, en dark no hay muros visibles
        → MCTS no tiene señal estructural fuerte → puntos dispersos. La
        restricción a celdas conectadas/puente fuerza chain building.
        """
        legal_set = set(legal)

        # 1. Defensa de puentes
        bridge_def = [
            m
            for m in _bridge_defense_moves(board_list, self.size, self.player)
            if m in legal_set
        ]
        if bridge_def:
            return bridge_def[0]

        # 2. Restringir legal a movidas que extienden cadena
        own_stones = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if board_list[r][c] == self.player
        ]

        if own_stones:
            # Celdas adyacentes a alguna propia
            connected_cells = set()
            for r, c in own_stones:
                for nr, nc in _get_neighbors(r, c, self.size):
                    if board_list[nr][nc] == 0:
                        connected_cells.add((nr, nc))

            # Celdas que forman bridge con piedras propias
            bridge_cells = set(_bridge_build_moves(board_list, self.size, self.player))

            restricted = [m for m in legal if m in connected_cells | bridge_cells]
            if restricted:
                legal = restricted
                legal_set = set(legal)

        # 3. MCTS classic sobre legal restrictado
        # Tree reuse imposible en dark (no vemos last_move del rival)
        self.root = Node(player_who_moved=self.opponent)
        self.root.untried = legal[:]

        while time.monotonic() < deadline:
            node, sim_board = _mcts_select(self.root, [list(r) for r in board_list])
            winner = _check_winner_local(sim_board, self.size)
            if winner == 0 and not node.is_fully_expanded():
                node = _mcts_expand(
                    node, sim_board, prior_matrix, self.player, self.opponent
                )
                winner = _check_winner_local(sim_board, self.size)
            if winner == 0:
                winner = _simulate_uf(sim_board, self.size)
            _backprop(node, winner)

        if not self.root.children:
            return random.choice(legal)

        # SAFETY: filtrar a movidas legales
        legal_children = [ch for ch in self.root.children if ch.move in legal_set]
        if not legal_children:
            return random.choice(legal)
        return max(legal_children, key=lambda ch: ch.N).move

    # ------------------------------------------------------------------
    # CLASSIC MODE
    # ------------------------------------------------------------------

    def _play_classic(
        self, board_list, board, legal, prior_matrix, deadline, last_move
    ) -> tuple:
        """
        Estrategia classic: opening book → MCTS+PUCT con prior de resistencia.

        Volvimos a la lógica original que ganaba 7/10 vs Tier 5 en classic.
        VC solver y endgame alpha-beta fueron removidos porque sus interacciones
        con MCTS introdujeron regresiones. La lógica simple es más robusta.
        """
        legal_set = set(legal)

        # Opening book (lookup O(1) en primeras movidas)
        stones = sum(
            board[r][c] != 0 for r in range(self.size) for c in range(self.size)
        )
        if stones < self._book_depth and self._opening_book:
            key = _board_key(board_list, self.size)
            if key in self._opening_book:
                bm = tuple(self._opening_book[key])
                if bm in legal_set:
                    return bm

        # MCTS + PUCT con tree reuse correcto
        reused = self._get_reused_root(last_move)
        if reused is not None:
            self.root = reused
            self.root.parent = None
            if self.root.untried is None:
                played = {ch.move for ch in self.root.children}
                self.root.untried = [m for m in legal if m not in played]
        else:
            self.root = Node(player_who_moved=self.opponent)
            self.root.untried = legal[:]

        while time.monotonic() < deadline:
            node, sim_board = _mcts_select(self.root, [list(r) for r in board_list])
            winner = _check_winner_local(sim_board, self.size)
            if winner == 0 and not node.is_fully_expanded():
                node = _mcts_expand(
                    node, sim_board, prior_matrix, self.player, self.opponent
                )
                winner = _check_winner_local(sim_board, self.size)
            if winner == 0:
                winner = _simulate_uf(sim_board, self.size)
            _backprop(node, winner)

        if not self.root.children:
            return random.choice(legal)

        # SAFETY: filtrar a movidas legales antes de elegir el max
        legal_children = [ch for ch in self.root.children if ch.move in legal_set]
        if not legal_children:
            return random.choice(legal)
        return max(legal_children, key=lambda ch: ch.N).move

    def _get_reused_root(self, opp_last_move):
        """
        Tree reuse correcto: desciende DOS niveles.
        Si solo descendemos uno (nuestra movida), self.root.children
        son movidas del RIVAL, no nuestras. Eso causa que MCTS retorne
        una movida del rival como propia → forfeit por movida ilegal.

        FIX: descender hasta el nodo después de la movida del rival.
        Si no se encuentra, retornar None (crear árbol fresco).
        """
        if self.root is None or not self.root.children or self._our_last_move is None:
            return None
        our_node = next(
            (ch for ch in self.root.children if ch.move == self._our_last_move), None
        )
        if our_node is None:
            return None
        if opp_last_move is None:
            # Sin movida del rival: no podemos reusar (es la primera del rival)
            return None
        opp_node = next(
            (ch for ch in our_node.children if ch.move == opp_last_move), None
        )
        return opp_node  # None si no se encuentra (no hay fallback peligroso)

    def end_game(self, board, winner, your_player) -> None:
        self.root = None
        self._prior_cache = None
        self._opp_prior_cache = None
        self._our_last_move = None


# ---------------------------------------------------------------------------
# Opening book I/O
# ---------------------------------------------------------------------------


def _load_opening_book(book: dict, variant: str) -> None:
    if not os.path.exists(_BOOK_PATH):
        return
    try:
        with open(_BOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        book.update(data.get(variant, {}))
    except Exception:
        pass
