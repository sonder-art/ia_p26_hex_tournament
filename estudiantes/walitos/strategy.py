"""
Estrategia MCTS + PUCT con prior de red de resistencia eléctrica.
Solo depende de numpy. Sin scipy ni otras librerías externas.

Técnicas:
  - Prior via red de resistencia (Kirchhoff) con numpy.linalg.lstsq
  - Rollout con Union-Find: O(α·n) por paso vs O(n²) de BFS
  - Tree reuse: reutiliza subárbol de la jugada anterior
  - FPU (First Play Urgency): selección estable para nodos no visitados
  - Tabla de vecinos precomputada al importar (cero costo en runtime)
  - Dead cell pruning corregido
  - Opening book: lookup O(1) para primeras movidas
  - Dark mode con mapa de colisiones
"""

from __future__ import annotations

import json
import math
import os
import random
import time
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


def _make_rollout_uf(
    board: list[list[int]], size: int
) -> tuple[_UF, _UF, int, int, int, int]:
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


def _check_winner_local(board: list[list[int]], size: int) -> int:
    for player in (1, 2):
        visited: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = []
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


def _board_key(board: list[list[int]], size: int) -> str:
    return ";".join(
        sorted(
            f"{board[r][c]}:{r},{c}"
            for r in range(size)
            for c in range(size)
            if board[r][c] != 0
        )
    )


# ---------------------------------------------------------------------------
# Dead cell pruning (corregido)
# ---------------------------------------------------------------------------


def _dead_cells(board: list[list[int]], size: int) -> set[tuple[int, int]]:
    """
    Celda muerta: todos sus vecinos están ocupados y son del mismo color.
    Completamente encerrada — no puede contribuir a ningún camino ganador.
    """
    dead: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                continue
            nbs = _get_neighbors(r, c, size)
            if not nbs:
                continue
            if any(board[nr][nc] == 0 for nr, nc in nbs):
                continue
            colors = {board[nr][nc] for nr, nc in nbs}
            if len(colors) == 1:
                dead.add((r, c))
    return dead


# ---------------------------------------------------------------------------
# Red de resistencia eléctrica
# ---------------------------------------------------------------------------


def _resistance_prior(board: list[list[int]], size: int, player: int) -> np.ndarray:
    N = size * size
    SRC = N
    SNK = N + 1
    opponent = 3 - player
    A = np.zeros((N + 2, N + 2), dtype=np.float64)
    b = np.zeros(N + 2, dtype=np.float64)

    def idx(r: int, c: int) -> int:
        return r * size + c

    def cond(r: int, c: int) -> float:
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
        total = prior.sum()
        return prior / total if total > 0 else prior

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
    total = prior.sum()
    if total > 0:
        prior /= total
    return prior


# ---------------------------------------------------------------------------
# Heurísticas de movida
# ---------------------------------------------------------------------------


def _find_winning_move(
    board: list[list[int]], size: int, player: int
) -> Optional[tuple[int, int]]:
    for r in range(size):
        for c in range(size):
            if board[r][c] == 0:
                board[r][c] = player
                won = _check_winner_local(board, size) == player
                board[r][c] = 0
                if won:
                    return (r, c)
    return None


def _detect_bridges(
    board: list[list[int]], size: int, player: int
) -> list[tuple[int, int]]:
    bridge_moves: set[tuple[int, int]] = set()
    opponent = 3 - player
    patterns = [
        ((-1, 0), (0, 1), (-1, 1)),
        ((-1, 1), (0, -1), (-1, 0)),
        ((0, -1), (1, 0), (1, -1)),
        ((0, 1), (1, -1), (1, 0)),
        ((1, 0), (-1, 1), (0, 1)),
        ((1, -1), (-1, 0), (0, -1)),
    ]
    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            for (dr1, dc1), (dr2, dc2), (drp, dcp) in patterns:
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
                    bridge_moves.add((r2, c2))
                if board[r2][c2] == opponent and board[r1][c1] == 0:
                    bridge_moves.add((r1, c1))
    return list(bridge_moves)


# ---------------------------------------------------------------------------
# Nodo MCTS con FPU
# ---------------------------------------------------------------------------

_C_PUCT = 1.2
_FPU_REDUCTION = 0.20


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

    def __init__(
        self,
        move: Optional[tuple[int, int]] = None,
        parent: Optional["Node"] = None,
        player_who_moved: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.move = move
        self.parent = parent
        self.children: list["Node"] = []
        self.untried: Optional[list[tuple[int, int]]] = None
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


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------


class HexMCTSPUCT(Strategy):

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
        self.revealed_opponent: set[tuple[int, int]] = set()
        self._our_last_move: Optional[tuple[int, int]] = None
        self._prior_cache: Optional[np.ndarray] = None
        self._prior_hash: Optional[int] = None
        self._opening_book: dict[str, list[int]] = {}
        self._book_depth = 16
        _load_opening_book(self._opening_book, config.variant)

    def on_move_result(self, move: tuple[int, int], success: bool) -> None:
        if not success:
            self.revealed_opponent.add(move)

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        deadline = time.monotonic() + self.time_limit - 0.5
        board_list = [list(row) for row in board]
        legal = self._legal_moves(board_list)

        if not legal:
            return (0, 0)
        if len(legal) == 1:
            self._our_last_move = legal[0]
            return legal[0]

        win = _find_winning_move(board_list, self.size, self.player)
        if win:
            self._our_last_move = win
            return win

        block = _find_winning_move(board_list, self.size, self.opponent)
        if block:
            self._our_last_move = block
            return block

        stones = sum(
            board[r][c] != 0 for r in range(self.size) for c in range(self.size)
        )
        if stones < self._book_depth and self._opening_book:
            key = _board_key(board_list, self.size)
            if key in self._opening_book:
                bm = tuple(self._opening_book[key])
                if bm in set(legal):
                    self._our_last_move = bm
                    return bm

        prior_matrix = self._get_prior(board_list)

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
            node, sim_board = self._select(self.root, [list(r) for r in board_list])
            winner = _check_winner_local(sim_board, self.size)
            if winner == 0 and not node.is_fully_expanded():
                node = self._expand(node, sim_board, prior_matrix)
                winner = _check_winner_local(sim_board, self.size)
            if winner == 0:
                winner = self._simulate(sim_board)
            self._backpropagate(node, winner)

        if not self.root.children:
            move = random.choice(legal)
            self._our_last_move = move
            return move

        best = max(self.root.children, key=lambda ch: ch.N)
        self._our_last_move = best.move
        return best.move

    def end_game(self, board, winner, your_player) -> None:
        self.root = None
        self._prior_cache = None
        self._our_last_move = None

    def _get_reused_root(self, opponent_last_move):
        if self.root is None or not self.root.children or self._our_last_move is None:
            return None
        our_node = next(
            (ch for ch in self.root.children if ch.move == self._our_last_move), None
        )
        if our_node is None:
            return None
        if opponent_last_move is None:
            return our_node
        opp_node = next(
            (ch for ch in our_node.children if ch.move == opponent_last_move), None
        )
        return opp_node if opp_node is not None else our_node

    def _legal_moves(self, board: list[list[int]]) -> list[tuple[int, int]]:
        dead = _dead_cells(board, self.size)
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if board[r][c] == 0
            and (r, c) not in self.revealed_opponent
            and (r, c) not in dead
        ]

    def _get_prior(self, board: list[list[int]]) -> np.ndarray:
        h = hash(tuple(board[r][c] for r in range(self.size) for c in range(self.size)))
        if h != self._prior_hash:
            self._prior_cache = _resistance_prior(board, self.size, self.player)
            self._prior_hash = h
        return self._prior_cache

    def _select(self, node: Node, board: list[list[int]]):
        while (
            not _check_winner_local(board, self.size)
            and node.is_fully_expanded()
            and node.children
        ):
            node = node.best_child()
            board[node.move[0]][node.move[1]] = node.player_who_moved
        return node, board

    def _expand(
        self, node: Node, board: list[list[int]], prior_matrix: np.ndarray
    ) -> Node:
        if node.untried is None:
            node.untried = self._legal_moves(board)
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
        next_player = 3 - (
            node.player_who_moved if node.player_who_moved else self.opponent
        )
        board[move[0]][move[1]] = next_player

        child = Node(
            move=move,
            parent=node,
            player_who_moved=next_player,
            prior=float(prior_matrix[move[0]][move[1]]),
        )
        child.untried = self._legal_moves(board)
        node.children.append(child)
        return child

    def _simulate(self, board: list[list[int]]) -> int:
        sim = [list(row) for row in board]
        size = self.size
        ones = twos = 0
        empties: list[tuple[int, int]] = []
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
            move: Optional[tuple[int, int]] = None
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

    def _backpropagate(self, node: Node, result: int) -> None:
        current: Optional[Node] = node
        while current is not None:
            current.N += 1
            if result == current.player_who_moved:
                current.W += 1
            current = current.parent


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
