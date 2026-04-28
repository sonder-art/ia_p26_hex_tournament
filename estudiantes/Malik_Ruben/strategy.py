"""Unified Hex strategy for team MALIK_RUBEN.

Classic:
- deterministic engine with alpha-beta, TT, iterative deepening
- v9-style evaluation using shortest path, second route, bridges, position

Dark:
- belief state over hidden opponent stones
- determinization with weighted hidden-stone sampling
- aggregate evaluation across sampled worlds
"""

from __future__ import annotations

import heapq
import importlib.util
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from strategy import GameConfig, Strategy as BaseStrategy
except ImportError:
    framework_path = Path(__file__).resolve().parents[2] / "strategy.py"
    spec = importlib.util.spec_from_file_location("hex_framework_strategy", framework_path)
    if spec is None or spec.loader is None:
        raise
    framework = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = framework
    spec.loader.exec_module(framework)
    GameConfig = framework.GameConfig
    BaseStrategy = framework.Strategy

SIZE = 11
EMPTY = 0
BLACK = 1  # top -> bottom
WHITE = 2  # left -> right

OPPONENT = {BLACK: WHITE, WHITE: BLACK}
HEX_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
INF = float("inf")

AXIS = {
    BLACK: ("row", 0, SIZE - 1),
    WHITE: ("col", 0, SIZE - 1),
}

BRIDGE_PATTERNS = [
    ((-2, 1), [(-1, 0), (-1, 1)]),
    ((-1, 2), [(-1, 1), (0, 1)]),
    ((1, 1), [(0, 1), (1, 0)]),
    ((2, -1), [(1, 0), (1, -1)]),
    ((1, -2), [(0, -1), (1, -1)]),
    ((-1, -1), [(-1, 0), (0, -1)]),
]

random.seed(42)
ZOBRIST = [[[random.getrandbits(64) for _ in range(3)] for _ in range(SIZE)] for _ in range(SIZE)]
ZOBRIST_TURN = {BLACK: random.getrandbits(64), WHITE: random.getrandbits(64)}


def _pos_table() -> list[list[int]]:
    mid = SIZE // 2
    return [[SIZE - max(abs(r - mid), abs(c - mid)) for c in range(SIZE)] for r in range(SIZE)]


POS_VAL = _pos_table()
CENTER = (SIZE // 2, SIZE // 2)
CENTER_DIAGONALS = [(4, 4), (6, 6), (4, 6), (6, 4), (5, 3), (5, 7), (3, 5), (7, 5)]


def _to_grid(board) -> list[list[int]]:
    """Convert any framework board representation to a mutable grid."""
    return [list(map(int, row)) for row in board]


class SearchTimeout(Exception):
    pass


class HexBoard:
    """Mutable board with incremental Zobrist hashing."""

    __slots__ = ("board", "current", "history", "zhash", "_winner_cache", "_winner_checked")

    def __init__(
        self,
        grid: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
        current: int = BLACK,
        last_move: tuple[int, int] | None = None,
    ) -> None:
        if grid is None:
            self.board = [[EMPTY] * SIZE for _ in range(SIZE)]
        else:
            self.board = [list(row) for row in grid]
        self.current = current
        self.history: list[tuple[int, int]] = []
        if last_move is not None and self.board[last_move[0]][last_move[1]] != EMPTY:
            self.history.append(last_move)
        self.zhash = ZOBRIST_TURN[current]
        for r in range(SIZE):
            for c in range(SIZE):
                stone = self.board[r][c]
                if stone != EMPTY:
                    self.zhash ^= ZOBRIST[r][c][EMPTY] ^ ZOBRIST[r][c][stone]
        self._winner_cache: Optional[int] = None
        self._winner_checked = False

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < SIZE and 0 <= c < SIZE

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        return [
            (r + dr, c + dc)
            for dr, dc in HEX_DIRS
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE
        ]

    def place(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c) or self.board[r][c] != EMPTY:
            return False
        player = self.current
        nxt = OPPONENT[player]
        self.zhash ^= ZOBRIST[r][c][EMPTY] ^ ZOBRIST[r][c][player]
        self.zhash ^= ZOBRIST_TURN[player] ^ ZOBRIST_TURN[nxt]
        self.board[r][c] = player
        self.history.append((r, c))
        self.current = nxt
        self._winner_cache = None
        self._winner_checked = False
        return True

    def undo(self) -> None:
        if not self.history:
            return
        r, c = self.history.pop()
        player = self.board[r][c]
        if player == EMPTY:
            return
        nxt = OPPONENT[player]
        self.zhash ^= ZOBRIST[r][c][player] ^ ZOBRIST[r][c][EMPTY]
        self.zhash ^= ZOBRIST_TURN[nxt] ^ ZOBRIST_TURN[player]
        self.board[r][c] = EMPTY
        self.current = player
        self._winner_cache = None
        self._winner_checked = False

    def check_winner(self) -> Optional[int]:
        """Fast winner check from the last move, used inside search."""
        if self._winner_checked:
            return self._winner_cache
        self._winner_checked = True
        if not self.history:
            self._winner_cache = None
            return None
        r, c = self.history[-1]
        player = self.board[r][c]
        if player == EMPTY:
            self._winner_cache = None
            return None
        self._winner_cache = player if self._flood_from(r, c, player) else None
        return self._winner_cache

    def check_winner_full(self) -> Optional[int]:
        """Full BFS winner check, safe for arbitrary external states."""
        for player in (BLACK, WHITE):
            axis, src, tgt = AXIS[player]
            if axis == "row":
                seeds = [(src, c) for c in range(SIZE) if self.board[src][c] == player]
            else:
                seeds = [(r, src) for r in range(SIZE) if self.board[r][src] == player]
            if not seeds:
                continue
            visited = set(seeds)
            stack = list(seeds)
            while stack:
                r, c = stack.pop()
                idx = r if axis == "row" else c
                if idx == tgt:
                    return player
                for nr, nc in self.neighbors(r, c):
                    if (nr, nc) not in visited and self.board[nr][nc] == player:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        return None

    def _flood_from(self, sr: int, sc: int, player: int) -> bool:
        axis, src, tgt = AXIS[player]
        touches_src = False
        touches_tgt = False
        visited = {(sr, sc)}
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            idx = r if axis == "row" else c
            touches_src |= idx == src
            touches_tgt |= idx == tgt
            if touches_src and touches_tgt:
                return True
            for nr, nc in self.neighbors(r, c):
                if (nr, nc) not in visited and self.board[nr][nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return False


def _dijkstra(
    board,
    player: int,
    blocked: Optional[set[tuple[int, int]]] = None,
    reverse: bool = False,
) -> tuple[list[list[float]], list[list[Optional[tuple[int, int]]]]]:
    rival = OPPONENT[player]
    blocked = blocked or set()
    dist = [[INF] * SIZE for _ in range(SIZE)]
    prev: list[list[Optional[tuple[int, int]]]] = [[None] * SIZE for _ in range(SIZE)]
    pq: list[tuple[float, int, int]] = []

    grid = board.board if isinstance(board, HexBoard) else board
    axis, src, tgt = AXIS[player]
    side = tgt if reverse else src
    sources = [(side, c) for c in range(SIZE)] if axis == "row" else [(r, side) for r in range(SIZE)]

    for r, c in sources:
        if (r, c) in blocked:
            continue
        cell = grid[r][c]
        if cell == rival:
            continue
        cost = 0 if cell == player else 1
        if cost < dist[r][c]:
            dist[r][c] = cost
            heapq.heappush(pq, (cost, r, c))

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r][c]:
            continue
        for nr, nc in [
            (r + dr, c + dc)
            for dr, dc in HEX_DIRS
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE
        ]:
            if (nr, nc) in blocked:
                continue
            cell = grid[nr][nc]
            if cell == rival:
                continue
            nd = d + (0 if cell == player else 1)
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                prev[nr][nc] = (r, c)
                heapq.heappush(pq, (nd, nr, nc))
    return dist, prev


def _best_target(dist: list[list[float]], player: int) -> tuple[float, int, int]:
    axis, _, tgt = AXIS[player]
    best_d, best_r, best_c = INF, 0, 0
    if axis == "row":
        for c in range(SIZE):
            if dist[tgt][c] < best_d:
                best_d, best_r, best_c = dist[tgt][c], tgt, c
    else:
        for r in range(SIZE):
            if dist[r][tgt] < best_d:
                best_d, best_r, best_c = dist[r][tgt], r, tgt
    return best_d, best_r, best_c


def _path_cells(prev, target_r: int, target_c: int) -> set[tuple[int, int]]:
    path: set[tuple[int, int]] = set()
    r, c = target_r, target_c
    while prev[r][c] is not None:
        path.add((r, c))
        r, c = prev[r][c]
    path.add((r, c))
    return path


def _cell_cost(board, player: int, r: int, c: int) -> int:
    grid = board.board if isinstance(board, HexBoard) else board
    stone = grid[r][c]
    if stone == player:
        return 0
    if stone == EMPTY:
        return 1
    return SIZE * SIZE


@dataclass
class NodeCache:
    fwd_b: list = field(default_factory=list)
    bwd_b: list = field(default_factory=list)
    prev_b: list = field(default_factory=list)
    fwd_w: list = field(default_factory=list)
    bwd_w: list = field(default_factory=list)
    prev_w: list = field(default_factory=list)
    md_b: float = INF
    md_w: float = INF
    tr_b: tuple[int, int] = (0, 0)
    tr_w: tuple[int, int] = (0, 0)
    d2_fwd_b: list = field(default_factory=list)
    d2_bwd_b: list = field(default_factory=list)
    d2_fwd_w: list = field(default_factory=list)
    d2_bwd_w: list = field(default_factory=list)
    md2_b: float = INF
    md2_w: float = INF


def build_cache(board) -> NodeCache:
    nc = NodeCache()
    nc.fwd_b, nc.prev_b = _dijkstra(board, BLACK, reverse=False)
    nc.bwd_b, _ = _dijkstra(board, BLACK, reverse=True)
    nc.fwd_w, nc.prev_w = _dijkstra(board, WHITE, reverse=False)
    nc.bwd_w, _ = _dijkstra(board, WHITE, reverse=True)
    nc.md_b, tr, tc = _best_target(nc.fwd_b, BLACK)
    nc.tr_b = (tr, tc)
    nc.md_w, tr, tc = _best_target(nc.fwd_w, WHITE)
    nc.tr_w = (tr, tc)

    grid = board.board if isinstance(board, HexBoard) else board

    if nc.md_b < INF:
        path_b = _path_cells(nc.prev_b, *nc.tr_b)
        blocked_b = {p for p in path_b if grid[p[0]][p[1]] == EMPTY}
        nc.d2_fwd_b, _ = _dijkstra(board, BLACK, blocked_b, False)
        nc.d2_bwd_b, _ = _dijkstra(board, BLACK, blocked_b, True)
        nc.md2_b, _, _ = _best_target(nc.d2_fwd_b, BLACK)

    if nc.md_w < INF:
        path_w = _path_cells(nc.prev_w, *nc.tr_w)
        blocked_w = {p for p in path_w if grid[p[0]][p[1]] == EMPTY}
        nc.d2_fwd_w, _ = _dijkstra(board, WHITE, blocked_w, False)
        nc.d2_bwd_w, _ = _dijkstra(board, WHITE, blocked_w, True)
        nc.md2_w, _, _ = _best_target(nc.d2_fwd_w, WHITE)

    return nc


def _independent_routes(player: int, nc: NodeCache) -> tuple[int, float, float]:
    if player == BLACK:
        d1, d2 = nc.md_b, nc.md2_b
    else:
        d1, d2 = nc.md_w, nc.md2_w
    if d1 == INF:
        return 0, INF, INF
    if d2 == INF:
        return 1, d1, INF
    return 2, d1, d2


def _count_bridges(board, player: int) -> int:
    grid = board.board if isinstance(board, HexBoard) else board
    count = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] != player:
                continue
            for (dr, dc), inters in BRIDGE_PATTERNS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                    continue
                if grid[nr][nc] != player:
                    continue
                if all(
                    0 <= r + ir < SIZE and 0 <= c + ic < SIZE and grid[r + ir][c + ic] == EMPTY
                    for ir, ic in inters
                ):
                    count += 1
    return count // 2


def _on_path(fwd, bwd, md: float, board, player: int, r: int, c: int, slack: int = 1) -> bool:
    f = fwd[r][c]
    b = bwd[r][c]
    if f == INF or b == INF:
        return False
    return (f + b - _cell_cost(board, player, r, c)) <= md + slack


def evaluate(board, player: int, nc: Optional[NodeCache] = None) -> float:
    hb = board if isinstance(board, HexBoard) else None
    winner = hb.check_winner() if hb is not None else HexBoard(board, player).check_winner_full()
    if winner == player:
        return 1_000_000.0
    if winner is not None:
        return -1_000_000.0

    if nc is None:
        nc = build_cache(board)
    opp = OPPONENT[player]

    _, d1_self, d2_self = _independent_routes(player, nc)
    _, d1_opp, d2_opp = _independent_routes(opp, nc)

    if d1_self == INF and d1_opp == INF:
        return 0.0
    if d1_self == INF:
        return -500_000.0
    if d1_opp == INF:
        return 500_000.0

    path_score = (d1_opp - d1_self) * 1000.0
    sec_self = d2_self if d2_self < INF else d1_self + SIZE
    sec_opp = d2_opp if d2_opp < INF else d1_opp + SIZE
    sec_score = (sec_opp - sec_self) * 120.0
    bridge_score = (_count_bridges(board, player) - _count_bridges(board, opp)) * 25.0
    grid = board.board if isinstance(board, HexBoard) else board
    pos_score = (
        sum(
            POS_VAL[r][c] * (1 if grid[r][c] == player else -1)
            for r in range(SIZE)
            for c in range(SIZE)
            if grid[r][c] != EMPTY
        )
        * 0.3
    )
    return path_score + sec_score + bridge_score + pos_score


def candidate_moves(board, nc: NodeCache, player: int, max_moves: int = 20) -> list[tuple[int, int]]:
    grid = board.board if isinstance(board, HexBoard) else board
    opp = OPPONENT[player]

    occupied = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] != EMPTY]
    if not occupied:
        mid = SIZE // 2
        opening = [(r, c) for r in range(mid - 2, mid + 3) for c in range(mid - 2, mid + 3)]
        return sorted(opening, key=lambda pos: -POS_VAL[pos[0]][pos[1]])[:max_moves]

    if player == BLACK:
        fwd_p, bwd_p, md_p = nc.fwd_b, nc.bwd_b, nc.md_b
        fwd_o, bwd_o, md_o = nc.fwd_w, nc.bwd_w, nc.md_w
        d2f_o, d2b_o, md2_o = nc.d2_fwd_w, nc.d2_bwd_w, nc.md2_w
    else:
        fwd_p, bwd_p, md_p = nc.fwd_w, nc.bwd_w, nc.md_w
        fwd_o, bwd_o, md_o = nc.fwd_b, nc.bwd_b, nc.md_b
        d2f_o, d2b_o, md2_o = nc.d2_fwd_b, nc.d2_bwd_b, nc.md2_b

    slack_opp = 1
    if md_o < INF and md_p < INF:
        slack_opp = 1 + min(3, round(max(0.0, md_p - md_o)))

    cands: set[tuple[int, int]] = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] != EMPTY:
                continue
            if md_p < INF and _on_path(fwd_p, bwd_p, md_p, grid, player, r, c, 1):
                cands.add((r, c))
            if md_o < INF and _on_path(fwd_o, bwd_o, md_o, grid, opp, r, c, slack_opp):
                cands.add((r, c))
            if md2_o < INF and d2f_o and d2b_o:
                f2 = d2f_o[r][c]
                b2 = d2b_o[r][c]
                if f2 < INF and b2 < INF and (f2 + b2 - _cell_cost(grid, opp, r, c)) <= md2_o + 1:
                    cands.add((r, c))

    if len(cands) < 8:
        cands |= {
            (r + dr, c + dc)
            for r, c in occupied
            for dr, dc in HEX_DIRS
            if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE and grid[r + dr][c + dc] == EMPTY
        }

    if not cands:
        cands = {(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == EMPTY}

    def score(move: tuple[int, int]) -> float:
        r, c = move
        neighbors = [(r + dr, c + dc) for dr, dc in HEX_DIRS if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE]
        own_adj = sum(1 for nr, nc2 in neighbors if grid[nr][nc2] == player)
        opp_adj = sum(1 for nr, nc2 in neighbors if grid[nr][nc2] == opp)
        fp = fwd_p[r][c] if fwd_p[r][c] < INF else SIZE * 2
        bp = bwd_p[r][c] if bwd_p[r][c] < INF else SIZE * 2
        fo = fwd_o[r][c] if fwd_o[r][c] < INF else SIZE * 2
        bo = bwd_o[r][c] if bwd_o[r][c] < INF else SIZE * 2
        f2 = d2f_o[r][c] if d2f_o and d2f_o[r][c] < INF else SIZE * 2
        b2 = d2b_o[r][c] if d2b_o and d2b_o[r][c] < INF else SIZE * 2
        sec_bonus = max(0, SIZE * 2 - (f2 + b2)) * 4
        return -(opp_adj * 40 + own_adj * 20 + POS_VAL[r][c] * 2 - (fp + bp) * 8 - (fo + bo) * 8 + sec_bonus)

    return sorted(cands, key=score)[:max_moves]


EXACT = 0
LOWER = 1
UPPER = 2


class TranspositionTable:
    def __init__(self, max_size: int = 500_000) -> None:
        self._table: dict[tuple[int, int], tuple[int, int, float]] = {}
        self._max_size = max_size

    def get(self, zhash: int, root: int, depth: int, alpha: float, beta: float) -> Optional[float]:
        entry = self._table.get((zhash, root))
        if entry is None:
            return None
        stored_depth, flag, value = entry
        if stored_depth < depth:
            return None
        if flag == EXACT:
            return value
        if flag == LOWER and value >= beta:
            return value
        if flag == UPPER and value <= alpha:
            return value
        return None

    def store(self, zhash: int, root: int, depth: int, flag: int, value: float) -> None:
        if len(self._table) >= self._max_size:
            for key in list(self._table)[: self._max_size // 2]:
                del self._table[key]
        self._table[(zhash, root)] = (depth, flag, value)


def opening_book_move(board, player: int) -> Optional[tuple[int, int]]:
    grid = board.board if isinstance(board, HexBoard) else board
    opp = OPPONENT[player]
    n_own = sum(grid[r][c] == player for r in range(SIZE) for c in range(SIZE))
    n_opp = sum(grid[r][c] == opp for r in range(SIZE) for c in range(SIZE))
    n_total = n_own + n_opp
    if n_total >= 5:
        return None

    axis, src, tgt = AXIS[player]
    cr, cc = CENTER

    if n_total == 0:
        return CENTER
    if n_total == 1 and n_own == 0:
        if grid[cr][cc] == opp:
            for r, c in CENTER_DIAGONALS:
                if grid[r][c] == EMPTY:
                    return (r, c)
        if grid[cr][cc] == EMPTY:
            return CENTER
    if n_total == 2 and n_own == 1:
        opp_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == opp]
        rival_on_axis = any((r if axis == "row" else c) in (src, tgt) for r, c in opp_cells)
        if rival_on_axis:
            for or_, oc in opp_cells:
                for nr, nc in [
                    (or_ + dr, oc + dc)
                    for dr, dc in HEX_DIRS
                    if 0 <= or_ + dr < SIZE and 0 <= oc + dc < SIZE
                ]:
                    idx = nr if axis == "row" else nc
                    if grid[nr][nc] == EMPTY and idx in (src, tgt):
                        return (nr, nc)
        if grid[cr][cc] == EMPTY:
            return CENTER
    if n_total in (3, 4):
        if n_own == 0 and grid[cr][cc] == EMPTY:
            return CENTER
        our_cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if grid[r][c] == player]
        best = None
        best_value = -1
        for or_, oc in our_cells:
            for nr, nc in [
                (or_ + dr, oc + dc)
                for dr, dc in HEX_DIRS
                if 0 <= or_ + dr < SIZE and 0 <= oc + dc < SIZE
            ]:
                if grid[nr][nc] != EMPTY:
                    continue
                idx = nr if axis == "row" else nc
                value = POS_VAL[nr][nc] + max(0, SIZE // 2 - abs(idx - SIZE // 2)) * 2
                if value > best_value:
                    best_value = value
                    best = (nr, nc)
        if best is not None:
            return best
    return None


def _minimax(
    board: HexBoard,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    root_player: int,
    tt: TranspositionTable,
    start: float,
    tlimit: float,
) -> float:
    if time.time() - start > tlimit:
        raise SearchTimeout

    cached = tt.get(board.zhash, root_player, depth, alpha, beta)
    if cached is not None:
        return cached

    winner = board.check_winner()
    if winner == root_player:
        return 1_000_000.0 + depth
    if winner is not None:
        return -1_000_000.0 - depth

    nc = build_cache(board)
    moves = candidate_moves(board, nc, board.current)
    if not moves or depth == 0:
        value = evaluate(board, root_player, nc)
        tt.store(board.zhash, root_player, depth, EXACT, value)
        return value

    orig_alpha = alpha
    orig_beta = beta
    best_val = -INF if maximizing else INF

    for r, c in moves:
        board.place(r, c)
        try:
            child = _minimax(board, depth - 1, alpha, beta, not maximizing, root_player, tt, start, tlimit)
        finally:
            board.undo()
        if maximizing:
            best_val = max(best_val, child)
            alpha = max(alpha, best_val)
        else:
            best_val = min(best_val, child)
            beta = min(beta, best_val)
        if alpha >= beta:
            break

    flag = UPPER if best_val <= orig_alpha else LOWER if best_val >= orig_beta else EXACT
    tt.store(board.zhash, root_player, depth, flag, best_val)
    return best_val


def _classic_best_move(
    board: HexBoard,
    player: int,
    time_limit: float = 13.0,
    max_depth: int = 6,
    max_cands: int = 20,
    tt: Optional[TranspositionTable] = None,
) -> Optional[tuple[int, int]]:
    book = opening_book_move(board, player)
    if book is not None:
        return book

    if tt is None:
        tt = TranspositionTable()
    nc0 = build_cache(board)
    moves = candidate_moves(board, nc0, player, max_cands)
    if not moves:
        return None

    start = time.time()
    best = moves[0]
    best_val = -INF

    for depth in range(1, max_depth + 1):
        if time.time() - start > time_limit:
            break
        depth_best = None
        depth_best_val = -INF
        timed_out = False

        for r, c in moves:
            if time.time() - start > time_limit:
                timed_out = True
                break
            board.place(r, c)
            try:
                value = _minimax(board, depth - 1, -INF, INF, False, player, tt, start, time_limit)
            except SearchTimeout:
                timed_out = True
                break
            finally:
                board.undo()
            if value > depth_best_val:
                depth_best_val = value
                depth_best = (r, c)

        if not timed_out and time.time() - start > time_limit:
            timed_out = True
        if not timed_out and depth_best is not None:
            best = depth_best
            best_val = depth_best_val
            moves = [best] + [m for m in moves if m != best]
        if abs(best_val) >= 999_000:
            break
    return best


class BeliefState:
    """Belief state for dark mode."""

    def __init__(self, player: int, first_mover: bool) -> None:
        self.player = player
        self.opp = OPPONENT[player]
        self.first_mover = first_mover
        self.confirmed_rival: set[tuple[int, int]] = set()
        self.n_collisions = 0
        self.n_succeeded = 0
        self.last_rival_heat = [[1.0] * SIZE for _ in range(SIZE)]

    def update(self, view: list[list[int]]) -> None:
        for r in range(SIZE):
            for c in range(SIZE):
                if view[r][c] == self.opp:
                    self.confirmed_rival.add((r, c))

    def on_result(self, move: tuple[int, int], success: bool) -> None:
        if success:
            self.n_succeeded += 1
        else:
            self.n_collisions += 1
            self.confirmed_rival.add(move)

    def estimate_n_hidden(self) -> int:
        """Estimate hidden opponent stones from move parity and collisions."""
        our_total_attempts = self.n_succeeded + self.n_collisions
        rival_turns = our_total_attempts if self.first_mover else our_total_attempts + 1
        return max(0, rival_turns - len(self.confirmed_rival))

    def update_rival_heat(self, view: list[list[int]]) -> None:
        dist_fwd, _ = _dijkstra(view, self.opp, reverse=False)
        dist_bwd, _ = _dijkstra(view, self.opp, reverse=True)
        md, _, _ = _best_target(dist_fwd, self.opp)

        heat = [[0.0] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            for c in range(SIZE):
                if view[r][c] != EMPTY:
                    continue
                f = dist_fwd[r][c]
                b = dist_bwd[r][c]
                if f == INF or b == INF:
                    continue
                relevance = max(0.0, (md + 2) - (f + b - 1))
                heat[r][c] = relevance + 0.1
        self.last_rival_heat = heat

    def sample_determinization(self, view: list[list[int]], n_hidden: int, rng: random.Random) -> list[list[int]]:
        det = [list(row) for row in view]
        uncertain = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if det[r][c] == EMPTY and (r, c) not in self.confirmed_rival
        ]
        if n_hidden == 0 or not uncertain:
            return det

        weights = [self.last_rival_heat[r][c] for r, c in uncertain]
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else [1.0 / len(uncertain)] * len(uncertain)
        sample_count = min(n_hidden, len(uncertain))
        chosen = _weighted_sample(uncertain, probs, sample_count, rng)
        for r, c in chosen:
            det[r][c] = self.opp
        return det


def _weighted_sample(
    items: list[tuple[int, int]],
    probs: list[float],
    k: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    pool = list(items)
    weights = list(probs)
    chosen: list[tuple[int, int]] = []
    for _ in range(k):
        if not pool:
            break
        total = sum(weights)
        if total <= 0:
            break
        target = rng.random() * total
        acc = 0.0
        idx = 0
        for i, weight in enumerate(weights):
            acc += weight
            if acc >= target:
                idx = i
                break
        chosen.append(pool.pop(idx))
        weights.pop(idx)
    return chosen


def _dark_best_move(
    view: list[list[int]],
    player: int,
    belief: BeliefState,
    time_limit: float = 13.0,
    n_det: int = 30,
    max_cands: int = 20,
) -> Optional[tuple[int, int]]:
    rng = random.Random()
    n_own = sum(view[r][c] == player for r in range(SIZE) for c in range(SIZE))

    if n_own == 0:
        if player == BLACK:
            if view[CENTER[0]][CENTER[1]] == EMPTY:
                return CENTER
        else:
            for r, c in CENTER_DIAGONALS:
                if view[r][c] == EMPTY:
                    return (r, c)

    belief.update_rival_heat(view)
    n_hidden = belief.estimate_n_hidden()

    nc_view = build_cache(view)
    cands = [move for move in candidate_moves(view, nc_view, player, max_cands) if view[move[0]][move[1]] == EMPTY]
    if not cands:
        cands = [(r, c) for r in range(SIZE) for c in range(SIZE) if view[r][c] == EMPTY][:max_cands]
    if not cands:
        return None

    scores = {move: 0.0 for move in cands}
    counts = {move: 0 for move in cands}
    start = time.time()

    for _ in range(n_det):
        if time.time() - start > time_limit * 0.92:
            break
        det = belief.sample_determinization(view, n_hidden, rng)

        for r, c in cands:
            if time.time() - start > time_limit * 0.95:
                break
            move = (r, c)
            if det[r][c] != EMPTY:
                scores[move] += -3000.0
                counts[move] += 1
                continue
            det[r][c] = player
            nc_after = build_cache(det)
            scores[move] += evaluate(det, player, nc_after)
            counts[move] += 1
            det[r][c] = EMPTY

    averaged = {move: (scores[move] / counts[move] if counts[move] > 0 else -INF) for move in cands}
    if all(value == -INF for value in averaged.values()):
        return cands[0]
    return max(averaged, key=averaged.get)


class MalikRubenStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "FogBridge_MALIK_RUBEN"

    def begin_game(self, config: GameConfig) -> None:
        self._player = config.player
        self._variant = config.variant
        self._time_limit = config.time_limit * 0.88
        self._belief: Optional[BeliefState] = None
        self._tt = TranspositionTable()
        if self._variant == "dark":
            self._belief = BeliefState(self._player, first_mover=self._player == BLACK)

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        grid = _to_grid(board)
        if self._variant == "classic":
            hb = HexBoard(grid, self._player, last_move)
            move = _classic_best_move(
                hb,
                self._player,
                self._time_limit,
                max_depth=6,
                max_cands=20,
                tt=self._tt,
            )
        else:
            assert self._belief is not None
            self._belief.update(grid)
            move = _dark_best_move(grid, self._player, self._belief, self._time_limit, n_det=30, max_cands=20)

        if move is not None:
            return move
        for r in range(SIZE):
            for c in range(SIZE):
                if grid[r][c] == EMPTY:
                    return (r, c)
        return (0, 0)

    def on_move_result(self, move: tuple[int, int], success: bool) -> None:
        if self._variant == "dark" and self._belief is not None:
            self._belief.on_result(move, success)

    def end_game(
        self,
        board: tuple[tuple[int, ...], ...],
        winner: int,
        your_player: int,
    ) -> None:
        return None


def _run_smoke_tests() -> None:
    hb = HexBoard()
    for r in range(SIZE):
        hb.place(r, 0)
        if r < SIZE - 1:
            hb.place(r, SIZE - 1)
    assert hb.check_winner_full() == BLACK

    bs = BeliefState(BLACK, True)
    bs.on_result((5, 5), True)
    assert bs.estimate_n_hidden() == 1

    blue_bs = BeliefState(WHITE, False)
    assert blue_bs.estimate_n_hidden() == 1


if __name__ == "__main__":
    _run_smoke_tests()
