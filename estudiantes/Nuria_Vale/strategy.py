"""Strategy template — copy this directory to estudiantes/<your_team>/

Rename the class and the ``name`` property, then implement your logic
in the ``play`` method.
"""

from __future__ import annotations

import random
import multiprocessing as mp

from strategy import Strategy, GameConfig
from hex_game import (
    get_neighbors,
    check_winner,
    shortest_path_distance,
    empty_cells,
)


class MyStrategy(Strategy):
    """Example strategy — replace with your own logic."""

    @property
    def name(self) -> str:
        # Convention: "StrategyName_teamname"
        return "Nuria_Vale"  # <-- CHANGE THIS

    def begin_game(self, config: GameConfig) -> None:
        """Called once at the start of each game.

        Available information in config:
          - config.board_size      (int)   — side length (default 11)
          - config.variant         (str)   — "classic" or "dark"
          - config.initial_board   (tuple) — starting board state (in dark: only your stones)
          - config.player          (int)   — your player number (1=Black, 2=White)
          - config.opponent        (int)   — opponent's number
          - config.time_limit      (float) — max seconds per move

        Player 1 (Black): connects top (row 0) to bottom (row N-1).
        Player 2 (White): connects left (col 0) to right (col N-1).

        Board cell values: 0=empty, 1=Black, 2=White.

        Dark mode (fog of war):
          - You only see your own stones + opponent stones discovered by collision.
          - last_move is always None (you don't know where the opponent played).
          - on_move_result(move, success) tells you if your move collided.
        """
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        
        self._start_time = None

    def on_move_result(
        self,
        move: tuple[int, int],
        success: bool,
    ) -> None:
        """Called after each play() with the result.

        Parameters
        ----------
        move : tuple[int, int]
            The move you just played.
        success : bool
            True if your stone was placed. False if collision (dark mode only):
            the cell had a hidden opponent stone — you lose your turn but now
            see that stone.
        """
        pass  # Track collisions here for dark mode

    def play(self, board, last_move):
        import time, random, math
        start = time.time()

        size = self._size
        player = self._player
        opponent = self._opponent

        moves = empty_cells(board, size)

    # ---------------------------
    # 1. GANAR / BLOQUEAR
    # ---------------------------
        for m in moves:
            b = [list(row) for row in board]
            b[m[0]][m[1]] = player
            if check_winner(b, size) == player:
                return m

        for m in moves:
            b = [list(row) for row in board]
            b[m[0]][m[1]] = opponent
            if check_winner(b, size) == opponent:
                return m

    # ---------------------------
    # 2. FILTRAR ZONA ACTIVA
    # ---------------------------
        active = set()
        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    for nr, nc in get_neighbors(r, c, size):
                        if board[nr][nc] == 0:
                            active.add((nr, nc))

        if not active:
            active = set(moves)

        moves = list(active)

    # ---------------------------
    # 3. EVALUACIÓN
    # ---------------------------
        def soft_eval(b):
            my_dist = shortest_path_distance(b, size, player)
            opp_dist = shortest_path_distance(b, size, opponent)
            return 1 / (1 + math.exp(-(opp_dist - my_dist)))

    # ---------------------------
    # 4. ROLLOUT
    # ---------------------------
        def rollout(b, current):
            b = [row[:] for row in b]
            for _ in range(20):
                empties = empty_cells(b, size)
                if not empties:
                    break

                move = random.choice(empties)
                b[move[0]][move[1]] = current
                current = 3 - current

            return soft_eval(b)

    # ---------------------------
    # 5. FUNCIÓN WORKER
    # ---------------------------
        def evaluate_move(move):
            r, c = move
            base = [list(row) for row in board]
            base[r][c] = player

            total = 0
            sims = 10

            for _ in range(sims):
                total += rollout(base, opponent)

            return (move, total / sims)

    # ---------------------------
    # 6. PARALELISMO
    # ---------------------------
        MAX_CANDIDATES = min(12, len(moves))
        candidate_moves = moves[:MAX_CANDIDATES]

        try:
            ctx = mp.get_context("fork")  # importante en Mac
            with ctx.Pool(4) as pool:
                results = pool.map(evaluate_move, candidate_moves)
        except:
            # fallback sin paralelismo
            results = [evaluate_move(m) for m in candidate_moves]

    # ---------------------------
    # 7. ELEGIR MEJOR
    # ---------------------------
        best_move = max(results, key=lambda x: x[1])[0]

        return best_move