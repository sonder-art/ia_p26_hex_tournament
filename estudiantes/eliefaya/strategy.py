"""Strategy template — copy this directory to estudiantes/<your_team>/

Rename the class and the ``name`` property, then implement your logic
in the ``play`` method.
"""

from __future__ import annotations

import random

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
        return "ElieStrategy_eliefaya"  # <-- CHANGE THIS

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
        self._failed_moves = set()

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
        if not success:
            self._failed_moves.add(move)  # Track collisions here for dark mode

    def play(self, board, last_move):
        moves = [m for m in empty_cells(board, self._size) if m not in self._failed_moves]

        best_move = None
        best_score = float("inf")
        # 1. verificar si puedo ganar inmediatamente
        for move in moves:
            r, c = move
            new_board = [list(row) for row in board]
            new_board[r][c] = self._player
            
            if check_winner(new_board, self._size) == self._player:
                return move
        # 2. bloquear victoria inmediata del oponente
        for move in moves:
            r, c = move
            new_board = [list(row) for row in board]
            new_board[r][c] = self._opponent
            
            if check_winner(new_board, self._size) == self._opponent:
                return move
            
        for move in moves:
            r, c = move
            
            # copiar tablero
            new_board = [list(row) for row in board]
            new_board[r][c] = self._player
            
            # qué tan cerca estoy de ganar
            my_dist = shortest_path_distance(new_board, self._size, self._player)
            
            # qué tan cerca está el oponente de ganar
            opp_dist = shortest_path_distance(new_board, self._size, self._opponent)
            
            # estrategia: acercarme y alejar al rival
            center = self._size // 2

            score = my_dist - opp_dist

# penalizar distancia al centro
            score += abs(r - center) + abs(c - center)

            if score < best_score:
                best_score = score
                best_move = move

        return best_move
            
            