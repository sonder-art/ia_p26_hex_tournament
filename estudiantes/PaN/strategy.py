from __future__ import annotations

from strategy import Strategy, GameConfig
from hex_game import empty_cells, shortest_path_distance


class EstrategiaPAN(Strategy):
    """Elige la jugada que más reduzca la distancia de conexión propia."""

    @property
    def name(self) -> str:
        return "EstrategiaPAN_PAN"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant

    def on_move_result(
        self,
        move: tuple[int, int],
        success: bool,
    ) -> None:
        pass

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        moves = empty_cells(board, self._size)

        center = (self._size // 2, self._size // 2)
        if center in moves:
            return center

        best_move = moves[0]
        best_score = float("inf")

        for move in moves:
            r, c = move

            board_list = [list(row) for row in board]
            board_list[r][c] = self._player
            simulated_board = tuple(tuple(row) for row in board_list)

            score = shortest_path_distance(simulated_board, self._size, self._player)

            if score < best_score:
                best_score = score
                best_move = move

        return best_move
