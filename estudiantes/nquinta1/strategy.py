import numpy as np
from typing import Optional, Tuple
from strategy import Strategy, GameConfig
from hex_game import empty_cells, shortest_path_distance


class NQuintaStrategy(Strategy):
    @property
    def name(self) -> str:
        return "NQuinta_Strategy_v1"

    def begin_game(self, config: GameConfig) -> None:
        self.size = config.board_size
        self.player = config.player
        self.opponent = config.opponent
        self.failed_moves = set()

    def play(
        self,
        board,
        last_move: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        legal_moves = empty_cells(board, self.size)

        best_score = -float("inf")
        best_move = legal_moves[0]
        center = self.size // 2

        for move in legal_moves:
            if move in self.failed_moves:
                continue

            r, c = move
            new_board = np.array(board)
            new_board[r][c] = self.player

            my_dist = shortest_path_distance(new_board, self.size, self.player)
            opp_dist = shortest_path_distance(new_board, self.size, self.opponent)

            score = opp_dist - my_dist

            center_bonus = -(abs(r - center) + abs(c - center))
            score += 0.5 * center_bonus

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def on_move_result(self, move: Tuple[int, int], success: bool) -> None:
        if not success:
            self.failed_moves.add(move)
