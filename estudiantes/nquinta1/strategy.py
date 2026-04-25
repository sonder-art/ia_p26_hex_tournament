import numpy as np
from typing import Optional, Tuple
from strategy import Strategy, GameConfig
from hex_game import empty_cells, shortest_path_distance


class NQuintaStrategy(Strategy):
    @property
    def name(self) -> str:
        return "NQuinta_Strategy_v2"

    def begin_game(self, config: GameConfig) -> None:
        self.size = config.board_size
        self.player = config.player
        self.opponent = config.opponent
        self.failed_moves = set()

    def _neighbors(self, r: int, c: int):
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def _count_adjacent(self, board, r: int, c: int, value) -> int:
        count = 0
        for nr, nc in self._neighbors(r, c):
            if board[nr][nc] == value:
                count += 1
        return count

    def play(
        self,
        board,
        last_move: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        legal_moves = empty_cells(board, self.size)

        if not legal_moves:
            return (0, 0)

        center = self.size // 2

        # Apertura: si el centro está libre, tómalo.
        if (center, center) in legal_moves:
            return (center, center)

        current_my_dist = shortest_path_distance(board, self.size, self.player)
        current_opp_dist = shortest_path_distance(board, self.size, self.opponent)

        best_score = -float("inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            if move in self.failed_moves:
                continue

            r, c = move

            # Simular mi jugada
            my_board = np.array(board, copy=True)
            my_board[r][c] = self.player

            my_dist_after = shortest_path_distance(my_board, self.size, self.player)
            opp_dist_after = shortest_path_distance(my_board, self.size, self.opponent)

            # Si gano inmediatamente, tomo esa jugada.
            if my_dist_after == 0:
                return move

            # Simular qué tan peligrosa sería esta casilla si la tomara el rival
            opp_board = np.array(board, copy=True)
            opp_board[r][c] = self.opponent
            opp_if_plays_here = shortest_path_distance(opp_board, self.size, self.opponent)

            score = 0.0

            # Mejora propia: bajar mi distancia es muy bueno
            score += 12.0 * (current_my_dist - my_dist_after)

            # Empeorar al rival también ayuda
            score += 10.0 * (opp_dist_after - current_opp_dist)

            # Bloqueo preventivo: si esta casilla le ayudaría mucho al rival, me interesa ocuparla
            score += 11.0 * (current_opp_dist - opp_if_plays_here)

            # Conexión local: prefiero pegarme a mis propias piedras
            my_adj = self._count_adjacent(board, r, c, self.player)
            opp_adj = self._count_adjacent(board, r, c, self.opponent)
            score += 2.5 * my_adj
            score += 1.5 * opp_adj

            # Bonus por cercanía al centro
            center_bonus = -(abs(r - center) + abs(c - center))
            score += 0.4 * center_bonus

            # Ligero bonus a jugadas que no quedan aisladas
            if my_adj > 0:
                score += 1.5

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def on_move_result(self, move: Tuple[int, int], success: bool) -> None:
        if not success:
            self.failed_moves.add(move)
