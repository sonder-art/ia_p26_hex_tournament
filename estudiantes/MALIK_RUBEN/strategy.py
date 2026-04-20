"""Hex strategy for team MALIK_RUBEN.

Single-file submission compatible with both classic and dark variants.
"""

from __future__ import annotations

import time

from strategy import GameConfig, Strategy
from hex_game import check_winner, empty_cells, get_neighbors, shortest_path_distance


class MalikRubenStrategy(Strategy):
    @property
    def name(self) -> str:
        return "FogBridge_MALIK_RUBEN"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit
        self._turn_index = 0
        self._known_collisions: set[tuple[int, int]] = set()
        self._last_attempt: tuple[int, int] | None = None
        self._opening = self._build_opening_book()

    def on_move_result(
        self,
        move: tuple[int, int],
        success: bool,
    ) -> None:
        self._last_attempt = move
        if not success:
            self._known_collisions.add(move)

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        self._turn_index += 1
        self._deadline = time.monotonic() + max(0.05, self._time_limit * 0.88)

        if self._variant == "classic":
            return self._play_classic(board, last_move)
        return self._play_dark(board)

    # ---------------------------------------------------------------
    # Variant handlers
    # ---------------------------------------------------------------

    def _play_classic(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        empties = empty_cells(board, self._size)
        if len(empties) == 1:
            return empties[0]

        winning = self._find_immediate_wins(board, self._player, empties)
        if winning:
            return self._pick_best_from(board, winning, classic=True)

        opp_wins = self._find_immediate_wins(board, self._opponent, empties)
        if opp_wins:
            return self._pick_best_from(board, opp_wins, classic=True)

        candidates = self._top_candidates(board, empties, limit=10, classic=True)
        if last_move is not None:
            neighbors = [
                move for move in get_neighbors(last_move[0], last_move[1], self._size)
                if board[move[0]][move[1]] == 0 and move not in candidates
            ]
            candidates.extend(neighbors[:3])

        best_move = candidates[0]
        best_score = -10**18

        for move in candidates:
            if time.monotonic() >= self._deadline:
                break
            my_board = self._with_move(board, move, self._player)
            if check_winner(my_board, self._size) == self._player:
                return move

            score = self._evaluate_board(my_board)
            opp_empties = empty_cells(my_board, self._size)
            opp_candidates = self._top_candidates(
                my_board, opp_empties, limit=5, classic=True, player=self._opponent
            )
            opp_best = -10**18
            for opp_move in opp_candidates:
                if time.monotonic() >= self._deadline:
                    break
                reply_board = self._with_move(my_board, opp_move, self._opponent)
                reply_score = -self._evaluate_board(reply_board)
                if check_winner(reply_board, self._size) == self._opponent:
                    reply_score += 5000.0
                if reply_score > opp_best:
                    opp_best = reply_score

            combined = score - 0.85 * opp_best
            if combined > best_score:
                best_score = combined
                best_move = move

        return best_move

    def _play_dark(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> tuple[int, int]:
        empties = empty_cells(board, self._size)
        if len(empties) == 1:
            return empties[0]

        for move in self._opening:
            if board[move[0]][move[1]] == 0:
                return move

        tactical = self._find_immediate_wins(board, self._player, empties)
        if tactical:
            return self._pick_best_from(board, tactical, classic=False)

        candidates = self._top_candidates(board, empties, limit=14, classic=False)
        best_move = candidates[0]
        best_score = -10**18

        for move in candidates:
            if time.monotonic() >= self._deadline:
                break
            visible_board = self._with_move(board, move, self._player)
            score = self._evaluate_board(visible_board)
            score += self._dark_information_bonus(board, move)
            score -= self._collision_risk(board, move) * 18.0
            score += self._adjacency_bonus(board, move, self._player) * 1.5
            score += self._adjacency_bonus(board, move, self._opponent) * 0.7
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ---------------------------------------------------------------
    # Core evaluation
    # ---------------------------------------------------------------

    def _evaluate_board(self, board: tuple[tuple[int, ...], ...]) -> float:
        my_dist = shortest_path_distance(board, self._size, self._player)
        opp_dist = shortest_path_distance(board, self._size, self._opponent)

        score = 16.0 * (opp_dist - my_dist)
        score += 2.4 * (self._span_score(board, self._player) - self._span_score(board, self._opponent))
        score += 2.0 * (self._bridge_score(board, self._player) - self._bridge_score(board, self._opponent))
        score += 1.1 * (self._central_mass(board, self._player) - self._central_mass(board, self._opponent))
        score += 0.8 * (self._edge_presence(board, self._player) - self._edge_presence(board, self._opponent))
        return score

    def _top_candidates(
        self,
        board: tuple[tuple[int, ...], ...],
        moves: list[tuple[int, int]],
        limit: int,
        classic: bool,
        player: int | None = None,
    ) -> list[tuple[int, int]]:
        who = self._player if player is None else player
        scored: list[tuple[float, tuple[int, int]]] = []
        for move in moves:
            score = self._static_move_score(board, move, who)
            if not classic:
                score += self._dark_information_bonus(board, move)
                score -= self._collision_risk(board, move) * 10.0
            scored.append((score, move))
        scored.sort(reverse=True)
        return [move for _, move in scored[: max(1, min(limit, len(scored)))]]

    def _static_move_score(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> float:
        r, c = move
        center = (self._size - 1) / 2.0
        progress_axis = r if player == 1 else c
        cross_axis = c if player == 1 else r
        progress = min(progress_axis, self._size - 1 - progress_axis)
        cross_center = -abs(cross_axis - center)
        centrality = -(abs(r - center) + abs(c - center))
        own_adj = self._count_adjacent(board, move, player)
        opp_adj = self._count_adjacent(board, move, 3 - player)
        two_hop = self._two_hop_support(board, move, player)

        trial = self._with_move(board, move, player)
        my_dist = shortest_path_distance(trial, self._size, player)
        opp_dist = shortest_path_distance(trial, self._size, 3 - player)

        return (
            11.0 * (opp_dist - my_dist)
            + 2.2 * own_adj
            + 1.4 * opp_adj
            + 0.8 * two_hop
            + 0.55 * centrality
            + 0.45 * cross_center
            - 0.35 * progress
        )

    def _find_immediate_wins(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        wins: list[tuple[int, int]] = []
        for move in moves:
            if time.monotonic() >= self._deadline:
                break
            trial = self._with_move(board, move, player)
            if check_winner(trial, self._size) == player:
                wins.append(move)
        return wins

    def _pick_best_from(
        self,
        board: tuple[tuple[int, ...], ...],
        moves: list[tuple[int, int]],
        classic: bool,
    ) -> tuple[int, int]:
        best = moves[0]
        best_score = -10**18
        for move in moves:
            score = self._static_move_score(board, move, self._player)
            if not classic:
                score -= self._collision_risk(board, move) * 12.0
            if score > best_score:
                best = move
                best_score = score
        return best

    # ---------------------------------------------------------------
    # Dark-mode helpers
    # ---------------------------------------------------------------

    def _collision_risk(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
    ) -> float:
        if move in self._known_collisions:
            return 1.0

        empties = empty_cells(board, self._size)
        if not empties:
            return 0.0

        hidden_estimate = self._estimated_hidden_opponents(board)
        base_density = min(0.9, hidden_estimate / float(len(empties)))

        opp_corridor = self._corridor_value(move, self._opponent)
        my_corridor = self._corridor_value(move, self._player)
        around_opp = self._count_adjacent(board, move, self._opponent)
        around_me = self._count_adjacent(board, move, self._player)
        local = 0.22 * around_opp + 0.08 * max(0, 2 - around_me)

        return max(0.0, min(1.0, base_density * (0.55 + opp_corridor - 0.35 * my_corridor) + local))

    def _estimated_hidden_opponents(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> int:
        visible_opp = sum(cell == self._opponent for row in board for cell in row)
        if self._player == 1:
            opp_turns = max(0, self._turn_index - 1)
        else:
            opp_turns = self._turn_index
        return max(0, opp_turns - visible_opp)

    def _dark_information_bonus(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
    ) -> float:
        unknown_neighbors = 0
        own_neighbors = 0
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            cell = board[nr][nc]
            if cell == 0:
                unknown_neighbors += 1
            elif cell == self._player:
                own_neighbors += 1

        corridor = self._corridor_value(move, self._player)
        return 0.55 * unknown_neighbors + 0.8 * own_neighbors + 1.2 * corridor

    # ---------------------------------------------------------------
    # Shape features
    # ---------------------------------------------------------------

    def _span_score(self, board: tuple[tuple[int, ...], ...], player: int) -> float:
        coords = [
            (r, c)
            for r in range(self._size)
            for c in range(self._size)
            if board[r][c] == player
        ]
        if not coords:
            return 0.0
        if player == 1:
            axis_vals = [r for r, _ in coords]
        else:
            axis_vals = [c for _, c in coords]
        return float(max(axis_vals) - min(axis_vals))

    def _bridge_score(self, board: tuple[tuple[int, ...], ...], player: int) -> float:
        score = 0.0
        patterns = ((1, 0), (0, 1), (1, -1), (1, 1), (2, -1), (1, -2))
        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] != player:
                    continue
                for dr, dc in patterns:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self._size and 0 <= nc < self._size and board[nr][nc] == player:
                        score += 0.45
        return score

    def _central_mass(self, board: tuple[tuple[int, ...], ...], player: int) -> float:
        center = (self._size - 1) / 2.0
        total = 0.0
        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] == player:
                    total += 1.0 / (1.0 + abs(r - center) + abs(c - center))
        return total

    def _edge_presence(self, board: tuple[tuple[int, ...], ...], player: int) -> float:
        total = 0.0
        if player == 1:
            for c in range(self._size):
                total += 1.0 if board[0][c] == player else 0.0
                total += 1.0 if board[self._size - 1][c] == player else 0.0
        else:
            for r in range(self._size):
                total += 1.0 if board[r][0] == player else 0.0
                total += 1.0 if board[r][self._size - 1] == player else 0.0
        return total

    def _count_adjacent(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> int:
        count = 0
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            if board[nr][nc] == player:
                count += 1
        return count

    def _two_hop_support(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> int:
        seen: set[tuple[int, int]] = set()
        count = 0
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            for rr, cc in get_neighbors(nr, nc, self._size):
                if (rr, cc) == move or (rr, cc) in seen:
                    continue
                seen.add((rr, cc))
                if board[rr][cc] == player:
                    count += 1
        return count

    def _adjacency_bonus(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> float:
        bonus = 0.0
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            if board[nr][nc] == player:
                bonus += 1.0
                for rr, cc in get_neighbors(nr, nc, self._size):
                    if board[rr][cc] == player and (rr, cc) != move:
                        bonus += 0.15
        return bonus

    def _corridor_value(self, move: tuple[int, int], player: int) -> float:
        r, c = move
        center = (self._size - 1) / 2.0
        if player == 1:
            axis = abs(c - center)
            side_a = r
            side_b = self._size - 1 - r
        else:
            axis = abs(r - center)
            side_a = c
            side_b = self._size - 1 - c
        edge_balance = min(side_a, side_b)
        return 1.0 / (1.0 + 0.6 * axis + 0.12 * edge_balance)

    def _with_move(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> tuple[tuple[int, ...], ...]:
        rows = [list(row) for row in board]
        rows[move[0]][move[1]] = player
        return tuple(tuple(row) for row in rows)

    def _build_opening_book(self) -> list[tuple[int, int]]:
        center = self._size // 2
        openings = [
            (center, center),
            (center - 1, center + 1),
            (center + 1, center - 1),
            (center, center - 1),
            (center, center + 1),
            (center - 1, center),
            (center + 1, center),
        ]
        result = []
        for r, c in openings:
            if 0 <= r < self._size and 0 <= c < self._size:
                result.append((r, c))
        return result
