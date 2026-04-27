from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

from strategy import GameConfig, Strategy
from hex_game import check_winner, empty_cells, get_neighbors, shortest_path_distance


DEFAULT_TUNING: dict[str, float | int] = {
    "COMPUS_C_PUCT": 1.65,
    "COMPUS_ROLLOUT_DEPTH": 12,
    "COMPUS_ROLLOUT_EPSILON": 0.08,
    "COMPUS_RISK_LOW": 0.45,
    "COMPUS_RISK_MID": 0.65,
    "COMPUS_RISK_HIGH": 0.8,
    "COMPUS_BUDGET_OPEN": 0.46,
    "COMPUS_BUDGET_MID": 0.46,
    "COMPUS_BUDGET_LATE": 0.58,
    "COMPUS_BUDGET_END": 0.72,
    "COMPUS_ROOT_LIMIT_SCALE": 1.25,
    "COMPUS_TREE_LIMIT_SCALE": 1.45,
    "COMPUS_ROLLOUT_DEPTH_BONUS": 0,
}


@dataclass
class _Node:
    board: list[list[int]]
    player_to_move: int
    parent: "_Node | None"
    move: tuple[int, int] | None
    prior: float
    unexpanded: list[tuple[tuple[int, int], float]]
    visits: int = 0
    value_sum: float = 0.0
    children: dict[tuple[int, int], "_Node"] | None = None

    @property
    def q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class CompusHybridMCTS(Strategy):
    @property
    def name(self) -> str:
        return "CompusHybridMCTS_compus" # "CompusHybridMCTS_compus"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit

        fixed_seed = self._read_optional_int("COMPUS_FIXED_SEED")
        if fixed_seed is not None:
            seed = fixed_seed ^ (self._player * 1009) ^ (self._size * 37)
        else:
            seed = time.time_ns() ^ (self._player * 1009) ^ (self._size * 37)
        self._rng = random.Random(seed)
        self._move_index = 0
        self._last_own_move: tuple[int, int] | None = None

        self._known_opp: set[tuple[int, int]] = set()
        self._collision_cells: set[tuple[int, int]] = set()
        self._risk_map = [[0.10 for _ in range(self._size)] for _ in range(self._size)]
        self._attempt_heat = [[0 for _ in range(self._size)] for _ in range(self._size)]

        self._cells = [(r, c) for r in range(self._size) for c in range(self._size)]
        self._neighbors = [
            [tuple(get_neighbors(r, c, self._size)) for c in range(self._size)]
            for r in range(self._size)
        ]
        self._center_score_map = self._build_center_score_map()
        self._axis_score_map = {
            1: self._build_axis_score_map(actor=1),
            2: self._build_axis_score_map(actor=2),
        }
        self._bridge_patterns = self._build_bridge_patterns()

        self._c_puct = self._read_float(
            "COMPUS_C_PUCT",
            float(DEFAULT_TUNING["COMPUS_C_PUCT"]),
            low=0.2,
            high=4.0,
        )
        self._rollout_depth = self._read_int(
            "COMPUS_ROLLOUT_DEPTH",
            int(DEFAULT_TUNING["COMPUS_ROLLOUT_DEPTH"]),
            low=2,
            high=24,
        )
        self._rollout_epsilon = self._read_float(
            "COMPUS_ROLLOUT_EPSILON",
            float(DEFAULT_TUNING["COMPUS_ROLLOUT_EPSILON"]),
            low=0.0,
            high=0.65,
        )
        self._risk_low = self._read_float(
            "COMPUS_RISK_LOW",
            float(DEFAULT_TUNING["COMPUS_RISK_LOW"]),
            low=0.0,
            high=2.0,
        )
        self._risk_mid = self._read_float(
            "COMPUS_RISK_MID",
            float(DEFAULT_TUNING["COMPUS_RISK_MID"]),
            low=0.0,
            high=2.0,
        )
        self._risk_high = self._read_float(
            "COMPUS_RISK_HIGH",
            float(DEFAULT_TUNING["COMPUS_RISK_HIGH"]),
            low=0.0,
            high=2.0,
        )
        self._budget_open = self._read_float(
            "COMPUS_BUDGET_OPEN",
            float(DEFAULT_TUNING["COMPUS_BUDGET_OPEN"]),
            low=0.10,
            high=0.92,
        )
        self._budget_mid = self._read_float(
            "COMPUS_BUDGET_MID",
            float(DEFAULT_TUNING["COMPUS_BUDGET_MID"]),
            low=0.10,
            high=0.92,
        )
        self._budget_late = self._read_float(
            "COMPUS_BUDGET_LATE",
            float(DEFAULT_TUNING["COMPUS_BUDGET_LATE"]),
            low=0.10,
            high=0.92,
        )
        self._budget_end = self._read_float(
            "COMPUS_BUDGET_END",
            float(DEFAULT_TUNING["COMPUS_BUDGET_END"]),
            low=0.10,
            high=0.94,
        )
        self._budget_mid = max(self._budget_open, self._budget_mid)
        self._budget_late = max(self._budget_mid, self._budget_late)
        self._budget_end = max(self._budget_late, self._budget_end)
        self._root_limit_scale = self._read_float(
            "COMPUS_ROOT_LIMIT_SCALE",
            float(DEFAULT_TUNING["COMPUS_ROOT_LIMIT_SCALE"]),
            low=0.7,
            high=2.4,
        )
        self._tree_limit_scale = self._read_float(
            "COMPUS_TREE_LIMIT_SCALE",
            float(DEFAULT_TUNING["COMPUS_TREE_LIMIT_SCALE"]),
            low=0.7,
            high=2.2,
        )
        self._rollout_depth_bonus = self._read_int(
            "COMPUS_ROLLOUT_DEPTH_BONUS",
            int(DEFAULT_TUNING["COMPUS_ROLLOUT_DEPTH_BONUS"]),
            low=0,
            high=8,
        )
        self._risk_weight = 0.65

    def on_move_result(
        self,
        move: tuple[int, int],
        success: bool,
    ) -> None:
        self._move_index += 1
        r, c = move

        if self._variant != "dark":
            if success:
                self._last_own_move = move
            return

        if success:
            self._last_own_move = move
            self._risk_map[r][c] = max(0.02, self._risk_map[r][c] * 0.35)
            for nr, nc in self._neighbors[r][c]:
                self._risk_map[nr][nc] = max(0.03, self._risk_map[nr][nc] * 0.90)
            return

        self._known_opp.add(move)
        self._collision_cells.add(move)
        self._risk_map[r][c] = min(4.0, max(self._risk_map[r][c], 1.80))
        for nr, nc in self._neighbors[r][c]:
            self._risk_map[nr][nc] = min(4.0, self._risk_map[nr][nc] + 0.42)
            for nnr, nnc in self._neighbors[nr][nc]:
                if (nnr, nnc) != (r, c):
                    self._risk_map[nnr][nnc] = min(4.0, self._risk_map[nnr][nnc] + 0.08)

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        t0 = time.monotonic()

        board_mut = [list(row) for row in board]
        legal_moves = empty_cells(board_mut, self._size)
        if not legal_moves:
            return (0, 0)
        soft_deadline = t0 + self._soft_budget(len(legal_moves))

        self._refresh_dark_knowledge(board_mut)
        self._risk_weight = self._compute_risk_weight(board_mut)

        tactical_move = self._find_tactical_move(board_mut, legal_moves, soft_deadline)
        if tactical_move is not None:
            self._attempt_heat[tactical_move[0]][tactical_move[1]] += 1
            return tactical_move

        k_root = self._root_candidate_limit(len(legal_moves))
        root_candidates = self._generate_candidates(
            board_mut,
            actor=self._player,
            limit=k_root,
            focus=last_move,
            include_path_boost=True,
        )
        if not root_candidates:
            fallback = legal_moves[0]
            self._attempt_heat[fallback[0]][fallback[1]] += 1
            return fallback

        best_mcts = self._run_mcts(board_mut, root_candidates, soft_deadline)
        if best_mcts is None or board[best_mcts[0]][best_mcts[1]] != 0:
            best_mcts = self._fallback_from_candidates(
                board_mut,
                legal_moves,
                focus=last_move,
            )

        self._attempt_heat[best_mcts[0]][best_mcts[1]] += 1
        return best_mcts

    def _run_mcts(
        self,
        root_board: list[list[int]],
        root_candidates: list[tuple[tuple[int, int], float]],
        deadline: float,
    ) -> tuple[int, int] | None:
        root = _Node(
            board=[row[:] for row in root_board],
            player_to_move=self._player,
            parent=None,
            move=None,
            prior=1.0,
            unexpanded=list(root_candidates),
            children={},
        )

        iterations = 0
        while time.monotonic() < deadline:
            node = root
            path = [node]

            while (not node.unexpanded) and node.children:
                node = self._select_child(node)
                path.append(node)

            if node.unexpanded and time.monotonic() < deadline:
                node = self._expand(node)
                path.append(node)

            value = self._rollout_value(node.board, node.player_to_move, deadline)
            for visited in path:
                visited.visits += 1
                visited.value_sum += value

            iterations += 1
            if iterations % 16 == 0 and time.monotonic() >= deadline:
                break

        if not root.children:
            return root_candidates[0][0] if root_candidates else None

        best_child = max(
            root.children.values(),
            key=lambda child: (child.visits, child.q),
        )
        return best_child.move

    def _select_child(self, node: _Node) -> _Node:
        parent_turn_is_root = node.player_to_move == self._player
        sqrt_parent = math.sqrt(max(1, node.visits))
        best_score = -10_000.0
        best_child: _Node | None = None

        for child in node.children.values():
            exploit = child.q if parent_turn_is_root else -child.q
            explore = self._c_puct * child.prior * sqrt_parent / (1 + child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else next(iter(node.children.values()))

    def _expand(self, node: _Node) -> _Node:
        take_idx = 0
        if len(node.unexpanded) > 2 and self._rng.random() < 0.20:
            take_idx = self._rng.randrange(0, min(3, len(node.unexpanded)))

        move, prior = node.unexpanded.pop(take_idx)
        r, c = move
        next_board = [row[:] for row in node.board]
        next_board[r][c] = node.player_to_move
        next_player = 3 - node.player_to_move

        branch_limit = self._tree_candidate_limit(len(empty_cells(next_board, self._size)))
        child_candidates = self._generate_candidates(
            next_board,
            actor=next_player,
            limit=branch_limit,
            focus=move,
            include_path_boost=False,
        )

        child = _Node(
            board=next_board,
            player_to_move=next_player,
            parent=node,
            move=move,
            prior=prior,
            unexpanded=child_candidates,
            children={},
        )
        node.children[move] = child
        return child

    def _rollout_value(
        self,
        board: list[list[int]],
        player_to_move: int,
        deadline: float,
    ) -> float:
        rollout_board = [row[:] for row in board]
        actor = player_to_move
        stones_played = (self._size * self._size) - len(empty_cells(board, self._size))
        phase_bonus = 2 if stones_played > int(self._size * self._size * 0.55) else 0
        max_depth = min(
            26,
            self._rollout_depth + self._rollout_depth_bonus + phase_bonus,
        )

        for _ in range(max_depth):
            if time.monotonic() >= deadline:
                break

            winner = check_winner(rollout_board, self._size)
            if winner != 0:
                return 1.0 if winner == self._player else -1.0

            moves = empty_cells(rollout_board, self._size)
            if not moves:
                break

            move = self._select_rollout_move(rollout_board, moves, actor)
            rollout_board[move[0]][move[1]] = actor
            actor = 3 - actor

        return self._evaluate_board(rollout_board)

    def _select_rollout_move(
        self,
        board: list[list[int]],
        moves: list[tuple[int, int]],
        actor: int,
    ) -> tuple[int, int]:
        if len(moves) == 1:
            return moves[0]

        base_sample = 10
        if self._time_limit >= 8:
            base_sample = 13
        if len(moves) <= 28:
            base_sample += 2
        sample_n = min(base_sample, len(moves))
        if len(moves) <= sample_n:
            sampled = moves
        else:
            sampled = self._rng.sample(moves, sample_n)

        if self._rng.random() < self._rollout_epsilon:
            return self._rng.choice(sampled)

        focus = self._last_own_move if actor == self._player else None
        return max(
            sampled,
            key=lambda mv: self._quick_move_score(board, mv, actor, focus),
        )

    def _find_tactical_move(
        self,
        board: list[list[int]],
        legal_moves: list[tuple[int, int]],
        deadline: float,
    ) -> tuple[int, int] | None:
        attack = self._find_immediate_win(board, legal_moves, self._player, deadline)
        if attack is not None:
            return attack

        block = self._find_immediate_win(board, legal_moves, self._opponent, deadline)
        if block is not None:
            return block

        return None

    def _find_immediate_win(
        self,
        board: list[list[int]],
        legal_moves: list[tuple[int, int]],
        actor: int,
        deadline: float,
    ) -> tuple[int, int] | None:
        ordered = sorted(
            legal_moves,
            key=lambda mv: self._quick_move_score(
                board,
                mv,
                actor,
                self._last_own_move if actor == self._player else None,
            ),
            reverse=True,
        )

        for move in ordered[: min(len(ordered), 88)]:
            if time.monotonic() >= deadline:
                return None
            r, c = move
            board[r][c] = actor
            winner = check_winner(board, self._size)
            board[r][c] = 0
            if winner == actor:
                return move
        return None

    def _generate_candidates(
        self,
        board: list[list[int]],
        actor: int,
        limit: int,
        focus: tuple[int, int] | None,
        include_path_boost: bool,
    ) -> list[tuple[tuple[int, int], float]]:
        moves = empty_cells(board, self._size)
        if not moves:
            return []
        if len(moves) <= limit:
            scored_small: list[tuple[tuple[int, int], float]] = []
            for mv in moves:
                score = self._quick_move_score(board, mv, actor, focus)
                scored_small.append((mv, score))
            if include_path_boost and len(moves) <= 18:
                base_actor = shortest_path_distance(board, self._size, actor)
                rival = 3 - actor
                base_rival = shortest_path_distance(board, self._size, rival)
                boosted_small: list[tuple[tuple[int, int], float]] = []
                for mv, quick_score in scored_small:
                    r, c = mv
                    board[r][c] = actor
                    new_actor = shortest_path_distance(board, self._size, actor)
                    new_rival = shortest_path_distance(board, self._size, rival)
                    board[r][c] = 0
                    path_gain = (base_actor - new_actor) + 0.65 * (new_rival - base_rival)
                    boosted_small.append((mv, quick_score + 0.55 * path_gain))
                scored_small = boosted_small
            return self._normalize_priors(scored_small)

        scored_quick: list[tuple[float, tuple[int, int]]] = []
        for mv in moves:
            scored_quick.append((self._quick_move_score(board, mv, actor, focus), mv))
        scored_quick.sort(reverse=True, key=lambda item: item[0])

        pre_limit = min(len(scored_quick), max(limit * 3, 24))
        shortlist = scored_quick[:pre_limit]

        final_scored: list[tuple[float, tuple[int, int]]] = []
        if include_path_boost:
            base_actor = shortest_path_distance(board, self._size, actor)
            rival = 3 - actor
            base_rival = shortest_path_distance(board, self._size, rival)
            for quick_score, mv in shortlist:
                r, c = mv
                board[r][c] = actor
                new_actor = shortest_path_distance(board, self._size, actor)
                new_rival = shortest_path_distance(board, self._size, rival)
                board[r][c] = 0

                path_gain = (base_actor - new_actor) + 0.65 * (new_rival - base_rival)
                final_scored.append((quick_score + 0.55 * path_gain, mv))
        else:
            final_scored = shortlist

        final_scored.sort(reverse=True, key=lambda item: item[0])
        chosen = [((r, c), score) for score, (r, c) in final_scored[:limit]]
        return self._normalize_priors(chosen)

    def _normalize_priors(
        self,
        scored_moves: list[tuple[tuple[int, int], float]],
    ) -> list[tuple[tuple[int, int], float]]:
        if not scored_moves:
            return []
        max_score = max(score for _, score in scored_moves)
        exp_vals = []
        for _, score in scored_moves:
            exp_vals.append(math.exp((score - max_score) * 0.85))
        total = sum(exp_vals)
        if total <= 0.0:
            uniform = 1.0 / len(scored_moves)
            return [(mv, uniform) for mv, _ in scored_moves]
        return [
            (scored_moves[i][0], exp_vals[i] / total)
            for i in range(len(scored_moves))
        ]

    def _quick_move_score(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        actor: int,
        focus: tuple[int, int] | None,
    ) -> float:
        r, c = move
        rival = 3 - actor
        own_n = 0
        rival_n = 0
        free_n = 0
        for nr, nc in self._neighbors[r][c]:
            cell = board[nr][nc]
            if cell == actor:
                own_n += 1
            elif cell == rival:
                rival_n += 1
            else:
                free_n += 1

        center_score = self._center_score_map[r][c]

        support = 1.00 * own_n + (0.40 if own_n >= 2 else 0.0)
        pressure = 0.72 * rival_n + (0.24 if rival_n >= 2 else 0.0)
        mobility = 0.11 * free_n
        bridge = self._bridge_touch_score(board, move, actor)

        axis = self._axis_progress(r, c, actor)
        focus_bonus = 0.0
        if focus is not None:
            manhattan = abs(r - focus[0]) + abs(c - focus[1])
            focus_bonus = max(0.0, 1.8 - 0.28 * manhattan)

        score = (
            1.08 * support
            + 0.86 * pressure
            + 0.48 * bridge
            + 0.34 * center_score
            + 0.22 * axis
            + 0.15 * mobility
            + 0.38 * focus_bonus
        )

        if self._variant == "dark" and actor == self._player:
            risk = self._risk_map[r][c] + 0.07 * self._attempt_heat[r][c]
            uncertainty = self._local_uncertainty(board, move)
            score -= self._risk_weight * risk
            score += 0.22 * uncertainty

        score += self._rng.uniform(-0.035, 0.035)
        return score

    def _bridge_touch_score(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        actor: int,
    ) -> float:
        r, c = move
        rival = 3 - actor
        score = 0.0
        for nr, nc in self._neighbors[r][c]:
            if board[nr][nc] != actor:
                continue
            common_empty = 0
            for nnr, nnc in self._neighbors[nr][nc]:
                if (nnr, nnc) == (r, c):
                    continue
                if board[nnr][nnc] == 0:
                    common_empty += 1
            if common_empty >= 2:
                score += 0.58
            elif common_empty == 1:
                score += 0.24

        for tr, tc, c1r, c1c, c2r, c2c in self._bridge_patterns[r][c]:
            target = board[tr][tc]
            conn_1 = board[c1r][c1c]
            conn_2 = board[c2r][c2c]
            if target == actor:
                blocked = int(conn_1 == rival) + int(conn_2 == rival)
                if blocked == 0:
                    score += 0.78
                elif blocked == 1:
                    score += 0.22
            elif target == 0:
                if conn_1 == 0 and conn_2 == 0:
                    score += 0.16
                elif (conn_1 == actor and conn_2 == 0) or (conn_2 == actor and conn_1 == 0):
                    score += 0.12
        return score

    def _axis_progress(self, r: int, c: int, actor: int) -> float:
        return self._axis_score_map[actor][r][c]

    def _local_uncertainty(
        self,
        board: list[list[int]],
        move: tuple[int, int],
    ) -> float:
        r, c = move
        unknown = 0
        known_blocked = 0
        for nr, nc in self._neighbors[r][c]:
            if board[nr][nc] == 0 and (nr, nc) not in self._known_opp:
                unknown += 1
            if (nr, nc) in self._known_opp:
                known_blocked += 1
        return 0.20 * unknown - 0.18 * known_blocked

    def _evaluate_board(self, board: list[list[int]]) -> float:
        winner = check_winner(board, self._size)
        if winner == self._player:
            return 1.0
        if winner == self._opponent:
            return -1.0

        my_dist = shortest_path_distance(board, self._size, self._player)
        opp_dist = shortest_path_distance(board, self._size, self._opponent)
        path_term = (opp_dist - my_dist) / max(1.0, self._size * 0.9)

        conn_term = (
            self._connectivity(board, self._player) - self._connectivity(board, self._opponent)
        )
        bridge_term = (
            self._bridge_density(board, self._player) - self._bridge_density(board, self._opponent)
        )
        frag_term = (
            self._fragmentation(board, self._player) - self._fragmentation(board, self._opponent)
        )

        score = (
            0.72 * path_term
            + 0.16 * conn_term
            + 0.12 * bridge_term
            - 0.10 * frag_term
        )
        return math.tanh(score * 0.62)

    def _connectivity(self, board: list[list[int]], player: int) -> float:
        stones = 0
        links = 0
        frontier = 0
        for r, c in self._cells:
            if board[r][c] != player:
                continue
            stones += 1
            for nr, nc in self._neighbors[r][c]:
                if board[nr][nc] == player:
                    links += 1
                elif board[nr][nc] == 0:
                    frontier += 1
        if stones == 0:
            return -0.3
        return ((links * 0.5) + 0.15 * frontier) / stones

    def _bridge_density(self, board: list[list[int]], player: int) -> float:
        score = 0.0
        empties = empty_cells(board, self._size)
        for r, c in empties:
            own_n = 0
            for nr, nc in self._neighbors[r][c]:
                if board[nr][nc] == player:
                    own_n += 1
            if own_n >= 2:
                score += 0.55 + 0.20 * (own_n - 2)
        return score / max(1.0, self._size)

    def _fragmentation(self, board: list[list[int]], player: int) -> float:
        seen = [[False for _ in range(self._size)] for _ in range(self._size)]
        groups = 0
        stones = 0

        for r, c in self._cells:
            if board[r][c] != player:
                continue
            stones += 1
            if seen[r][c]:
                continue
            groups += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in self._neighbors[cr][cc]:
                    if board[nr][nc] == player and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

        if stones <= 1:
            return 0.0
        return (groups - 1) / stones

    def _fallback_from_candidates(
        self,
        board: list[list[int]],
        legal_moves: list[tuple[int, int]],
        focus: tuple[int, int] | None,
    ) -> tuple[int, int]:
        ranked = sorted(
            legal_moves,
            key=lambda mv: self._quick_move_score(board, mv, self._player, focus),
            reverse=True,
        )
        return ranked[0] if ranked else legal_moves[0]

    def _refresh_dark_knowledge(self, board: list[list[int]]) -> None:
        if self._variant != "dark":
            return
        for r, c in self._cells:
            self._risk_map[r][c] *= 0.985
            if board[r][c] == self._opponent:
                self._known_opp.add((r, c))
                self._risk_map[r][c] = max(self._risk_map[r][c], 1.65)

    def _compute_risk_weight(self, board: list[list[int]]) -> float:
        if self._variant != "dark":
            return 0.0
        my_dist = shortest_path_distance(board, self._size, self._player)
        if my_dist <= 2:
            return self._risk_low
        if my_dist <= 4:
            return self._risk_mid
        return self._risk_high

    def _soft_budget(self, num_moves: int) -> float:
        if num_moves > 90:
            frac = self._budget_open
        elif num_moves > 60:
            frac = self._budget_mid
        elif num_moves > 30:
            frac = self._budget_late
        else:
            frac = self._budget_end

        reserve = min(0.85, max(0.10, self._time_limit * 0.06))
        hard_cap = self._time_limit * 0.88
        if num_moves > 35:
            hard_cap = min(hard_cap, 5.8)
        elif num_moves > 18:
            hard_cap = min(hard_cap, 6.6)
        else:
            hard_cap = min(hard_cap, 7.4)

        candidate = self._time_limit * frac
        return max(0.05, min(hard_cap, candidate, self._time_limit - reserve))

    def _root_candidate_limit(self, num_moves: int) -> int:
        if num_moves > 90:
            base = 20
        elif num_moves > 60:
            base = 24
        elif num_moves > 35:
            base = 28
        elif num_moves > 18:
            base = 32
        else:
            base = 36
        scaled = int(round(base * self._root_limit_scale))
        return min(max(8, scaled), num_moves)

    def _tree_candidate_limit(self, num_moves: int) -> int:
        if num_moves > 80:
            base = 10
        elif num_moves > 45:
            base = 12
        elif num_moves > 20:
            base = 14
        else:
            base = 16
        scaled = int(round(base * self._tree_limit_scale))
        return min(max(6, scaled), num_moves)

    def _build_center_score_map(self) -> list[list[float]]:
        center = (self._size - 1) * 0.5
        norm = max(1.0, float(self._size))
        table = [[0.0 for _ in range(self._size)] for _ in range(self._size)]
        for r, c in self._cells:
            center_dist = abs(r - center) + abs(c - center)
            table[r][c] = 1.0 - (center_dist / norm)
        return table

    def _build_axis_score_map(self, actor: int) -> list[list[float]]:
        center = (self._size - 1) * 0.5
        norm = max(1.0, center)
        table = [[0.0 for _ in range(self._size)] for _ in range(self._size)]
        for r, c in self._cells:
            if actor == 1:
                perp_dist = abs(c - center)
            else:
                perp_dist = abs(r - center)
            table[r][c] = 1.0 - (perp_dist / norm)
        return table

    def _build_bridge_patterns(
        self,
    ) -> list[list[list[tuple[int, int, int, int, int, int]]]]:
        neigh_sets = [
            [set(self._neighbors[r][c]) for c in range(self._size)]
            for r in range(self._size)
        ]
        patterns: list[list[list[tuple[int, int, int, int, int, int]]]] = [
            [[] for _ in range(self._size)]
            for _ in range(self._size)
        ]

        for r, c in self._cells:
            base_neighbors = neigh_sets[r][c]
            for tr, tc in self._cells:
                if (tr, tc) == (r, c) or (tr, tc) in base_neighbors:
                    continue
                shared = base_neighbors.intersection(neigh_sets[tr][tc])
                if len(shared) != 2:
                    continue
                connectors = sorted(shared)
                patterns[r][c].append(
                    (
                        tr,
                        tc,
                        connectors[0][0],
                        connectors[0][1],
                        connectors[1][0],
                        connectors[1][1],
                    )
                )
        return patterns

    def _read_float(self, name: str, default: float, low: float, high: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return min(high, max(low, value))

    def _read_int(self, name: str, default: int, low: int, high: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return min(high, max(low, value))

    def _read_optional_int(self, name: str) -> int | None:
        raw = os.getenv(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
