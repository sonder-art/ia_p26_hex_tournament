"""Hex strategy for team MALIK_RUBEN.

Version 10:
- classic uses wider corridor-based candidate generation
- a flat root Monte Carlo search evaluates 18-24 moves with biased rollouts
- classic also scores pure cuts and explicit opponent replies before rollouts
- dark mode stays conservative and collision-aware
"""

from __future__ import annotations

import heapq
import math
import random
import time

from strategy import GameConfig, Strategy
from hex_game import check_winner, empty_cells, get_neighbors, shortest_path_distance


class _SearchTimeout(Exception):
    pass


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
        self._turn_count = 0
        self._known_collisions: set[tuple[int, int]] = set()
        self._opening = self._build_opening_book()
        self._rng = random.Random()

    def on_move_result(
        self,
        move: tuple[int, int],
        success: bool,
    ) -> None:
        if not success:
            self._known_collisions.add(move)

    def play(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        self._turn_count += 1
        self._deadline = time.monotonic() + max(0.05, self._time_limit * 0.92)
        self._ctx_cache: dict[tuple[tuple[tuple[int, ...], ...], int], dict] = {}
        self._endgame_cache: dict[
            tuple[tuple[tuple[int, ...], ...], int],
            tuple[int, tuple[int, int] | None, int],
        ] = {}

        if self._variant == "classic":
            return self._play_classic(board, last_move)
        return self._play_dark(board)

    # ---------------------------------------------------------------
    # Classic mode
    # ---------------------------------------------------------------

    def _play_classic(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        empties = empty_cells(board, self._size)
        if len(empties) == 1:
            return empties[0]

        opening = self._opening_move(board)
        if opening is not None:
            return opening

        winning = self._find_immediate_wins(board, self._player, empties)
        if winning:
            return self._pick_best(board, winning)

        blocking = self._find_immediate_wins(board, self._opponent, empties)
        if blocking:
            return self._pick_best(board, blocking)

        my_ctx = self._critical_context(board, self._player)
        opp_ctx = self._critical_context(board, self._opponent)
        opp_corridor = self._narrow_corridor_pressure(board, self._opponent) if self._player == 2 else None
        my_profile = self._shape_profile(board, self._player)
        conversion_ctx = self._conversion_context(my_ctx, opp_ctx, my_profile, opp_corridor)

        # When the board is nearly full, exact search is more reliable than rollout noise.
        solved = self._maybe_solve_endgame(board, empties, last_move, my_ctx, opp_ctx)
        if solved is not None:
            return solved

        if (
            opp_ctx["best"] <= 5
            or (opp_ctx["best"] <= 6 and opp_ctx["best"] <= my_ctx["best"] + 1)
            or (
                self._player == 2
                and opp_corridor is not None
                and opp_corridor["threat"] >= 7.0
                and opp_ctx["best"] <= my_ctx["best"] + 2
            )
        ):
            forced = self._forced_block_candidates(board, opp_ctx, my_ctx, last_move, opp_corridor)
            if forced:
                return self._choose_classic_move(
                    board,
                    forced,
                    my_ctx,
                    opp_ctx,
                    last_move,
                    force_block=True,
                    opp_corridor=opp_corridor,
                    my_profile=my_profile,
                    conversion_ctx=conversion_ctx,
                )

        candidates = self._candidate_moves_classic(
            board,
            empties,
            my_ctx,
            opp_ctx,
            last_move,
            opp_corridor,
            my_profile,
        )
        return self._choose_classic_move(
            board,
            candidates,
            my_ctx,
            opp_ctx,
            last_move,
            force_block=False,
            opp_corridor=opp_corridor,
            my_profile=my_profile,
            conversion_ctx=conversion_ctx,
        )

    def _candidate_moves_classic(
        self,
        board: tuple[tuple[int, ...], ...],
        empties: list[tuple[int, int]],
        my_ctx: dict,
        opp_ctx: dict,
        last_move: tuple[int, int] | None,
        opp_corridor: dict | None = None,
        my_profile: dict | None = None,
    ) -> list[tuple[int, int]]:
        scored: list[tuple[float, tuple[int, int]]] = []
        contact_scored: list[tuple[float, tuple[int, int]]] = []
        cut_scored: list[tuple[float, tuple[int, int]]] = []
        threat = max(0, 7 - opp_ctx["best"])
        opp_weight = 2.0 + 0.45 * threat
        my_weight = 3.2 - 0.15 * min(threat, 4)
        for move in empties:
            own_adj = self._count_adjacent(board, move, self._player)
            opp_adj = self._count_adjacent(board, move, self._opponent)
            own_merge = self._group_merge_potential(board, move, self._player)
            opp_merge = self._group_merge_potential(board, move, self._opponent)
            cut_gain = self._block_distance_delta(board, move, self._opponent, opp_ctx["best"])
            pure_cut = self._pure_cut_score(board, move, opp_ctx, cut_gain)
            my_critical = my_ctx["critical"].get(move, 0.0)
            opp_critical = opp_ctx["critical"].get(move, 0.0)
            bridge_bonus = self._bridge_move_bonus(board, move, self._player)
            axis_bonus = self._axis_alignment(move, self._player)
            goal_bonus = self._goal_edge_bonus(move, self._player)
            near_goal_bonus = self._near_goal_edge_bonus(move, self._player)
            white_mid_bonus = 0.0
            contact = (
                1.4 * min(1, own_adj)
                + 1.7 * min(1, opp_adj)
                + 0.8 * own_merge
                + 0.5 * opp_merge
            )
            score = (
                opp_weight * opp_critical
                + my_weight * my_critical
                + 1.8 * own_adj
                + 1.0 * opp_adj
                + 1.4 * own_merge
                + 0.8 * opp_merge
                + 1.5 * contact
                + 1.25 * pure_cut
                + 1.1 * bridge_bonus
                + 0.8 * axis_bonus
                + 0.8 * goal_bonus
                + 0.6 * near_goal_bonus
            )
            if self._player == 2 and opp_corridor is not None:
                white_mid_bonus = self._white_midgame_bonus(
                    move,
                    opp_corridor,
                    opp_critical,
                    opp_adj,
                    opp_merge,
                    pure_cut,
                )
                score += white_mid_bonus
                if my_profile is not None:
                    span_extend = self._objective_span_extension_from_profile(my_profile, move, self._player)
                    if span_extend > 0.0:
                        score += 0.9 * span_extend
                    overcommit = self._white_overcommit_pressure(my_profile)
                    if overcommit > 0.0 and span_extend <= 0.0 and own_merge == 0 and pure_cut <= 0.0 and white_mid_bonus < 1.0:
                        score -= overcommit
            if own_adj == 0 and my_ctx["critical"].get(move, 0.0) < 1.0 and opp_ctx["best"] > 5:
                score -= 2.5
            if last_move is not None and move in get_neighbors(last_move[0], last_move[1], self._size):
                score += 0.9
            scored.append((score, move))
            if contact > 0.0 or cut_gain > 0:
                contact_scored.append((contact + 1.4 * cut_gain, move))
            cut_scored.append((pure_cut, move))

        scored.sort(reverse=True)
        contact_scored.sort(reverse=True)
        cut_scored.sort(reverse=True)
        candidates = [move for _, move in scored[:16]]
        extras = [move for move, _ in opp_ctx["top"][:10]] + [move for move, _ in my_ctx["top"][:8]]
        extras.extend(move for _, move in contact_scored[:8])
        extras.extend(move for _, move in cut_scored[:10])
        if last_move is not None:
            extras.extend(
                move
                for move in get_neighbors(last_move[0], last_move[1], self._size)
                if board[move[0]][move[1]] == 0
            )
        return self._dedupe_moves(extras + candidates)

    def _forced_block_candidates(
        self,
        board: tuple[tuple[int, ...], ...],
        opp_ctx: dict,
        my_ctx: dict,
        last_move: tuple[int, int] | None,
        opp_corridor: dict | None = None,
    ) -> list[tuple[int, int]]:
        moves = [move for move, _ in opp_ctx["top"][:8]]
        moves.extend(move for move, _ in my_ctx["top"][:2])
        empties = empty_cells(board, self._size)
        cut_scored = [
            (
                self._pure_cut_score(
                    board,
                    move,
                    opp_ctx,
                    self._block_distance_delta(board, move, self._opponent, opp_ctx["best"]),
                ),
                move,
            )
            for move in empties
        ]
        cut_scored.sort(reverse=True)
        moves.extend(move for _, move in cut_scored[:10])
        if opp_corridor is not None and opp_corridor["frontier"]:
            frontier_scored = []
            for move in opp_corridor["frontier"]:
                if board[move[0]][move[1]] != 0:
                    continue
                frontier_scored.append(
                    (
                        opp_ctx["critical"].get(move, 0.0)
                        + 1.5 * self._count_adjacent(board, move, self._opponent)
                        + 0.9 * self._group_merge_potential(board, move, self._opponent),
                        move,
                    )
                )
            frontier_scored.sort(reverse=True)
            moves.extend(move for _, move in frontier_scored[:10])
        if last_move is not None:
            moves.extend(
                move
                for move in get_neighbors(last_move[0], last_move[1], self._size)
                if board[move[0]][move[1]] == 0
            )
        return self._dedupe_moves(moves)

    def _choose_classic_move(
        self,
        board: tuple[tuple[int, ...], ...],
        candidates: list[tuple[int, int]],
        my_ctx: dict,
        opp_ctx: dict,
        last_move: tuple[int, int] | None,
        force_block: bool,
        opp_corridor: dict | None = None,
        my_profile: dict | None = None,
        conversion_ctx: dict | None = None,
    ) -> tuple[int, int]:
        candidates = [move for move in candidates if board[move[0]][move[1]] == 0]
        if not candidates:
            return empty_cells(board, self._size)[0]

        use_two_stage = self._player == 2
        shortlisted: list[dict | tuple[int, int]]
        if use_two_stage:
            stage_entries: list[dict] = []
            for move in candidates:
                if self._out_of_time():
                    break
                stage_entries.append(
                    self._cheap_classic_stage_entry(
                        board,
                        move,
                        my_ctx,
                        opp_ctx,
                        last_move,
                        force_block,
                        opp_corridor,
                        my_profile,
                    )
                )

            if not stage_entries:
                return candidates[0]

            shortlisted = self._stage_b_shortlist(
                stage_entries,
                my_ctx,
                opp_ctx,
                last_move,
                force_block,
            )
        else:
            shortlisted = list(candidates)

        base_entries: list[dict] = []
        for candidate_entry in shortlisted:
            if self._out_of_time():
                break
            if use_two_stage:
                stage_entry = candidate_entry
                move = stage_entry["move"]
                own_adj = stage_entry["own_adj"]
                own_merge = stage_entry["own_merge"]
                dual_bonus = stage_entry["dual_bonus"]
                cut_gain = stage_entry["cut_gain"]
                pure_cut = stage_entry["pure_cut"]
                white_mid_bonus = stage_entry["white_mid_bonus"]
                span_extend = stage_entry["span_extend"]
                overcommit = stage_entry["overcommit"]
            else:
                move = candidate_entry
                own_adj = self._count_adjacent(board, move, self._player)
                own_merge = self._group_merge_potential(board, move, self._player)
                dual_bonus = self._dual_purpose_bonus(board, move, my_ctx, opp_ctx)
                cut_gain = self._block_distance_delta(board, move, self._opponent, opp_ctx["best"])
                pure_cut = self._pure_cut_score(board, move, opp_ctx, cut_gain)
                white_mid_bonus = 0.0
                span_extend = (
                    self._objective_span_extension_from_profile(my_profile, move, self._player)
                    if my_profile is not None
                    else 0.0
                )
                overcommit = self._white_overcommit_pressure(my_profile) if my_profile is not None else 0.0
            trial = self._with_move(board, move, self._player)
            if check_winner(trial, self._size) == self._player:
                return move

            after_my = self._critical_context(trial, self._player)
            after_opp = self._critical_context(trial, self._opponent)

            block_delta = after_opp["best"] - opp_ctx["best"]
            build_delta = my_ctx["best"] - after_my["best"]
            block_reduction = opp_ctx["mass"] - after_opp["mass"]
            build_reduction = after_my["mass"] - my_ctx["mass"]
            reply_value, reply_pressure = self._opponent_reply_signal(trial, move)
            score = self._position_score(trial, after_my, after_opp)
            if force_block:
                score += 18.0 * block_delta + 1.7 * block_reduction + 4.0 * build_delta + 3.5 * dual_bonus
                score += 5.0 * cut_gain + 14.0 * (reply_value - 0.5) - 3.0 * reply_pressure
            else:
                score += 10.0 * block_delta + 5.0 * build_delta + 1.2 * block_reduction + 0.8 * build_reduction + 2.2 * dual_bonus
                score += 2.0 * cut_gain + 10.0 * (reply_value - 0.5) - 2.4 * reply_pressure
            if self._player == 2 and opp_corridor is not None:
                if not use_two_stage:
                    white_mid_bonus = self._white_midgame_bonus(
                        move,
                        opp_corridor,
                        opp_ctx["critical"].get(move, 0.0),
                        self._count_adjacent(board, move, self._opponent),
                        self._group_merge_potential(board, move, self._opponent),
                        pure_cut,
                    )
                score += 1.15 * white_mid_bonus
                after_corridor = self._narrow_corridor_pressure(trial, self._opponent)
                score += 0.95 * max(0.0, opp_corridor["threat"] - after_corridor["threat"])
                if my_profile is not None:
                    if span_extend > 0.0:
                        score += 1.0 * span_extend
                    if (
                        overcommit > 0.0
                        and span_extend <= 0.0
                        and own_merge == 0
                        and cut_gain == 0
                        and white_mid_bonus < 1.0
                    ):
                        score -= 1.1 * overcommit
            if conversion_ctx is not None and conversion_ctx["active"] and not force_block and my_profile is not None:
                after_profile = self._shape_profile(trial, self._player)
                raw_progress_gain = after_profile["progress_span"] - my_profile["progress_span"]
                connected_progress_gain = 0.0
                if own_adj > 0 or own_merge > 0 or build_delta > 0:
                    connected_progress_gain = max(0.0, raw_progress_gain)
                connected_span_extend = max(0.0, span_extend) if (own_adj > 0 or own_merge > 0 or build_delta > 0) else 0.0
                conversion_bonus = (
                    3.2 * max(0.0, build_delta)
                    + 0.7 * max(0.0, build_reduction)
                    + 1.25 * connected_progress_gain
                    + 0.55 * connected_span_extend
                    + 0.65 * own_merge
                    + 0.2 * min(3, own_adj)
                )
                if build_delta <= 0 and connected_progress_gain <= 0.0 and own_merge == 0 and connected_span_extend <= 0.0:
                    conversion_bonus -= 0.9
                score += conversion_ctx["strength"] * conversion_bonus

            base_entries.append(
                {
                    "move": move,
                    "board": trial,
                    "after_my": after_my,
                    "after_opp": after_opp,
                    "base": score,
                    "block_delta": block_delta,
                    "build_delta": build_delta,
                    "heur_value": 0.58 * self._normalized_state_value(after_my, after_opp) + 0.42 * reply_value,
                    "reply_value": reply_value,
                    "reply_pressure": reply_pressure,
                    "visits": 0,
                    "value": 0.0,
                }
            )

        if not base_entries:
            if use_two_stage:
                return shortlisted[0]["move"]
            return shortlisted[0]

        base_entries.sort(key=lambda entry: entry["base"], reverse=True)
        width = 24 if self._time_left() > max(2.2, self._time_limit * 0.18) else 18
        finalists = base_entries[: min(width, len(base_entries))]
        if len(finalists) == 1 or self._time_left() < max(0.35, self._time_limit * 0.08):
            return finalists[0]["move"]

        return self._flat_root_search(finalists, force_block)

    def _cheap_classic_stage_entry(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        my_ctx: dict,
        opp_ctx: dict,
        last_move: tuple[int, int] | None,
        force_block: bool,
        opp_corridor: dict | None = None,
        my_profile: dict | None = None,
    ) -> dict:
        own_adj = self._count_adjacent(board, move, self._player)
        opp_adj = self._count_adjacent(board, move, self._opponent)
        own_merge = self._group_merge_potential(board, move, self._player)
        opp_merge = self._group_merge_potential(board, move, self._opponent)
        my_critical = my_ctx["critical"].get(move, 0.0)
        opp_critical = opp_ctx["critical"].get(move, 0.0)
        cut_gain = self._block_distance_delta(board, move, self._opponent, opp_ctx["best"])
        pure_cut = self._pure_cut_score(board, move, opp_ctx, cut_gain)
        dual_bonus = 0.9 * own_adj + 1.4 * own_merge + 0.7 * my_critical + 0.5 * opp_critical
        bridge_bonus = self._bridge_move_bonus(board, move, self._player)
        axis_bonus = self._axis_alignment(move, self._player)
        goal_bonus = self._goal_edge_bonus(move, self._player)
        near_goal_bonus = self._near_goal_edge_bonus(move, self._player)
        contact = (
            1.3 * min(1, own_adj)
            + 1.5 * min(1, opp_adj)
            + 0.75 * own_merge
            + 0.45 * opp_merge
        )
        span_extend = (
            self._objective_span_extension_from_profile(my_profile, move, self._player)
            if my_profile is not None
            else 0.0
        )
        white_mid_bonus = 0.0
        overcommit = 0.0

        if force_block:
            stage_score = (
                2.7 * opp_critical
                + 2.2 * pure_cut
                + 1.7 * cut_gain
                + 1.25 * opp_adj
                + 0.8 * opp_merge
                + 1.1 * dual_bonus
                + 0.7 * own_adj
                + 0.45 * own_merge
                + 0.35 * bridge_bonus
            )
        else:
            stage_score = (
                1.9 * my_critical
                + 1.55 * opp_critical
                + 1.25 * own_adj
                + 1.2 * own_merge
                + 0.9 * opp_adj
                + 0.55 * opp_merge
                + 1.15 * dual_bonus
                + 0.95 * contact
                + 0.75 * pure_cut
                + 0.45 * cut_gain
                + 0.5 * bridge_bonus
                + 0.25 * axis_bonus
                + 0.3 * goal_bonus
                + 0.15 * near_goal_bonus
            )

        if self._player == 2 and opp_corridor is not None:
            white_mid_bonus = self._white_midgame_bonus(
                move,
                opp_corridor,
                opp_critical,
                opp_adj,
                opp_merge,
                pure_cut,
            )
            stage_score += white_mid_bonus
            if span_extend > 0.0:
                stage_score += 0.7 * span_extend
            if my_profile is not None:
                overcommit = self._white_overcommit_pressure(my_profile)
                if overcommit > 0.0 and span_extend <= 0.0 and own_merge == 0 and pure_cut <= 0.0 and white_mid_bonus < 1.0:
                    stage_score -= 0.8 * overcommit

        if own_adj == 0 and my_critical < 1.0 and opp_ctx["best"] > 5:
            stage_score -= 2.0
        if last_move is not None and move in get_neighbors(last_move[0], last_move[1], self._size):
            stage_score += 0.75

        return {
            "move": move,
            "stage": stage_score,
            "own_adj": own_adj,
            "opp_adj": opp_adj,
            "own_merge": own_merge,
            "opp_merge": opp_merge,
            "my_critical": my_critical,
            "opp_critical": opp_critical,
            "cut_gain": cut_gain,
            "pure_cut": pure_cut,
            "dual_bonus": dual_bonus,
            "white_mid_bonus": white_mid_bonus,
            "span_extend": span_extend,
            "overcommit": overcommit,
        }

    def _stage_b_shortlist(
        self,
        stage_entries: list[dict],
        my_ctx: dict,
        opp_ctx: dict,
        last_move: tuple[int, int] | None,
        force_block: bool,
    ) -> list[dict]:
        if not stage_entries:
            return []

        ordered = sorted(stage_entries, key=lambda entry: entry["stage"], reverse=True)
        time_left = self._time_left()
        if force_block:
            base_width = 14 if time_left > max(2.2, self._time_limit * 0.15) else 12
            cap = min(len(ordered), base_width + 6)
        else:
            if time_left > max(2.4, self._time_limit * 0.16):
                base_width = 12
            elif time_left > max(1.3, self._time_limit * 0.09):
                base_width = 10
            else:
                base_width = 8
            cap = min(len(ordered), base_width + 6)

        if len(ordered) <= cap:
            return ordered

        entry_by_move = {entry["move"]: entry for entry in ordered}
        locked: set[tuple[int, int]] = set()

        for move, _ in opp_ctx["top"][: 6 if force_block else 4]:
            if move in entry_by_move:
                locked.add(move)
        for move, _ in my_ctx["top"][:4]:
            if move in entry_by_move:
                locked.add(move)
        for entry in sorted(ordered, key=lambda item: item["pure_cut"], reverse=True)[: 5 if force_block else 4]:
            locked.add(entry["move"])
        for entry in sorted(
            ordered,
            key=lambda item: item["opp_critical"] + 0.8 * item["cut_gain"],
            reverse=True,
        )[:4]:
            locked.add(entry["move"])
        for entry in sorted(
            ordered,
            key=lambda item: item["my_critical"] + 0.9 * item["own_merge"] + 0.45 * item["own_adj"],
            reverse=True,
        )[:4]:
            locked.add(entry["move"])

        if last_move is not None:
            neighbor_set = set(get_neighbors(last_move[0], last_move[1], self._size))
            neighbor_entries = [entry for entry in ordered if entry["move"] in neighbor_set]
            for entry in neighbor_entries[:4]:
                locked.add(entry["move"])

        shortlist: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for entry in sorted((entry_by_move[move] for move in locked), key=lambda item: item["stage"], reverse=True):
            shortlist.append(entry)
            seen.add(entry["move"])

        for entry in ordered:
            if entry["move"] in seen:
                continue
            shortlist.append(entry)
            seen.add(entry["move"])
            if len(shortlist) >= cap:
                break

        return shortlist

    def _maybe_solve_endgame(
        self,
        board: tuple[tuple[int, ...], ...],
        empties: list[tuple[int, int]],
        last_move: tuple[int, int] | None,
        my_ctx: dict,
        opp_ctx: dict,
    ) -> tuple[int, int] | None:
        count = len(empties)
        if count > 10:
            if count > 12:
                return None
            if self._time_left() < max(4.5, self._time_limit * 0.28):
                return None
            if min(my_ctx["best"], opp_ctx["best"]) > 5:
                return None

        self._solver_reserve = max(0.12, self._time_limit * 0.04)
        try:
            _, move = self._solve_endgame(board, self._player, last_move, -256, 256)
        except _SearchTimeout:
            return None
        return move

    def _solve_endgame(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        last_move: tuple[int, int] | None,
        alpha: int,
        beta: int,
    ) -> tuple[int, tuple[int, int] | None]:
        if self._time_left() <= self._solver_reserve:
            raise _SearchTimeout

        empties = empty_cells(board, self._size)
        remaining = len(empties)
        winner = check_winner(board, self._size)
        if winner != 0:
            value = remaining + 1
            return (value if winner == player else -value), None
        if not empties:
            return -1, None

        alpha_orig = alpha
        beta_orig = beta
        cache_key = (board, player)
        cached = self._endgame_cache.get(cache_key)
        tt_move: tuple[int, int] | None = None
        if cached is not None:
            cached_value, cached_move, cached_flag = cached
            tt_move = cached_move
            if cached_flag == 0:
                return cached_value, cached_move
            if cached_flag > 0:
                alpha = max(alpha, cached_value)
            else:
                beta = min(beta, cached_value)
            if alpha >= beta:
                return cached_value, cached_move

        moves = self._endgame_ordered_moves(board, player, empties, last_move, tt_move)
        best_value = -256
        best_move = moves[0]

        for move in moves:
            trial = self._with_move(board, move, player)
            if check_winner(trial, self._size) == player:
                value = remaining
            else:
                child_value, _ = self._solve_endgame(
                    trial,
                    3 - player,
                    move,
                    -beta,
                    -alpha,
                )
                value = -child_value

            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        flag = 0
        if best_value <= alpha_orig:
            flag = -1
        elif best_value >= beta_orig:
            flag = 1
        self._endgame_cache[cache_key] = (best_value, best_move, flag)
        return best_value, best_move

    def _endgame_ordered_moves(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        empties: list[tuple[int, int]],
        last_move: tuple[int, int] | None,
        tt_move: tuple[int, int] | None,
    ) -> list[tuple[int, int]]:
        opponent = 3 - player
        winning = set(self._find_immediate_wins(board, player, empties))
        blocking = set(self._find_immediate_wins(board, opponent, empties))
        scored: list[tuple[float, tuple[int, int]]] = []

        for move in empties:
            own_adj = self._count_adjacent(board, move, player)
            opp_adj = self._count_adjacent(board, move, opponent)
            own_merge = self._group_merge_potential(board, move, player)
            opp_merge = self._group_merge_potential(board, move, opponent)
            bridge = self._bridge_move_bonus(board, move, player)
            score = (
                2.8 * own_adj
                + 2.1 * own_merge
                + 1.7 * opp_adj
                + 1.2 * opp_merge
                + 1.0 * bridge
                + 0.8 * self._corridor_value(move, player)
                + 0.6 * self._goal_edge_bonus(move, player)
                + 0.35 * self._near_goal_edge_bonus(move, player)
                + 0.2 * self._axis_alignment(move, player)
            )
            if move == tt_move:
                score += 250.0
            if move in winning:
                score += 1200.0
            if move in blocking:
                score += 320.0
            if last_move is not None and move in get_neighbors(last_move[0], last_move[1], self._size):
                score += 0.75
            scored.append((score, move))

        scored.sort(reverse=True)
        return [move for _, move in scored]

    def _flat_root_search(
        self,
        entries: list[dict],
        force_block: bool,
    ) -> tuple[int, int]:
        entries = list(entries)
        count = max(1, len(entries) - 1)
        for index, entry in enumerate(entries):
            entry["prior"] = 1.0 - (index / count if count else 0.0)

        reserve = max(0.06, self._time_limit * 0.03)
        total_visits = 0

        for entry in entries:
            if self._time_left() <= reserve:
                break
            result = self._root_rollout_value(entry)
            entry["visits"] = 1
            entry["value"] = result
            total_visits += 1

        while self._time_left() > reserve:
            entry = max(
                entries,
                key=lambda item: self._root_selection_score(item, total_visits, force_block),
            )
            result = self._root_rollout_value(entry)
            entry["visits"] += 1
            entry["value"] += result
            total_visits += 1

        def final_score(entry: dict) -> float:
            mean = entry["value"] / entry["visits"] if entry["visits"] else entry["heur_value"]
            reliability = 0.015 * min(entry["visits"], 12)
            block = 0.028 * max(0.0, entry["block_delta"])
            build = 0.03 * max(0.0, entry["build_delta"])
            if force_block:
                block *= 1.35
            return mean + reliability + 0.03 * entry["prior"] + block + build + 0.055 * entry["reply_value"] - 0.03 * entry["reply_pressure"]

        return max(entries, key=final_score)["move"]

    def _root_selection_score(
        self,
        entry: dict,
        total_visits: int,
        force_block: bool,
    ) -> float:
        if entry["visits"] == 0:
            bonus = 0.05 * entry["prior"]
            bonus += 0.05 * max(0.0, entry["block_delta"]) if force_block else 0.03 * max(0.0, entry["build_delta"])
            bonus += 0.03 * entry["reply_value"] - 0.02 * entry["reply_pressure"]
            return 10.0 + bonus

        mean = entry["value"] / entry["visits"]
        explore = 0.92 * math.sqrt(math.log(total_visits + 1.0) / entry["visits"])
        bonus = 0.04 * entry["prior"] + 0.035 * entry["heur_value"] + 0.03 * entry["reply_value"] - 0.018 * entry["reply_pressure"]
        if force_block:
            bonus += 0.04 * max(0.0, entry["block_delta"])
        else:
            bonus += 0.02 * max(0.0, entry["block_delta"]) + 0.025 * max(0.0, entry["build_delta"])
        return mean + explore + bonus / (1.0 + 0.15 * entry["visits"])

    def _root_rollout_value(
        self,
        entry: dict,
    ) -> float:
        rollout = self._rollout_value(entry["board"], 3 - self._player, entry["move"])
        return 0.72 * rollout + 0.28 * entry["heur_value"]

    def _rollout_value(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        last_move: tuple[int, int] | None,
    ) -> float:
        winner = check_winner(board, self._size)
        if winner != 0:
            return 1.0 if winner == self._player else 0.0

        work = [list(row) for row in board]
        empties = [
            (r, c)
            for r in range(self._size)
            for c in range(self._size)
            if work[r][c] == 0
        ]
        current = player
        last = last_move
        reserve = max(0.03, self._time_limit * 0.01)
        max_steps = min(16, len(empties))

        while empties and max_steps > 0 and self._time_left() > reserve:
            idx = self._rollout_pick_index(work, empties, current, last)
            move = empties[idx]
            empties[idx] = empties[-1]
            empties.pop()
            work[move[0]][move[1]] = current
            winner = check_winner(work, self._size)
            if winner != 0:
                return 1.0 if winner == self._player else 0.0
            last = move
            current = 3 - current
            max_steps -= 1

        frozen = tuple(tuple(row) for row in work)
        return self._rollout_heuristic_value(frozen)

    def _rollout_pick_index(
        self,
        board: list[list[int]],
        empties: list[tuple[int, int]],
        player: int,
        last_move: tuple[int, int] | None,
    ) -> int:
        sample_count = min(14, len(empties))
        sample_indices = set(self._rng.sample(range(len(empties)), sample_count))
        if last_move is not None and len(sample_indices) < len(empties):
            neighbors = set(get_neighbors(last_move[0], last_move[1], self._size))
            added = 0
            for idx, move in enumerate(empties):
                if move in neighbors:
                    sample_indices.add(idx)
                    added += 1
                    if added >= 4:
                        break
        sample_list = list(sample_indices)
        best_idx = sample_list[0]
        best_score = -10**18
        for idx in sample_list:
            move = empties[idx]
            score = self._rollout_bias_score(board, move, player, last_move)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _rollout_bias_score(
        self,
        board: list[list[int]],
        move: tuple[int, int],
        player: int,
        last_move: tuple[int, int] | None,
    ) -> float:
        own_adj = self._count_adjacent(board, move, player)
        opp_adj = self._count_adjacent(board, move, 3 - player)
        own_merge = self._group_merge_potential_tupleish(board, move, player)
        opp_merge = self._group_merge_potential_tupleish(board, move, 3 - player)
        bridge = self._bridge_move_bonus_tupleish(board, move, player)
        score = (
            2.5 * own_adj
            + 1.4 * opp_adj
            + 1.6 * own_merge
            + 0.8 * opp_merge
            + 1.0 * bridge
            + 1.2 * self._corridor_value(move, player)
            + 0.6 * self._near_goal_edge_bonus(move, player)
            + 0.35 * self._goal_edge_bonus(move, player)
        )
        if own_adj > 0 and opp_adj > 0:
            score += 1.7
        if last_move is not None and move in get_neighbors(last_move[0], last_move[1], self._size):
            score += 0.8
        return score

    def _rollout_heuristic_value(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> float:
        winner = check_winner(board, self._size)
        if winner == self._player:
            return 1.0
        if winner == self._opponent:
            return 0.0

        my_dist = shortest_path_distance(board, self._size, self._player)
        opp_dist = shortest_path_distance(board, self._size, self._opponent)
        span_delta = self._span(board, self._player) - self._span(board, self._opponent)
        score = 0.5 + 0.11 * (opp_dist - my_dist) + 0.018 * span_delta
        return self._clamp01(score)

    def _opponent_reply_signal(
        self,
        board: tuple[tuple[int, ...], ...],
        last_move: tuple[int, int],
    ) -> tuple[float, float]:
        opp_ctx = self._critical_context(board, self._opponent)
        my_ctx = self._critical_context(board, self._player)
        empties = empty_cells(board, self._size)
        if not empties:
            return self._normalized_state_value(my_ctx, opp_ctx), 0.0

        wins = self._find_immediate_wins(board, self._opponent, empties)
        if wins:
            return 0.0, 10.0

        reply_moves = [move for move, _ in opp_ctx["top"][:10]]
        reply_moves.extend(move for move, _ in my_ctx["top"][:8])
        cut_scored = []
        for move in empties:
            cut_gain = self._block_distance_delta(board, move, self._player, my_ctx["best"])
            if cut_gain <= 0 and my_ctx["critical"].get(move, 0.0) <= 0.0:
                continue
            cut_scored.append((2.0 * cut_gain + my_ctx["critical"].get(move, 0.0), move))
        cut_scored.sort(reverse=True)
        reply_moves.extend(move for _, move in cut_scored[:8])
        reply_moves.extend(
            move
            for move in get_neighbors(last_move[0], last_move[1], self._size)
            if board[move[0]][move[1]] == 0
        )
        reply_moves = self._dedupe_moves(reply_moves)

        worst_value = 1.0
        max_pressure = 0.0
        for move in reply_moves[:12]:
            if self._time_left() < max(0.08, self._time_limit * 0.02):
                break
            reply = self._with_move(board, move, self._opponent)
            winner = check_winner(reply, self._size)
            if winner == self._opponent:
                return 0.0, 10.0

            reply_my = self._critical_context(reply, self._player)
            reply_opp = self._critical_context(reply, self._opponent)
            value = self._normalized_state_value(reply_my, reply_opp)
            pressure = max(
                0.0,
                6.0 - float(reply_opp["best"]),
            )
            pressure += max(0.0, float(reply_my["best"] - my_ctx["best"]))

            if value < worst_value:
                worst_value = value
            if pressure > max_pressure:
                max_pressure = pressure

        return worst_value, max_pressure

    # ---------------------------------------------------------------
    # Dark mode
    # ---------------------------------------------------------------

    def _play_dark(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> tuple[int, int]:
        empties = empty_cells(board, self._size)
        if len(empties) == 1:
            return empties[0]

        opening = self._opening_move(board)
        if opening is not None:
            return opening

        winning = self._find_immediate_wins(board, self._player, empties)
        if winning:
            return self._pick_best(board, winning)

        my_ctx = self._critical_context(board, self._player)
        opp_ctx = self._critical_context(board, self._opponent)

        scored: list[tuple[float, tuple[int, int]]] = []
        for move in empties:
            score = (
                2.2 * my_ctx["critical"].get(move, 0.0)
                + 2.7 * opp_ctx["critical"].get(move, 0.0)
                + 1.2 * self._dark_information_bonus(board, move)
                - 18.0 * self._collision_risk(board, move)
                + 1.2 * self._count_adjacent(board, move, self._player)
            )
            scored.append((score, move))

        scored.sort(reverse=True)
        return scored[0][1]

    # ---------------------------------------------------------------
    # Critical corridors
    # ---------------------------------------------------------------

    def _critical_context(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
    ) -> dict:
        key = (board, player)
        cached = self._ctx_cache.get(key)
        if cached is not None:
            return cached

        start_dist = self._dijkstra(board, player, reverse=False)
        goal_dist = self._dijkstra(board, player, reverse=True)
        best = self._distance(board, player)

        shortest_from_start = self._shortest_path_counts(board, player, start_dist, reverse=False)
        shortest_from_goal = self._shortest_path_counts(board, player, goal_dist, reverse=True)

        critical: dict[tuple[int, int], float] = {}
        INF = self._size * self._size + 5

        for move in empty_cells(board, self._size):
            ds = start_dist.get(move, INF)
            dg = goal_dist.get(move, INF)
            total = ds + dg - 1
            gap = total - best
            if gap > 2:
                continue

            if gap == 0:
                multiplicity = shortest_from_start.get(move, 0) * shortest_from_goal.get(move, 0)
                score = 5.0 + min(10.0, float(multiplicity))
            elif gap == 1:
                score = 3.0
            else:
                score = 1.5

            own_adj = self._count_adjacent(board, move, player)
            merge = self._group_merge_potential(board, move, player)
            bridge = self._bridge_move_bonus(board, move, player)

            score += 0.45 * own_adj
            score += 0.3 * bridge
            score += 1.15 * merge
            score += 0.25 * self._axis_alignment(move, player)
            score += 0.9 * self._goal_edge_bonus(move, player)
            score += 0.45 * self._near_goal_edge_bonus(move, player)
            if self._goal_edge_bonus(move, player) > 0 and own_adj == 0 and merge == 0 and gap > 0:
                score -= 1.4
            if self._near_goal_edge_bonus(move, player) > 0 and own_adj == 0 and merge == 0 and gap > 0:
                score -= 0.7
            critical[move] = score

        prelim = sorted(critical.items(), key=lambda item: item[1], reverse=True)
        for move, _ in prelim[:14]:
            critical[move] += 4.0 * self._block_distance_delta(board, move, player, best)

        top = sorted(critical.items(), key=lambda item: item[1], reverse=True)
        ctx = {
            "best": best,
            "critical": critical,
            "top": top,
            "mass": sum(score for _, score in top[:10]),
        }
        self._ctx_cache[key] = ctx
        return ctx

    def _dijkstra(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        reverse: bool,
    ) -> dict[tuple[int, int], int]:
        opponent = 3 - player
        heap: list[tuple[int, int, int]] = []
        dist: dict[tuple[int, int], int] = {}

        for r, c in self._edge_cells(player, reverse):
            cell = board[r][c]
            if cell == opponent:
                continue
            cost = 0 if cell == player else 1
            heapq.heappush(heap, (cost, r, c))

        while heap:
            d, r, c = heapq.heappop(heap)
            if (r, c) in dist:
                continue
            dist[(r, c)] = d
            for nr, nc in get_neighbors(r, c, self._size):
                if (nr, nc) in dist:
                    continue
                cell = board[nr][nc]
                if cell == opponent:
                    continue
                step = 0 if cell == player else 1
                heapq.heappush(heap, (d + step, nr, nc))

        return dist

    def _shortest_path_counts(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        dist: dict[tuple[int, int], int],
        reverse: bool,
    ) -> dict[tuple[int, int], int]:
        starts = set(self._edge_cells(player, reverse))
        counts: dict[tuple[int, int], int] = {}
        ordered = sorted(dist.items(), key=lambda item: item[1])

        for (r, c), d in ordered:
            if (r, c) in starts:
                cell = board[r][c]
                cost = 0 if cell == player else 1
                counts[(r, c)] = 1 if d == cost else 0
                continue

            cell_cost = 0 if board[r][c] == player else 1
            total = 0
            for nr, nc in get_neighbors(r, c, self._size):
                prev = dist.get((nr, nc))
                if prev is None:
                    continue
                if prev + cell_cost == d:
                    total += counts.get((nr, nc), 0)
            counts[(r, c)] = min(total, 64)

        return counts

    def _edge_cells(
        self,
        player: int,
        reverse: bool,
    ) -> list[tuple[int, int]]:
        if player == 1:
            row = self._size - 1 if reverse else 0
            return [(row, c) for c in range(self._size)]
        col = self._size - 1 if reverse else 0
        return [(r, col) for r in range(self._size)]

    # ---------------------------------------------------------------
    # Evaluation helpers
    # ---------------------------------------------------------------

    def _position_score(
        self,
        board: tuple[tuple[int, ...], ...],
        my_ctx: dict,
        opp_ctx: dict,
    ) -> float:
        winner = check_winner(board, self._size)
        if winner == self._player:
            return 100000.0
        if winner == self._opponent:
            return -100000.0

        score = 14.0 * (opp_ctx["best"] - my_ctx["best"])
        score += 1.4 * (self._top_score(my_ctx) - self._top_score(opp_ctx))
        score += 0.8 * (self._span(board, self._player) - self._span(board, self._opponent))
        return score

    def _top_score(self, ctx: dict) -> float:
        return sum(score for _, score in ctx["top"][:6])

    def _normalized_state_value(
        self,
        my_ctx: dict,
        opp_ctx: dict,
    ) -> float:
        score = 0.5 + 0.1 * (opp_ctx["best"] - my_ctx["best"])
        score += 0.006 * (self._top_score(my_ctx) - self._top_score(opp_ctx))
        return self._clamp01(score)

    def _distance(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
    ) -> int:
        return shortest_path_distance(board, self._size, player)

    def _span(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
    ) -> float:
        coords = [
            (r, c)
            for r in range(self._size)
            for c in range(self._size)
            if board[r][c] == player
        ]
        if not coords:
            return 0.0
        if player == 1:
            axis = [r for r, _ in coords]
        else:
            axis = [c for _, c in coords]
        return float(max(axis) - min(axis))

    def _cross_span(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        player: int,
    ) -> float:
        coords = [
            (r, c)
            for r in range(self._size)
            for c in range(self._size)
            if board[r][c] == player
        ]
        if not coords:
            return 0.0
        if player == 1:
            axis = [c for _, c in coords]
        else:
            axis = [r for r, _ in coords]
        return float(max(axis) - min(axis))

    def _shape_profile(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        player: int,
    ) -> dict:
        progress_vals = []
        cross_vals = []
        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] != player:
                    continue
                progress_vals.append(r if player == 1 else c)
                cross_vals.append(c if player == 1 else r)
        if not progress_vals:
            return {
                "stones": 0,
                "progress_low": None,
                "progress_high": None,
                "progress_span": 0.0,
                "cross_span": 0.0,
            }
        return {
            "stones": len(progress_vals),
            "progress_low": min(progress_vals),
            "progress_high": max(progress_vals),
            "progress_span": float(max(progress_vals) - min(progress_vals)),
            "cross_span": float(max(cross_vals) - min(cross_vals)),
        }

    def _objective_span_extension_from_profile(
        self,
        profile: dict,
        move: tuple[int, int],
        player: int,
    ) -> float:
        if profile["stones"] <= 0:
            return 0.0
        axis = move[0] if player == 1 else move[1]
        low = profile["progress_low"]
        high = profile["progress_high"]
        if low is None or high is None:
            return 0.0
        if axis < low or axis > high:
            return 1.2
        if axis == low or axis == high:
            return 0.5
        return 0.0

    def _white_overcommit_pressure(
        self,
        profile: dict | None,
    ) -> float:
        if profile is None or profile["stones"] < 5:
            return 0.0
        if profile["progress_span"] >= 6.0 or profile["cross_span"] > 2.0:
            return 0.0
        return 0.6 + 0.18 * (profile["stones"] - 4) + 0.15 * max(0.0, 2.0 - profile["cross_span"])

    def _conversion_context(
        self,
        my_ctx: dict,
        opp_ctx: dict,
        my_profile: dict | None,
        opp_corridor: dict | None = None,
    ) -> dict:
        if my_profile is None:
            return {"active": False, "strength": 0.0}

        lead = float(opp_ctx["best"] - my_ctx["best"])
        if lead < 2.0 or my_ctx["best"] > 4 or opp_ctx["best"] < 7:
            return {"active": False, "strength": 0.0}
        if my_profile["stones"] < 7 or my_profile["progress_span"] < 5.0:
            return {"active": False, "strength": 0.0}
        if opp_ctx["mass"] > my_ctx["mass"] + 4.0:
            return {"active": False, "strength": 0.0}
        if opp_corridor is not None and opp_corridor["threat"] >= 4.5:
            return {"active": False, "strength": 0.0}

        strength = 0.32 + 0.08 * min(3.0, lead - 1.0)
        if my_ctx["best"] <= 3:
            strength += 0.08
        if my_profile["progress_span"] >= 6.0:
            strength += 0.06
        return {"active": True, "strength": min(0.7, strength)}

    def _narrow_corridor_pressure(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        player: int,
    ) -> dict:
        seen: set[tuple[int, int]] = set()
        best = {
            "threat": 0.0,
            "frontier": set(),
            "progress_span": 0.0,
            "cross_span": float(self._size),
        }

        for r in range(self._size):
            for c in range(self._size):
                if board[r][c] != player or (r, c) in seen:
                    continue
                component = self._collect_component(board, (r, c), player, seen)
                if len(component) < 4:
                    continue

                progress_vals = [rr if player == 1 else cc for rr, cc in component]
                cross_vals = [cc if player == 1 else rr for rr, cc in component]
                progress_low = min(progress_vals)
                progress_high = max(progress_vals)
                cross_low = min(cross_vals)
                cross_high = max(cross_vals)
                progress_span = float(progress_high - progress_low)
                cross_span = float(cross_high - cross_low)
                edge_touch = 0
                if progress_low <= 1:
                    edge_touch += 1
                if progress_high >= self._size - 2:
                    edge_touch += 1

                threat = 1.15 * progress_span + 0.5 * len(component) - 0.95 * cross_span + 0.85 * edge_touch
                if cross_span <= 3.0:
                    threat += 1.2
                if len(component) >= 6 and cross_span <= 3.0:
                    threat += 0.6
                if progress_span < 3.0:
                    threat *= 0.45

                if threat <= best["threat"]:
                    continue

                frontier = set()
                for rr, cc in component:
                    for nr, nc in get_neighbors(rr, cc, self._size):
                        if board[nr][nc] == 0:
                            frontier.add((nr, nc))
                for rr in range(max(0, progress_low - 1), min(self._size, progress_high + 2)):
                    for cc in range(max(0, cross_low - 1), min(self._size, cross_high + 2)):
                        nr, nc = (rr, cc) if player == 1 else (cc, rr)
                        if board[nr][nc] == 0:
                            frontier.add((nr, nc))

                best = {
                    "threat": threat,
                    "frontier": frontier,
                    "progress_span": progress_span,
                    "cross_span": cross_span,
                }

        return best

    def _collect_component(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        start: tuple[int, int],
        player: int,
        seen: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for nr, nc in get_neighbors(r, c, self._size):
                if (nr, nc) in seen or board[nr][nc] != player:
                    continue
                seen.add((nr, nc))
                stack.append((nr, nc))
        return component

    def _white_midgame_bonus(
        self,
        move: tuple[int, int],
        opp_corridor: dict | None,
        opp_critical: float,
        opp_adj: int,
        opp_merge: int,
        pure_cut: float,
    ) -> float:
        if opp_corridor is None or opp_corridor["threat"] < 4.5:
            return 0.0
        threat = opp_corridor["threat"]
        bonus = 0.0
        if move in opp_corridor["frontier"]:
            bonus += 0.8 + 0.22 * threat
        bonus += 0.14 * threat * min(5.0, opp_critical)
        bonus += 0.42 * min(2, opp_adj)
        bonus += 0.25 * opp_merge
        bonus += 0.18 * threat if pure_cut > 0.0 else 0.0
        bonus += 0.45 * pure_cut
        return bonus

    def _find_immediate_wins(
        self,
        board: tuple[tuple[int, ...], ...],
        player: int,
        moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        wins: list[tuple[int, int]] = []
        for move in moves:
            if self._out_of_time():
                break
            trial = self._with_move(board, move, player)
            if check_winner(trial, self._size) == player:
                wins.append(move)
        return wins

    def _pick_best(
        self,
        board: tuple[tuple[int, ...], ...],
        moves: list[tuple[int, int]],
    ) -> tuple[int, int]:
        my_ctx = self._critical_context(board, self._player)
        opp_ctx = self._critical_context(board, self._opponent)
        scored = []
        for move in moves:
            trial = self._with_move(board, move, self._player)
            after_my = self._critical_context(trial, self._player)
            after_opp = self._critical_context(trial, self._opponent)
            score = self._position_score(trial, after_my, after_opp)
            score += 6.0 * (after_opp["best"] - opp_ctx["best"])
            score += 5.0 * (my_ctx["best"] - after_my["best"])
            scored.append((score, move))
        scored.sort(reverse=True)
        return scored[0][1]

    # ---------------------------------------------------------------
    # Dark helpers
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

        visible_opp = sum(cell == self._opponent for row in board for cell in row)
        hidden_estimate = max(0, self._estimated_opponent_turns() - visible_opp)
        density = min(0.9, hidden_estimate / float(len(empties)))
        opp_adj = self._count_adjacent(board, move, self._opponent)
        own_adj = self._count_adjacent(board, move, self._player)
        corridor = self._corridor_value(move, self._opponent)
        local = 0.24 * opp_adj + 0.08 * max(0, 2 - own_adj)
        return max(0.0, min(1.0, density * (0.55 + corridor) + local))

    def _estimated_opponent_turns(self) -> int:
        if self._player == 1:
            return max(0, self._turn_count - 1)
        return self._turn_count

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
        return (
            0.7 * unknown_neighbors
            + 0.9 * own_neighbors
            + 1.0 * self._axis_alignment(move, self._player)
        )

    # ---------------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------------

    def _opening_move(
        self,
        board: tuple[tuple[int, ...], ...],
    ) -> tuple[int, int] | None:
        stones = sum(cell != 0 for row in board for cell in row)
        if stones > 2:
            return None
        for move in self._opening:
            if board[move[0]][move[1]] == 0:
                return move
        return None

    def _build_opening_book(self) -> list[tuple[int, int]]:
        c = self._size // 2
        if self._player == 1:
            moves = [
                (c, c),
                (c - 1, c),
                (c + 1, c),
                (c - 1, c + 1),
                (c + 1, c - 1),
                (c - 2, c),
                (c + 2, c),
                (c, c - 1),
                (c, c + 1),
            ]
        else:
            moves = [
                (c, c),
                (c, c - 1),
                (c, c + 1),
                (c + 1, c - 1),
                (c - 1, c + 1),
                (c, c - 2),
                (c, c + 2),
                (c - 1, c),
                (c + 1, c),
            ]
        return [
            move
            for move in moves
            if 0 <= move[0] < self._size and 0 <= move[1] < self._size
        ]

    def _axis_alignment(self, move: tuple[int, int], player: int) -> float:
        r, c = move
        center = (self._size - 1) / 2.0
        return -abs(c - center) if player == 1 else -abs(r - center)

    def _goal_edge_bonus(self, move: tuple[int, int], player: int) -> float:
        r, c = move
        if player == 1:
            return 1.0 if r == 0 or r == self._size - 1 else 0.0
        return 1.0 if c == 0 or c == self._size - 1 else 0.0

    def _near_goal_edge_bonus(self, move: tuple[int, int], player: int) -> float:
        r, c = move
        if player == 1:
            return 1.0 if r == 1 or r == self._size - 2 else 0.0
        return 1.0 if c == 1 or c == self._size - 2 else 0.0

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

    def _group_merge_potential(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> int:
        groups = []
        seen_roots: set[tuple[int, int]] = set()
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            if board[nr][nc] != player:
                continue
            root = self._component_root(board, (nr, nc), player)
            if root not in seen_roots:
                seen_roots.add(root)
                groups.append(root)
        return max(0, len(groups) - 1)

    def _component_root(
        self,
        board: tuple[tuple[int, ...], ...],
        start: tuple[int, int],
        player: int,
    ) -> tuple[int, int]:
        stack = [start]
        seen = {start}
        best = start
        while stack:
            r, c = stack.pop()
            if (r, c) < best:
                best = (r, c)
            for nr, nc in get_neighbors(r, c, self._size):
                if (nr, nc) in seen or board[nr][nc] != player:
                    continue
                seen.add((nr, nc))
                stack.append((nr, nc))
        return best

    def _bridge_move_bonus(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> float:
        bonus = 0.0
        for rr, cc in self._ring_two(move):
            if board[rr][cc] == player:
                bonus += 0.7
        return bonus

    def _dual_purpose_bonus(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        my_ctx: dict,
        opp_ctx: dict,
    ) -> float:
        own_adj = self._count_adjacent(board, move, self._player)
        own_merge = self._group_merge_potential(board, move, self._player)
        own_critical = my_ctx["critical"].get(move, 0.0)
        opp_critical = opp_ctx["critical"].get(move, 0.0)
        return (
            0.9 * own_adj
            + 1.4 * own_merge
            + 0.7 * own_critical
            + 0.5 * opp_critical
        )

    def _pure_cut_score(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        opp_ctx: dict,
        cut_gain: int,
    ) -> float:
        opp_adj = self._count_adjacent(board, move, self._opponent)
        opp_merge = self._group_merge_potential(board, move, self._opponent)
        return (
            2.2 * opp_ctx["critical"].get(move, 0.0)
            + 2.8 * cut_gain
            + 1.3 * opp_adj
            + 1.0 * opp_merge
            + 0.9 * self._goal_edge_bonus(move, self._opponent)
            + 0.55 * self._near_goal_edge_bonus(move, self._opponent)
        )

    def _corridor_value(self, move: tuple[int, int], player: int) -> float:
        r, c = move
        center = (self._size - 1) / 2.0
        if player == 1:
            axis = abs(c - center)
            edge_balance = min(r, self._size - 1 - r)
        else:
            axis = abs(r - center)
            edge_balance = min(c, self._size - 1 - c)
        return 1.0 / (1.0 + 0.6 * axis + 0.12 * edge_balance)

    def _ring_two(self, move: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = move
        offsets = (
            (-2, 1), (-1, -1), (-1, 2),
            (1, -2), (1, 1), (2, -1),
        )
        result = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self._size and 0 <= nc < self._size:
                result.append((nr, nc))
        return result

    def _with_move(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
    ) -> tuple[tuple[int, ...], ...]:
        rows = [list(row) for row in board]
        rows[move[0]][move[1]] = player
        return tuple(tuple(row) for row in rows)

    def _block_distance_delta(
        self,
        board: tuple[tuple[int, ...], ...],
        move: tuple[int, int],
        player: int,
        best: int,
    ) -> int:
        if board[move[0]][move[1]] != 0:
            return 0
        blocked = self._with_move(board, move, 3 - player)
        after = self._distance(blocked, player)
        if after > self._size * self._size:
            return self._size
        return max(0, after - best)

    def _dedupe_moves(
        self,
        moves: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        result = []
        for move in moves:
            if move in seen:
                continue
            seen.add(move)
            result.append(move)
        return result

    def _group_merge_potential_tupleish(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        move: tuple[int, int],
        player: int,
    ) -> int:
        groups = []
        seen_roots: set[tuple[int, int]] = set()
        for nr, nc in get_neighbors(move[0], move[1], self._size):
            if board[nr][nc] != player:
                continue
            root = self._component_root(board, (nr, nc), player)
            if root not in seen_roots:
                seen_roots.add(root)
                groups.append(root)
        return max(0, len(groups) - 1)

    def _bridge_move_bonus_tupleish(
        self,
        board: tuple[tuple[int, ...], ...] | list[list[int]],
        move: tuple[int, int],
        player: int,
    ) -> float:
        bonus = 0.0
        for rr, cc in self._ring_two(move):
            if board[rr][cc] == player:
                bonus += 0.7
        return bonus

    def _time_left(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def _clamp01(self, value: float) -> float:
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    def _out_of_time(self) -> bool:
        return time.monotonic() >= self._deadline
