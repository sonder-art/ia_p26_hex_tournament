from __future__ import annotations

import time
import math
import random
from collections import defaultdict
from strategy import Strategy, GameConfig
from hex_game import empty_cells, shortest_path_distance, check_winner


class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.rave_visits = 0
        self.rave_wins = 0
        moves = empty_cells(board, len(board))
        random.shuffle(moves)
        self.untried_moves = moves


class EstrategiaPAN(Strategy):

    @property
    def name(self) -> str:
        return "EstrategiaPAN_PAN"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit
        self._adj_cache = {}

    def on_move_result(self, move, success):
        pass

    def _neighbors(self, r, c):
        key = (r, c)
        if key not in self._adj_cache:
            n = self._size
            self._adj_cache[key] = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1))
                if 0 <= r + dr < n and 0 <= c + dc < n
            ]
        return self._adj_cache[key]

    def evaluate(self, board):
        my_dist = shortest_path_distance(board, self._size, self._player)
        opp_dist = shortest_path_distance(board, self._size, self._opponent)
        return my_dist - opp_dist

    def _relevant_moves(self, board, all_moves):
        if not all_moves:
            return all_moves
        relevant = []
        for r, c in all_moves:
            for nr, nc in self._neighbors(r, c):
                if board[nr][nc] != 0:
                    relevant.append((r, c))
                    break
        if len(relevant) < max(4, len(all_moves) // 4):
            return all_moves
        return relevant

    def _bridge_responses(self, board, last_opp_move, all_moves_set):
        """
        Detecta si el último movimiento oponente amenaza un puente propio
        y devuelve la celda de respuesta para mantenerlo.
        Un puente virtual: dos fichas propias separadas por dos celdas
        vacías mutuamente adyacentes. Si el oponente entra en una,
        responder en la otra preserva la conexión.
        Costo: O(6 * 6) = O(1) por llamada.
        """
        if last_opp_move is None:
            return []
        r0, c0 = last_opp_move
        responses = []
        for nr, nc in self._neighbors(r0, c0):
            if (nr, nc) not in all_moves_set:
                continue
            # (nr,nc) vacía adyacente al movimiento oponente
            # ¿hay una ficha propia adyacente a (nr,nc) distinta de (r0,c0)?
            for pr, pc in self._neighbors(nr, nc):
                if (pr, pc) != (r0, c0) and board[pr][pc] == self._player:
                    responses.append((nr, nc))
                    break
        return responses

    def find_winning_move(self, board):
        for r, c in empty_cells(board, self._size):
            b = [list(row) for row in board]
            b[r][c] = self._player
            if check_winner(tuple(tuple(row) for row in b), self._size) == self._player:
                return (r, c)
        return None

    def find_block_move(self, board):
        for r, c in empty_cells(board, self._size):
            b = [list(row) for row in board]
            b[r][c] = self._opponent
            if check_winner(tuple(tuple(row) for row in b), self._size) == self._opponent:
                return (r, c)
        return None

    def determinize(self, board):
        b = [list(row) for row in board]
        empty = [(r, c) for r in range(self._size) for c in range(self._size) if b[r][c] == 0]
        k = min(7, max(1, len(empty) // 9))
        chosen = random.sample(empty, k)
        for r, c in chosen:
            b[r][c] = self._opponent
        return tuple(tuple(row) for row in b)

    # -----------------------------------------------
    # SELECCIÓN: UCT+RAVE para classic, UCT para dark
    # RAVE mezcla estadística de árbol con estadística
    # acumulada de rollouts. beta→0 al crecer visits,
    # así UCT domina cuando hay buena estadística real.
    # -----------------------------------------------
    def select_rave(self, node):
        best = None
        best_score = -float("inf")
        for child in node.children:
            if child.visits == 0:
                return child
            uct = (child.wins / child.visits
                   + 1.2 * math.sqrt(math.log(node.visits) / child.visits))
            if child.rave_visits > 0:
                rave_val = child.rave_wins / child.rave_visits
                beta = child.rave_visits / (child.rave_visits + child.visits + 1e-5)
                score = (1 - beta) * uct + beta * rave_val
            else:
                score = uct
            if score > best_score:
                best_score = score
                best = child
        return best

    def select_uct(self, node):
        best = None
        best_uct = -float("inf")
        for child in node.children:
            if child.visits == 0:
                return child
            uct = (child.wins / child.visits
                   + 1.2 * math.sqrt(math.log(node.visits) / child.visits))
            if uct > best_uct:
                best_uct = uct
                best = child
        return best

    def expand(self, node):
        move = node.untried_moves.pop()
        r, c = move
        b = [list(row) for row in node.board]
        b[r][c] = node.player
        new_board = tuple(tuple(row) for row in b)
        child = Node(new_board, 3 - node.player, parent=node, move=move)
        node.children.append(child)
        return child

    # -----------------------------------------------
    # ROLLOUT CLASSIC: incluye detección de puentes
    # Devuelve (winner, moves_played) para RAVE update.
    # -----------------------------------------------
    def rollout_classic(self, board, player):
        current_player = player
        moves_played = []
        last_move = None

        for _ in range(30):
            winner = check_winner(board, self._size)
            if winner != 0:
                return winner, moves_played

            all_moves = empty_cells(board, self._size)
            if not all_moves:
                return 0, moves_played

            move = None

            # Respuesta de puente: solo cuando es nuestro turno y el oponente acaba de jugar
            if last_move is not None and current_player == self._player:
                all_set = set(all_moves)
                bridges = self._bridge_responses(board, last_move, all_set)
                if bridges:
                    move = random.choice(bridges)

            if move is None:
                pool = self._relevant_moves(board, all_moves) if random.random() < 0.60 else all_moves
                move = random.choice(pool)

            r, c = move
            b = [list(row) for row in board]
            b[r][c] = current_player
            board = tuple(tuple(row) for row in b)
            moves_played.append((move, current_player))
            last_move = move
            current_player = 3 - current_player

        winner = self._player if self.evaluate(board) < 0 else self._opponent
        return winner, moves_played

    # Rollout simple para dark — sin overhead extra
    def rollout_dark(self, board, player):
        current_player = player
        for _ in range(30):
            winner = check_winner(board, self._size)
            if winner != 0:
                return winner
            all_moves = empty_cells(board, self._size)
            if not all_moves:
                return 0
            pool = self._relevant_moves(board, all_moves) if random.random() < 0.60 else all_moves
            r, c = random.choice(pool)
            b = [list(row) for row in board]
            b[r][c] = current_player
            board = tuple(tuple(row) for row in b)
            current_player = 3 - current_player
        return self._player if self.evaluate(board) < 0 else self._opponent

    # -----------------------------------------------
    # BACKPROP CLASSIC: actualiza árbol + RAVE
    # Para cada nodo en el camino, marca como RAVE
    # todos los hijos cuyo movimiento apareció en el
    # rollout — distinguiendo si ganó o no.
    # -----------------------------------------------
    def backpropagate_rave(self, node, winner, moves_played):
        winning_moves = {m for m, p in moves_played if p == winner}
        all_rollout_moves = {m for m, _ in moves_played}

        current = node
        while current is not None:
            current.visits += 1
            if winner == self._player:
                current.wins += 1
            for child in current.children:
                if child.move in all_rollout_moves:
                    child.rave_visits += 1
                    if child.move in winning_moves:
                        child.rave_wins += 1
            current = current.parent

    def backpropagate_simple(self, node, winner):
        while node is not None:
            node.visits += 1
            if winner == self._player:
                node.wins += 1
            node = node.parent

    def _run_mcts_classic(self, board, time_limit):
        root = Node(board, self._player)
        start = time.monotonic()
        while time.monotonic() - start < time_limit:
            node = root
            while node.children and not node.untried_moves:
                node = self.select_rave(node)
            if node.untried_moves:
                node = self.expand(node)
            winner, moves_played = self.rollout_classic(node.board, node.player)
            self.backpropagate_rave(node, winner, moves_played)
        return {child.move: child.visits for child in root.children}

    def _run_mcts_dark(self, board, time_limit):
        root = Node(board, self._player)
        start = time.monotonic()
        while time.monotonic() - start < time_limit:
            node = root
            while node.children and not node.untried_moves:
                node = self.select_uct(node)
            if node.untried_moves:
                node = self.expand(node)
            winner = self.rollout_dark(node.board, node.player)
            self.backpropagate_simple(node, winner)
        return {child.move: child.visits for child in root.children}

    def _opening_move(self, moves):
        n = self._size
        mid = n // 2
        priority = [
            (mid, mid),
            (mid - 1, mid + 1), (mid + 1, mid - 1),
            (mid - 1, mid),     (mid + 1, mid),
            (mid, mid - 1),     (mid, mid + 1),
            (mid - 2, mid + 2), (mid + 2, mid - 2),
        ]
        moves_set = set(moves)
        for pos in priority:
            if pos in moves_set:
                return pos
        return None

    def play(self, board, last_move):

        # 1. Ganar
        win = self.find_winning_move(board)
        if win:
            return win

        # 2. Bloquear
        block = self.find_block_move(board)
        if block:
            return block

        moves = empty_cells(board, self._size)
        occupied = self._size * self._size - len(moves)

        # 3. Apertura
        if occupied <= 2:
            op = self._opening_move(moves)
            if op:
                return op

        limit = self._time_limit * 0.88

        if self._variant == "dark":
            # MD-MCTS: 4 escenarios independientes, votos agregados
            N = 4
            time_per_tree = limit / N
            vote_counts = defaultdict(int)
            for _ in range(N):
                det_board = self.determinize(board)
                tree_votes = self._run_mcts_dark(det_board, time_per_tree)
                for move, visits in tree_votes.items():
                    vote_counts[move] += visits
            if vote_counts:
                return max(vote_counts, key=vote_counts.get)

        else:
            # Classic: MCTS + RAVE, todo el tiempo disponible
            tree_votes = self._run_mcts_classic(board, limit)
            if tree_votes:
                return max(tree_votes, key=tree_votes.get)

        fallback = self._relevant_moves(board, moves)
        return random.choice(fallback if fallback else moves)
