from __future__ import annotations
import math
import random
import time
from strategy import Strategy, GameConfig
from hex_game import get_neighbors, check_winner, shortest_path_distance, empty_cells

class MCTSNode:
    __slots__ = ['move', 'parent', 'children', 'wins', 'visits', 'untried']
    
    def __init__(self, move=None, parent=None, untried=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = untried or []

    def uct_score(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self):
        return max(self.children, key=lambda n: n.uct_score())

    def most_visited(self):
        return max(self.children, key=lambda n: n.visits)


class ISLStrategy(Strategy):

    @property
    def name(self) -> str:
        return "ISL_strategy"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit
        self._collision_cells = set()

    def on_move_result(self, move, success):
        if not success:
            self._collision_cells.add(move)

    def play(self, board, last_move):
        # Convertir a lista mutable
        board_list = [list(row) for row in board]
        
        # Jugada de apertura: centro
        size = self._size
        center = size // 2
        if board_list[center][center] == 0:
            empty = empty_cells(board, size)
            if len(empty) > size * size - 2:
                return (center, center)

        # Verificar victoria/bloqueo inmediato
        quick = self._quick_win_or_block(board_list)
        if quick:
            return quick

        # MCTS
        return self._mcts(board_list)

    def _quick_win_or_block(self, board):
        """Detecta si hay un movimiento que gana o bloquea victoria rival."""
        size = self._size
        # Checa si podemos ganar en 1 jugada
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    board[r][c] = self._player
                    if check_winner(board, size) == self._player:
                        board[r][c] = 0
                        return (r, c)
                    board[r][c] = 0
        # Checa si el oponente gana en 1 jugada
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    board[r][c] = self._opponent
                    if check_winner(board, size) == self._opponent:
                        board[r][c] = 0
                        return (r, c)
                    board[r][c] = 0
        return None

    def _mcts(self, board):
        size = self._size
        player = self._player
        moves = empty_cells(board, size)
        
        if not moves:
            return (0, 0)
        if len(moves) == 1:
            return moves[0]

        # Ordenar movimientos por heurística antes de MCTS
        moves = self._sort_moves(board, moves, player)
        
        root = MCTSNode(untried=moves[:])
        t0 = time.monotonic()
        budget = self._time_limit * 0.85
        iterations = 0

        while time.monotonic() - t0 < budget:
            # Clonar tablero
            sim_board = [row[:] for row in board]
            current_player = player
            node = root

            # 1. SELECCIÓN
            while not node.untried and node.children:
                node = node.best_child()
                sim_board[node.move[0]][node.move[1]] = current_player
                current_player = 3 - current_player

            # 2. EXPANSIÓN
            if node.untried:
                move = node.untried.pop(0)
                sim_board[move[0]][move[1]] = current_player
                child_moves = empty_cells(sim_board, size)
                child_moves = self._sort_moves(sim_board, child_moves, 3 - current_player)
                child = MCTSNode(move=move, parent=node, untried=child_moves)
                node.children.append(child)
                node = child
                current_player = 3 - current_player

            # 3. SIMULACIÓN (rollout sesgado)
            result = self._rollout(sim_board, current_player, size)

            # 4. BACKPROPAGACIÓN
            while node is not None:
                node.visits += 1
                # result es 1 si gana self._player
                node.wins += result
                node = node.parent

            iterations += 1

        if not root.children:
            return moves[0]
        
        best = root.most_visited()
        return best.move

    def _sort_moves(self, board, moves, player):
        """Ordena movimientos por heurística Dijkstra (menor distancia = mejor)."""
        opponent = 3 - player
        scored = []
        for r, c in moves:
            board[r][c] = player
            d_self = shortest_path_distance(board, self._size, player)
            d_opp = shortest_path_distance(board, self._size, opponent)
            board[r][c] = 0
            # Mejor movimiento: minimiza propia distancia, maximiza distancia rival
            score = -d_self + 0.5 * d_opp
            scored.append((score, r, c))
        scored.sort(reverse=True)
        return [(r, c) for _, r, c in scored]

    def _rollout(self, board, current_player, size):
        """Simulación con política sesgada hacia mejores movimientos."""
        sim = [row[:] for row in board]
        player = current_player
        max_moves = size * size

        for _ in range(max_moves):
            winner = check_winner(sim, size)
            if winner != 0:
                return 1.0 if winner == self._player else 0.0

            moves = empty_cells(sim, size)
            if not moves:
                break

            # Política sesgada: 70% mejor movimiento, 30% aleatorio
            if random.random() < 0.7 and len(moves) > 1:
                # Toma el mejor de una muestra de 5
                sample = random.sample(moves, min(5, len(moves)))
                best_move = min(sample, key=lambda m: self._dist_after_move(sim, m, player, size))
                move = best_move
            else:
                move = random.choice(moves)

            sim[move[0]][move[1]] = player
            player = 3 - player

        # Si no hubo ganador, usar distancia Dijkstra como evaluación
        d_self = shortest_path_distance(sim, size, self._player)
        d_opp = shortest_path_distance(sim, size, self._opponent)
        if d_self < d_opp:
            return 1.0
        elif d_opp < d_self:
            return 0.0
        return 0.5

    def _dist_after_move(self, board, move, player, size):
        board[move[0]][move[1]] = player
        d = shortest_path_distance(board, size, player)
        board[move[0]][move[1]] = 0
        return d
    