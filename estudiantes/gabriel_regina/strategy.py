"""
Estrategia MCTS + RAVE para Hex 11x11.
Autores: <gabriel_regina>

Version 3: Volvemos a la base que daba 2/10 vs Tier 2,
con UN solo cambio: mejorar la selección del primer movimiento
y el rollout priorizando el camino Dijkstra CORRECTAMENTE.

Cambios respecto a v1 (la que daba 2/10):
  - _path_cells arreglado: solo devuelve celdas que reducen la distancia
  - PATH_P bajo (0.30) para no sobre-sesgar
  - CUTOFF_FILL de vuelta a 0.65 (no cambiar esto fue un error)
  - UCT_C de vuelta a 1.2 (necesitamos exploración con 121 celdas)
  - _expand NO usa path_cells (eso era el bug más grave)
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict

from strategy import Strategy, GameConfig
from hex_game import (
    get_neighbors,
    check_winner,
    shortest_path_distance,
    empty_cells,
)

# ---------------------------------------------------------------------------
# Constantes — conservadoras, solo un cambio real vs v1
# ---------------------------------------------------------------------------
UCT_C        = 1.2      # De vuelta al original
RAVE_K       = 400
RAVE_BLEND   = 0.8
TIME_BUDGET  = 0.85     # Margen seguro en Docker ARM
CUTOFF_FILL  = 0.65     # De vuelta al original — 0.50 era demasiado agresivo
NEIGHBOR_P   = 0.75
PATH_P       = 0.30     # Bajo: solo sesga el rollout levemente hacia el camino


# ---------------------------------------------------------------------------
# Nodo del árbol MCTS
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = (
        "move", "parent", "children", "visits", "wins",
        "rave_visits", "rave_wins", "untried_moves", "player_to_move"
    )

    def __init__(self, move, parent, untried_moves, player_to_move):
        self.move           = move
        self.parent         = parent
        self.children       = []
        self.visits         = 0
        self.wins           = 0.0
        self.rave_visits    = defaultdict(int)
        self.rave_wins      = defaultdict(float)
        self.untried_moves  = untried_moves
        self.player_to_move = player_to_move

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def uct_rave_score(self, parent_visits, my_player):
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = UCT_C * math.sqrt(math.log(parent_visits) / self.visits)
        uct_val = exploit + explore

        rv = self.parent.rave_visits.get(self.move, 0) if self.parent else 0
        if rv > 0:
            rave_val = self.parent.rave_wins.get(self.move, 0) / rv
            beta = math.sqrt(RAVE_K / (3 * self.visits + RAVE_K))
            beta = min(beta, RAVE_BLEND)
            return (1 - beta) * uct_val + beta * rave_val
        return uct_val


# ---------------------------------------------------------------------------
# Utilidades de tablero
# ---------------------------------------------------------------------------

def _board_to_lists(board):
    return [list(row) for row in board]


def _count_filled(board, size):
    total = 0
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                total += 1
    return total


def _find_bridges(board, size, player):
    defend = set()
    pieces = [(r, c) for r in range(size) for c in range(size)
              if board[r][c] == player]
    for i, (r1, c1) in enumerate(pieces):
        n1 = set(get_neighbors(r1, c1, size))
        for r2, c2 in pieces[i+1:]:
            n2 = set(get_neighbors(r2, c2, size))
            shared = [x for x in n1 & n2 if board[x[0]][x[1]] == 0]
            if len(shared) == 2:
                defend.update(shared)
    return defend


def _bridges_threatened(board, size, player, last_opp_move):
    if last_opp_move is None:
        return set()
    bridges = _find_bridges(board, size, player)
    r, c = last_opp_move
    threatened = set()
    opp_neighbors = set(get_neighbors(r, c, size))
    for cell in bridges:
        if cell in opp_neighbors:
            threatened.add(cell)
    return threatened


def _dijkstra_eval(board, size, player):
    my_dist  = shortest_path_distance(board, size, player)
    opp      = 3 - player
    opp_dist = shortest_path_distance(board, size, opp)
    return opp_dist - my_dist


def _dijkstra_path_cells(board, size, player):
    """
    Devuelve celdas vacías que están en algún camino mínimo del jugador.
    Una celda vacía está en el camino mínimo si:
      dist_desde_inicio[r][c] + dist_hasta_fin[r][c] == distancia_total
    
    Esta es la forma correcta de encontrar celdas en el camino Dijkstra.
    Solo funciona bien cuando el jugador ya tiene algunas piezas.
    """
    import heapq
    INF = float("inf")
    opp = 3 - player
    size_sq = size

    def dijkstra_forward(start_cells):
        dist = [[INF] * size_sq for _ in range(size_sq)]
        heap = []
        for (r, c, cost) in start_cells:
            if cost < dist[r][c]:
                dist[r][c] = cost
                heapq.heappush(heap, (cost, r, c))
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            for nr, nc in get_neighbors(r, c, size_sq):
                if board[nr][nc] == opp:
                    continue
                add = 0 if board[nr][nc] == player else 1
                nd = d + add
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
        return dist

    # Forward: desde el borde de inicio
    if player == 1:  # Negro: fila 0 → fila N-1
        fwd_starts = [(0, c, 0 if board[0][c] == player else (1 if board[0][c] == 0 else INF))
                      for c in range(size_sq) if board[0][c] != opp]
        bwd_starts = [(size_sq-1, c, 0 if board[size_sq-1][c] == player else (1 if board[size_sq-1][c] == 0 else INF))
                      for c in range(size_sq) if board[size_sq-1][c] != opp]
    else:  # Blanco: col 0 → col N-1
        fwd_starts = [(r, 0, 0 if board[r][0] == player else (1 if board[r][0] == 0 else INF))
                      for r in range(size_sq) if board[r][0] != opp]
        bwd_starts = [(r, size_sq-1, 0 if board[r][size_sq-1] == player else (1 if board[r][size_sq-1] == 0 else INF))
                      for r in range(size_sq) if board[r][size_sq-1] != opp]

    fwd = dijkstra_forward(fwd_starts)
    bwd = dijkstra_forward(bwd_starts)

    # Distancia total del camino más corto
    if player == 1:
        total = min(fwd[size_sq-1][c] for c in range(size_sq))
    else:
        total = min(fwd[r][size_sq-1] for r in range(size_sq))

    # Count player pieces — if none, no path bias
    has_pieces = any(board[r][c] == player for r in range(size_sq) for c in range(size_sq))
    if total == INF or not has_pieces:
        return []

    # Celdas vacías en algún camino mínimo
    on_path = []
    for r in range(size_sq):
        for c in range(size_sq):
            if board[r][c] == 0:
                if fwd[r][c] != INF and bwd[r][c] != INF:
                    if fwd[r][c] + bwd[r][c] == total + 1:
                        on_path.append((r, c))
    return on_path


# ---------------------------------------------------------------------------
# Rollout informado — Dijkstras pre-calculados, 0 Dijkstras en el loop MCTS
# ---------------------------------------------------------------------------

def _seed_last_moves(board, size):
    """Pieza más central de cada jugador como semilla de NEIGHBOR_P."""
    center = (size - 1) / 2.0
    result = {1: None, 2: None}
    for player in [1, 2]:
        best, best_d = None, float("inf")
        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    d = abs(r - center) + abs(c - center)
                    if d < best_d:
                        best_d, best = d, (r, c)
        result[player] = best
    return result


def _informed_rollout(board, size, player_to_move, root_player,
                      path_cells, last_move=None):
    """path_cells: dict {player: set} pre-calculado en play(). 0 Dijkstras aqui."""
    b = _board_to_lists(board)
    current = player_to_move
    total_cells = size * size
    cutoff = int(CUTOFF_FILL * total_cells)
    moves_played = []

    last_by_player = _seed_last_moves(board, size)
    last_opp_move = last_move
    filled = sum(1 for r in range(size) for c in range(size) if b[r][c] != 0)

    for _ in range(total_cells * 2):
        if filled >= cutoff:
            eval_score = _dijkstra_eval(b, size, root_player)
            return (1.0 if eval_score > 0 else 0.0), moves_played

        empties = empty_cells(b, size)
        if not empties:
            break

        my_last = last_by_player[current]
        my_path = path_cells[current]   # set O(1)

        # 1. Bridge defense
        defend = _bridges_threatened(b, size, current, last_opp_move)
        defend = [m for m in defend if b[m[0]][m[1]] == 0]
        if defend:
            chosen = random.choice(defend)

        # 2. Path bias — O(1), sin Dijkstra
        elif my_path and random.random() < PATH_P:
            candidates = [m for m in my_path if b[m[0]][m[1]] == 0]
            if candidates:
                chosen = random.choice(candidates)
            elif my_last is not None:
                r, c = my_last
                nbrs = [n for n in get_neighbors(r, c, size) if b[n[0]][n[1]] == 0]
                chosen = random.choice(nbrs) if nbrs else random.choice(empties)
            else:
                chosen = random.choice(empties)

        # 3. Vecino de MI ultima pieza
        elif my_last is not None and random.random() < NEIGHBOR_P:
            r, c = my_last
            nbrs = [n for n in get_neighbors(r, c, size) if b[n[0]][n[1]] == 0]
            chosen = random.choice(nbrs) if nbrs else random.choice(empties)

        # 4. Aleatorio
        else:
            chosen = random.choice(empties)

        b[chosen[0]][chosen[1]] = current
        moves_played.append((chosen, current))
        last_by_player[current] = chosen
        last_opp_move = chosen
        filled += 1

        winner = check_winner(b, size)
        if winner != 0:
            return (1.0 if winner == root_player else 0.0), moves_played

        current = 3 - current

    eval_score = _dijkstra_eval(b, size, root_player)
    return (1.0 if eval_score > 0 else 0.0), moves_played


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------

class MiEstrategia(Strategy):

    @property
    def name(self) -> str:
        return "gabriel_regina"

    def begin_game(self, config: GameConfig) -> None:
        self._size       = config.board_size
        self._player     = config.player
        self._opponent   = config.opponent
        self._time_limit = config.time_limit
        self._variant    = config.variant
        self._hidden_opp   = set()
        self._my_moves     = set()
        self._failed_moves = set()

    def on_move_result(self, move, success):
        if success:
            self._my_moves.add(move)
        else:
            self._failed_moves.add(move)
            self._hidden_opp.add(move)

    def play(self, board, last_move):
        t0 = time.monotonic()
        deadline = t0 + self._time_limit * TIME_BUDGET
        size = self._size

        if self._variant == "dark":
            board = self._determinize(board)

        empties = empty_cells(board, size)
        if len(empties) == 1:
            return empties[0]

        if len(empties) == size * size:
            return (size // 2, size // 2)

        # Movimiento ganador inmediato
        for m in empties:
            b = _board_to_lists(board)
            b[m[0]][m[1]] = self._player
            if check_winner(b, size) == self._player:
                return m

        # Bloquear victoria inmediata del oponente
        for m in empties:
            b = _board_to_lists(board)
            b[m[0]][m[1]] = self._opponent
            if check_winner(b, size) == self._opponent:
                return m

        # ── Calcular Dijkstra UNA SOLA VEZ por turno ─────────────────────
        # Con este pre-calculo: 2 Dijkstras totales por turno.
        # Sin el: ~4 Dijkstras por iteracion MCTS (expand×2 + rollout×2).
        b_tuple = tuple(tuple(row) for row in board)
        path_cells = {
            1: set(_dijkstra_path_cells(b_tuple, size, 1)),
            2: set(_dijkstra_path_cells(b_tuple, size, 2)),
        }
        # ─────────────────────────────────────────────────────────────────

        root = _Node(
            move=None,
            parent=None,
            untried_moves=list(empties),
            player_to_move=self._player,
        )

        while time.monotonic() < deadline:
            node, b_sim = self._select(root, board)
            if node.untried_moves:
                node, b_sim = self._expand(node, b_sim, path_cells)
            result, sim_moves = _informed_rollout(
                b_sim, size, node.player_to_move, self._player,
                path_cells, last_move   # FIX: last_move real, no None
            )
            self._backpropagate(node, result, sim_moves)

        if not root.children:
            return random.choice(empties)

        best = max(root.children, key=lambda n: n.visits)
        return best.move

    def _select(self, node, board):
        b = _board_to_lists(board)
        while node.is_fully_expanded() and node.children:
            best_score = -float("inf")
            best_child = None
            for child in node.children:
                s = child.uct_rave_score(node.visits, node.player_to_move)
                if s > best_score:
                    best_score = s
                    best_child = child
            node = best_child
            b[node.move[0]][node.move[1]] = 3 - node.player_to_move
        return node, b

    def _expand(self, node, board, path_cells):
        """Path bias + centro. 0 Dijkstras (usa path_cells pre-calculado)."""
        player = node.player_to_move
        path_set = path_cells.get(player, set())
        center = (self._size - 1) / 2.0
        PATH_BONUS = 4.0

        weights = []
        for (r, c) in node.untried_moves:
            dist = abs(r - center) + abs(c - center)
            center_w = 1.0 / (1.0 + dist)
            path_w = PATH_BONUS if (r, c) in path_set else 1.0
            weights.append(center_w * path_w)

        total_w = sum(weights)
        r_val = random.random() * total_w
        cumul = 0.0
        move = node.untried_moves[-1]
        for i, w in enumerate(weights):
            cumul += w
            if r_val <= cumul:
                move = node.untried_moves[i]
                break

        node.untried_moves.remove(move)
        b = [list(row) for row in board]
        b[move[0]][move[1]] = node.player_to_move
        next_player = 3 - node.player_to_move

        child = _Node(
            move=move,
            parent=node,
            untried_moves=list(empty_cells(b, self._size)),
            player_to_move=next_player,
        )
        node.children.append(child)
        return child, b

    def _backpropagate(self, node, result, sim_moves):
        amaf = defaultdict(set)
        for (move, player) in sim_moves:
            amaf[player].add(move)

        current = node
        while current is not None:
            current.visits += 1
            current.wins   += result
            if current.parent is not None:
                player_moved = current.parent.player_to_move
                for m in amaf[player_moved]:
                    current.parent.rave_visits[m] += 1
                    current.parent.rave_wins[m]   += result
            current = current.parent

    def _determinize(self, board):
        size = self._size
        known_opp = self._hidden_opp
        b = _board_to_lists(board)

        for (r, c) in known_opp:
            if b[r][c] == 0:
                b[r][c] = self._opponent

        my_count    = len(self._my_moves)
        known_count = len(known_opp)
        estimated_hidden = max(0, my_count - known_count - 1)

        available = [
            (r, c) for r in range(size) for c in range(size)
            if b[r][c] == 0 and (r, c) not in self._failed_moves
        ]

        center = size / 2
        def weight(rc):
            r, c = rc
            dist = abs(r - center) + abs(c - center)
            return max(1, size - dist)

        n_place = min(estimated_hidden, len(available))
        if n_place > 0:
            avail_copy = list(available)
            chosen = []
            for _ in range(n_place):
                if not avail_copy:
                    break
                total_w = sum(weight(rc) for rc in avail_copy)
                r_val = random.random()
                cumul = 0.0
                idx = 0
                for i, rc in enumerate(avail_copy):
                    cumul += weight(rc) / total_w
                    if r_val <= cumul:
                        idx = i
                        break
                chosen.append(avail_copy[idx])
                avail_copy.pop(idx)

            for (r, c) in chosen:
                b[r][c] = self._opponent

        return tuple(tuple(row) for row in b)
