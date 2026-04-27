"""
politopos

Arquitectura: multiprocessing (main + 3 workers), tree reuse, poda de candidatos
Inteligencia: íntegramente de v6 — _score, _combined, _distance, _rollout, _sample_world

Arquitectura modificaciones:
  - Motor paralelo: 3 procesos worker adicionales hacen MCTS simultáneamente
    y al final suman sus visitas para votar el movimiento. ~4x más iteraciones.
  - Tree reuse: el árbol del turno anterior se reutiliza descendiendo 2 niveles
    (mi movimiento → respuesta del oponente). No se tira el trabajo acumulado.
  - Poda de candidatos: solo se consideran casillas en vecindad radio-2 de las
    piezas existentes (~20 candidatos vs ~100). El árbol explora más densamente.
  - _EmptyPool O(1): eliminación de casillas en O(1) en lugar de list.remove() O(n).
  - Dijkstra correcto con heapq para _distance.
  - _sample_world corregida: estima piedras ocultas del oponente en lugar de 50%.
  - Guard para root.children vacío.
  - Victoria/bloqueo inmediato detectado antes de entrar al MCTS.

inteligencia original:
  - _score: progreso + conectividad
  - _combined: score propio - 0.7 * score oponente
  - _distance: BFS ponderado (Dijkstra) para evaluar fin de rollout
  - _rollout: sample de 6, max por _combined, hasta MAX_SIMULATION_DEPTH
  - _sample_world: generación de mundos para dark mode
  - Lógica de dark: _known_board, _collision_history, on_move_result
"""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict

from strategy import Strategy, GameConfig
from hex_game import get_neighbors, check_winner, empty_cells

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
EXPLORATION_CONSTANT = 1.25
MAX_SIMULATION_DEPTH = 100
NUM_BELIEF_SAMPLES = 4
NUM_WORKERS = 3  # procesos paralelos adicionales
TIME_BUDGET = 0.92  # fracción del tiempo límite que usamos
EXPAND_RADIUS = 2  # radio de vecindad para poda de candidatos


# ---------------------------------------------------------------------------
# _EmptyPool — eliminación O(1) con swap-and-pop
# Reemplaza list.remove() que era O(n) en cada paso del MCTS
# ---------------------------------------------------------------------------
class _EmptyPool:
    __slots__ = ("cells", "pos")

    def __init__(self, iterable=()):
        self.cells = []
        self.pos = {}
        for cell in iterable:
            if cell not in self.pos:
                self.pos[cell] = len(self.cells)
                self.cells.append(cell)

    def copy(self):
        p = _EmptyPool()
        p.cells = self.cells[:]
        p.pos = dict(self.pos)
        return p

    def remove(self, cell):
        if cell not in self.pos:
            return
        idx = self.pos.pop(cell)
        last = self.cells.pop()
        if idx < len(self.cells):
            self.cells[idx] = last
            self.pos[last] = idx

    def sample(self, k, rng):
        return rng.sample(self.cells, min(k, len(self.cells)))

    def __len__(self):
        return len(self.cells)

    def __bool__(self):
        return bool(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __contains__(self, cell):
        return cell in self.pos


# ---------------------------------------------------------------------------
# Nodo MCTS


# ---------------------------------------------------------------------------
class Node:
    __slots__ = (
        "move",
        "parent",
        "children",
        "visits",
        "wins",
        "player",
        "untried",
        "rave_wins",
        "rave_visits",
    )

    def __init__(self, move=None, parent=None, player=1, moves=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.player = player
        self.untried = list(moves) if moves else []
        self.rave_wins = defaultdict(float)
        self.rave_visits = defaultdict(int)

    def uct_rave(self):
        if self.visits == 0:
            return float("inf")

        q = self.wins / self.visits
        u = EXPLORATION_CONSTANT * math.sqrt(math.log(self.parent.visits) / self.visits)

        # RAVE: estadísticas del padre sobre este movimiento
        if self.move and self.parent:
            rv = self.parent.rave_visits[self.move]
            if rv > 0:
                rq = self.parent.rave_wins[self.move] / rv
                beta = rv / (rv + self.visits + 1e-6)
                q = (1 - beta) * q + beta * rq

        return q + u


# ---------------------------------------------------------------------------
# Funciones standalone (deben ser pickleable para multiprocessing)
# Contienen la inteligencia
# ---------------------------------------------------------------------------


def _score(board, size, r, c, player):
    """Heurística de v6: progreso hacia borde objetivo + conectividad."""
    center = (size - 1) / 2
    if player == 1:
        progress = 1 - abs(r - center) / center if center > 0 else 1.0
    else:
        progress = 1 - abs(c - center) / center if center > 0 else 1.0
    conn = sum(1 for nr, nc in get_neighbors(r, c, size) if board[nr][nc] == player)
    return 0.7 * progress + 0.3 * (conn / 6)


def _combined(board, size, r, c, player):
    """Score compuesto de v6: propio - 0.7 * oponente."""
    return _score(board, size, r, c, player) - 0.7 * _score(
        board, size, r, c, 3 - player
    )


def _distance(board, size, player):
    """
    Dijkstra correcto con heapq.
    Corrige el BFS de v6 que no garantizaba mínimos con pesos heterogéneos.
    La lógica de distancia es la misma — pesos 0 (piedra propia) y 1 (vacío).
    """
    opp = 3 - player
    INF = size * size + 1
    dist = {}
    heap = []

    if player == 1:
        sources = [(0, c) for c in range(size) if board[0][c] != opp]
    else:
        sources = [(r, 0) for r in range(size) if board[r][0] != opp]

    for r, c in sources:
        d = 0 if board[r][c] == player else 1
        if d < dist.get((r, c), INF):
            dist[(r, c)] = d
            heapq.heappush(heap, (d, r, c))

    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist.get((r, c), INF):
            continue
        for nr, nc in get_neighbors(r, c, size):
            if board[nr][nc] == opp:
                continue
            nd = d + (0 if board[nr][nc] == player else 1)
            if nd < dist.get((nr, nc), INF):
                dist[(nr, nc)] = nd
                heapq.heappush(heap, (nd, nr, nc))

    if player == 1:
        return min(dist.get((size - 1, c), INF) for c in range(size))
    else:
        return min(dist.get((r, size - 1), INF) for r in range(size))


def _rollout(board, size, pool: _EmptyPool, player, root_player, rng):
    """
    Rollout de v6: sample de 6 candidatos, elige por _combined.
    Pool es un _EmptyPool — se copia antes de llamar para no mutarlo.
    """
    b = [row[:] for row in board]
    free = pool.copy()
    current = player
    seq = []

    for _ in range(MAX_SIMULATION_DEPTH):
        if not free:
            break

        sample = free.sample(6, rng)
        move = max(sample, key=lambda m: _combined(b, size, m[0], m[1], current))

        b[move[0]][move[1]] = current
        free.remove(move)
        seq.append((move, current))

        w = check_winner(b, size)
        if w != 0:
            return (1.0 if w == root_player else 0.0), seq

        current = 3 - current

    d1 = _distance(b, size, root_player)
    d2 = _distance(b, size, 3 - root_player)
    result = 1.0 if d1 <= d2 else 0.0
    return result, seq


def _candidates(board, size, empties, radius=EXPAND_RADIUS):
    """
    Poda de candidatos: solo casillas en vecindad radio-2 de piezas existentes.
    Reduce candidatos de ~100 a ~20-30 en partidas medianas.
    Si no hay piezas aún, retorna las casillas centrales.
    """
    pieces_exist = False
    in_nbhd = [[False] * size for _ in range(size)]

    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                pieces_exist = True
                for nr in range(max(0, r - radius), min(size, r + radius + 1)):
                    for nc in range(max(0, c - radius), min(size, c + radius + 1)):
                        in_nbhd[nr][nc] = True

    if not pieces_exist:
        # Apertura: zona central
        center = size // 2
        rad = size // 3
        return [
            (r, c)
            for r, c in empties
            if abs(r - center) <= rad and abs(c - center) <= rad
        ]

    result = [m for m in empties if in_nbhd[m[0]][m[1]]]
    return result if len(result) >= 4 else list(empties)


def _build_root(board, size, player, moves):
    """
    Construye el nodo raíz con candidatos ordenados por _score descendente.
    El orden es ascendente en la lista para que pop() (del final) dé el mejor.
    """
    cands = _candidates(board, size, moves)
    if not cands:
        cands = list(moves)
    # Ordenar ascendente → pop() saca el mejor
    cands.sort(key=lambda m: _score(board, size, m[0], m[1], player))
    return Node(player=player, moves=cands)


def _mcts_iteration(node, board, size, root_player, initial_pool, rng):
    """Una iteración completa de MCTS: selección → expansión → rollout → backprop."""
    b = [row[:] for row in board]
    pool = initial_pool.copy()
    current = root_player

    # --- Selección ---
    while not node.untried and node.children:
        node = max(node.children, key=lambda n: n.uct_rave())
        r, c = node.move
        b[r][c] = current
        pool.remove(node.move)
        current = 3 - current

    # --- Expansión ---
    if node.untried:
        move = node.untried.pop()  # pop del final = mejor heurístico
        r, c = move
        b[r][c] = current
        pool.remove(move)

        # Hijos ordenados ascendente por _score del próximo jugador
        child_cands = _candidates(b, size, list(pool))
        child_cands.sort(key=lambda m: _score(b, size, m[0], m[1], 3 - current))
        child = Node(move=move, parent=node, player=3 - current, moves=child_cands)
        node.children.append(child)
        node = child
        current = 3 - current

    # --- Rollout con inteligencia de v6 ---
    result, seq = _rollout(b, size, pool, current, root_player, rng)

    # --- Backpropagación con RAVE ---
    n = node
    while n is not None:
        n.visits += 1
        n.wins += result
        # RAVE: actualizar estadísticas del padre para cada movimiento del rollout
        if n.parent is not None:
            for mv, mv_player in seq:
                if mv_player == n.parent.player:
                    n.parent.rave_visits[mv] += 1
                    n.parent.rave_wins[mv] += result
        n = n.parent

    return node  # no usado, pero útil para depuración


def _worker_run(args):
    """
    Proceso worker independiente: corre MCTS con rollout de v6 por `duration` segundos.
    Retorna dict {move: visits} para sumar al proceso principal.
    """
    board_tuple, size, player, duration, seed = args
    rng = random.Random(seed)

    empties = [
        (r, c) for r in range(size) for c in range(size) if board_tuple[r][c] == 0
    ]
    if not empties:
        return {}

    board = [list(row) for row in board_tuple]
    root = _build_root(board, size, player, empties)
    pool = _EmptyPool(empties)

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline and (root.untried or root.children):
        _mcts_iteration(root, board, size, player, pool, rng)

    return {child.move: child.visits for child in root.children}


# ---------------------------------------------------------------------------
# Estrategia principal
# ---------------------------------------------------------------------------
class Politopos(Strategy):

    @property
    def name(self):
        return "politopos"

    def begin_game(self, config: GameConfig):
        self.size = config.board_size
        self.player = config.player
        self.opponent = config.opponent
        self.time_limit = config.time_limit
        self.variant = getattr(config, "variant", "classic")
        self.rng = random.Random()

        self._my_stone_count = 0
        self._move_count = 0

        # Tree reuse
        self._root = None
        self._last_my_move = None

        # Dark mode
        if self.variant == "dark":
            self._known_board = [[0] * self.size for _ in range(self.size)]
            self._collision_history = set()

        # Pool de workers paralelos (fork para velocidad)
        self._shutdown_pool()
        try:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(NUM_WORKERS)
        except Exception:
            self._pool = None

    def _shutdown_pool(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        self._pool = None

    def on_move_result(self, move, success):
        if self.variant == "dark":
            r, c = move
            if success:
                self._known_board[r][c] = self.player
                self._my_stone_count += 1
            else:
                self._known_board[r][c] = self.opponent
                self._collision_history.add(move)

    # -----------------------------------------------------------------------
    # play — punto de entrada principal
    # -----------------------------------------------------------------------
    def play(self, board, last_move):
        self._move_count += 1
        board = [list(r) for r in board]

        # --- Dark mode: actualizar conocimiento y generar mundos ---
        if self.variant == "dark":
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] != 0:
                        self._known_board[r][c] = board[r][c]
            belief_boards = [self._sample_world() for _ in range(NUM_BELIEF_SAMPLES)]
            # En dark no podemos hacer tree reuse (el tablero cambia impredeciblemente)
            self._root = None
        else:
            belief_boards = [board]

        moves = empty_cells(board, self.size)

        if self.variant == "dark" and self._collision_history:
            filtered = [m for m in moves if m not in self._collision_history]
            if filtered:
                moves = filtered

        if not moves:
            return (0, 0)
        if len(moves) == 1:
            self._root = None
            return moves[0]

        # --- Victoria inmediata ---
        for m in moves:
            b = [row[:] for row in board]
            b[m[0]][m[1]] = self.player
            if check_winner(b, self.size) == self.player:
                self._root = None
                return m

        # --- Bloqueo de victoria del oponente ---
        for m in moves:
            b = [row[:] for row in board]
            b[m[0]][m[1]] = self.opponent
            if check_winner(b, self.size) == self.opponent:
                self._root = None
                return m

        duration = self.time_limit * TIME_BUDGET

        if self.variant == "dark":
            best = self._play_dark(board, belief_boards, moves, duration)
        else:
            best = self._play_classic(board, moves, last_move, duration)

        self._last_my_move = best
        return best

    # -----------------------------------------------------------------------
    # Classic: MCTS paralelo con tree reuse
    # -----------------------------------------------------------------------
    def _play_classic(self, board, moves, last_move, duration):
        board_tuple = tuple(tuple(r) for r in board)

        # Intentar reutilizar el árbol del turno anterior
        root = self._descend_root(last_move)
        if root is None:
            root = _build_root(board, self.size, self.player, moves)

        pool_obj = _EmptyPool(
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if board[r][c] == 0
        )

        # Lanzar workers paralelos
        async_result = None
        worker_budget = max(0.05, duration - 0.1)
        if self._pool is not None:
            try:
                args = [
                    (
                        board_tuple,
                        self.size,
                        self.player,
                        worker_budget,
                        random.randint(0, 2**31),
                    )
                    for _ in range(NUM_WORKERS)
                ]
                async_result = self._pool.map_async(_worker_run, args)
            except Exception:
                async_result = None

        # MCTS en proceso principal
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and (root.untried or root.children):
            _mcts_iteration(root, board, self.size, self.player, pool_obj, self.rng)

        # Recoger votos de workers
        vote_counts = defaultdict(int)
        for child in root.children:
            vote_counts[child.move] += child.visits

        if async_result is not None:
            try:
                for worker_votes in async_result.get(timeout=1.5):
                    for mv, v in worker_votes.items():
                        vote_counts[mv] += v
            except Exception:
                pass

        self._root = root

        if not vote_counts:
            return moves[0]

        return max(vote_counts, key=vote_counts.get)

    # -----------------------------------------------------------------------
    # Dark: MCTS sobre múltiples mundos posibles (lógica original de v6)
    # -----------------------------------------------------------------------
    def _play_dark(self, board, belief_boards, moves, duration):
        """
        Mantiene la lógica original de v6: corre MCTS en cada mundo posible
        y vota el movimiento más elegido.
        Con el motor paralelo, los workers también samplearán sobre mundos.
        """
        move_scores = defaultdict(int)
        time_per_world = duration / NUM_BELIEF_SAMPLES

        for b_world in belief_boards:
            # Filtrar moves al mundo actual
            world_moves = [m for m in moves if b_world[m[0]][m[1]] == 0]
            if not world_moves:
                world_moves = moves

            root = _build_root(b_world, self.size, self.player, world_moves)
            pool_obj = _EmptyPool(
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if b_world[r][c] == 0
            )

            deadline = time.monotonic() + time_per_world
            while time.monotonic() < deadline and (root.untried or root.children):
                _mcts_iteration(
                    root, b_world, self.size, self.player, pool_obj, self.rng
                )

            if root.children:
                best = max(root.children, key=self._final).move
                move_scores[best] += 1

        if not move_scores:
            return moves[0]
        return max(move_scores, key=move_scores.get)

    # -----------------------------------------------------------------------
    # Tree reuse: descender 2 niveles (mi mov → respuesta del oponente)
    # -----------------------------------------------------------------------
    def _descend_root(self, opp_last_move):
        if self._root is None or self._last_my_move is None or opp_last_move is None:
            return None

        my_child = next(
            (c for c in self._root.children if c.move == self._last_my_move), None
        )
        if my_child is None:
            return None

        opp_child = next(
            (c for c in my_child.children if c.move == opp_last_move), None
        )
        if opp_child is None:
            return None

        opp_child.parent = None
        return opp_child

    # -----------------------------------------------------------------------
    # Sample world — corregido respecto a v6
    # -----------------------------------------------------------------------
    def _sample_world(self):
        """
        Genera un mundo posible para dark mode.

        CORRECCIÓN vs v6 original: en lugar de asignar el oponente al 50%
        de todas las casillas vacías (mundos imposibles), estimamos cuántas
        piedras ocultas tiene el oponente y colocamos exactamente ese número.

        Estimación: el oponente ha jugado aproximadamente los mismos turnos
        que yo (_my_stone_count), menos las ya conocidas por colisión.
        """
        b = [row[:] for row in self._known_board]

        known_opp = sum(
            1
            for r in range(self.size)
            for c in range(self.size)
            if b[r][c] == self.opponent
        )
        hidden_opp = max(0, self._my_stone_count - known_opp)

        if hidden_opp > 0:
            available = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if b[r][c] == 0 and (r, c) not in self._collision_history
            ]
            n_place = min(hidden_opp, len(available))
            if n_place > 0:
                for r, c in self.rng.sample(available, n_place):
                    b[r][c] = self.opponent

        return b

    # -----------------------------------------------------------------------
    # Selección final del movimiento
    # -----------------------------------------------------------------------
    def _final(self, node):
        if node.visits == 0:
            return 0
        wr = node.wins / node.visits
        return 0.6 * node.visits + 0.4 * wr * 1000
