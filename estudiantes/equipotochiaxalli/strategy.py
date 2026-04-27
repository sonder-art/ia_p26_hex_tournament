"""
Estrategia EquipoTochiaXalli para el torneo de Hex

Algoritmo: MCTS con UCT + rollouts heuristicos + bridge-pattern + tree reuse.

"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from strategy import Strategy, GameConfig
from hex_game import (
    NEIGHBORS,
    get_neighbors,
    check_winner,
    shortest_path_distance,
    empty_cells,
)

# Parametros del MCTS

UCT_C = 1.2                # constante de exploracion (algo menor a sqrt(2) =~1.41
                           # para favorecer un poco la explotacion en Hex 11x11)
TIME_FRACTION = 0.85       # usamos 85% del time budget por jugada
ROLLOUT_GREEDY_PROB = 0.0 # Cambiar de 0.85 a 0.0
MAX_ROLLOUT_STEPS = 200    # corte duro por seguridad
SIM_BATCH = 16             # cada cuantas iteraciones revisamos el reloj


# Heuristica: bridge / two-bridge en Hex

# Para una celda (r,c) en coordenadas hex offset, los 6 "two-bridge" patterns
# consisten en una celda destino y las dos celdas vacias que forman el puente.
# Si el oponente juega una de esas dos celdas vacias, debemos jugar la otra
# para mantener la conexion.
BRIDGE_PATTERNS = [
    # (delta_destino, delta_carrier1, delta_carrier2)
    ((-2,  1), (-1,  0), (-1,  1)),
    ((-1,  2), (-1,  1), ( 0,  1)),
    (( 1,  1), ( 0,  1), ( 1,  0)),
    (( 2, -1), ( 1,  0), ( 1, -1)),
    (( 1, -2), ( 1, -1), ( 0, -1)),
    ((-1, -1), ( 0, -1), (-1,  0)),
]


def find_forced_bridge_save(board, size, player, opponent, last_move):
    """Si el oponente jugo en una celda carrier de un two-bridge nuestro,
    devolvemos la otra celda carrier para salvar la conexion. None si no aplica.
    """
    if last_move is None:
        return None
    or_, oc = last_move
    # Para cada piedra nuestra, revisamos si forma un puente con otra piedra
    # nuestra y si la celda jugada por el oponente es uno de los carriers.
    for r in range(size):
        for c in range(size):
            if board[r][c] != player:
                continue
            for d_dest, d_c1, d_c2 in BRIDGE_PATTERNS:
                dr, dc = r + d_dest[0], c + d_dest[1]
                if not (0 <= dr < size and 0 <= dc < size):
                    continue
                if board[dr][dc] != player:
                    continue
                cr1, cc1 = r + d_c1[0], c + d_c1[1]
                cr2, cc2 = r + d_c2[0], c + d_c2[1]
                if not (0 <= cr1 < size and 0 <= cc1 < size):
                    continue
                if not (0 <= cr2 < size and 0 <= cc2 < size):
                    continue
                # Caso: oponente acaba de invadir el carrier 1, salvamos con carrier 2
                if (or_, oc) == (cr1, cc1) and board[cr2][cc2] == 0:
                    return (cr2, cc2)
                if (or_, oc) == (cr2, cc2) and board[cr1][cc1] == 0:
                    return (cr1, cc1)
    return None


# Rollout inform

def rollout_move(board, size, to_move):
    """Elige una jugada para 'to_move' favoreciendo movimientos que reducen
    la distancia mas corta de su conexion, con probabilidad ROLLOUT_GREEDY_PROB.
    Con 1 - ROLLOUT_GREEDY_PROB juega aleatorio entre las celdas vacias.
    """
    moves = empty_cells(board, size)
    if not moves:
        return None
    if random.random() > ROLLOUT_GREEDY_PROB:
        return random.choice(moves)

    # Considerar solo un subconjunto de candidatos para velocidad
    if len(moves) > 25:
        candidates = random.sample(moves, 25)
    else:
        candidates = moves

    best = None
    best_score = float("inf")
    opponent = 3 - to_move
    for (r, c) in candidates:
        board[r][c] = to_move
        d_self = shortest_path_distance(board, size, to_move)
        d_opp = shortest_path_distance(board, size, opponent)
        board[r][c] = 0
        # Queremos minimizar nuestra distancia y a la vez no facilitar al oponente.
        score = d_self - 0.5 * d_opp
        # Pequeno ruido para romper empates
        score += random.random() * 0.01
        if score < best_score:
            best_score = score
            best = (r, c)
    return best if best is not None else random.choice(moves)


def simulate(board_list, size, current_player, root_player):
    """Simulacion (rollout) hasta que haya ganador o se agoten celdas.
    Modifica board_list in-place y luego lo restaura. Devuelve +1 si root_player gana, -1 si pierde.
    """
    placed = []
    cp = current_player
    winner = check_winner(board_list, size)
    steps = 0
    while winner == 0 and steps < MAX_ROLLOUT_STEPS:
        mv = rollout_move(board_list, size, cp)
        if mv is None:
            break
        r, c = mv
        board_list[r][c] = cp
        placed.append((r, c))
        winner = check_winner(board_list, size)
        cp = 3 - cp
        steps += 1

    # Restaurar tablero
    for (r, c) in placed:
        board_list[r][c] = 0

    if winner == root_player:
        return 1.0
    if winner == 3 - root_player:
        return -1.0
    return 0.0


# Nodo MCTS

class Node:
    __slots__ = ("parent", "move", "to_move", "children", "untried", "N", "Q", "terminal_winner")

    def __init__(self, parent, move, to_move, untried, terminal_winner=0):
        self.parent = parent
        self.move = move                  # movimiento que llevo a este nodo (desde el padre)
        self.to_move = to_move            # jugador a mover en este nodo
        self.children = []
        self.untried = untried            # lista de movimientos no expandidos
        self.N = 0
        self.Q = 0.0
        self.terminal_winner = terminal_winner  # 0 si no terminal

    def is_fully_expanded(self):
        return not self.untried

    def best_uct_child(self, c_param):
        log_n = math.log(self.N) if self.N > 0 else 0.0
        best = None
        best_val = -float("inf")
        for child in self.children:
            if child.N == 0:
                return child  # priorizar no visitados
            exploit = child.Q / child.N
            explore = c_param * math.sqrt(log_n / child.N)
            val = exploit + explore
            if val > best_val:
                best_val = val
                best = child
        return best


class HexMCTSStrategy(Strategy):

    @property
    def name(self) -> str:
        return "EquipoTochiaXalli"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size
        self._player = config.player
        self._opponent = config.opponent
        self._variant = config.variant
        self._time_limit = config.time_limit
        self._root: Optional[Node] = None
        # Para dark mode: celdas donde colisionamos (sabemos que hay piedra del oponente)
        self._known_opp_cells = set()
        # Ultimo movimiento que retornamos en play() (para usarlo en on_move_result)
        self._last_attempted: Optional[tuple] = None
        # En dark, vamos guardando piedras propias para reconstruir el tablero "creible"
        self._my_stones = set()

    def on_move_result(self, move, success):
        if success:
            if self._variant == "dark":
                self._my_stones.add(tuple(move))
        else:
            # Colision: hay piedra del oponente en esa celda
            self._known_opp_cells.add(tuple(move))
            # Invalidamos el arbol porque la informacion cambio
            self._root = None

    # principal
    def play(self, board, last_move):
        # Convertir a list-of-list mutable
        size = self._size
        bd = [list(row) for row in board]

        # 0) Si es el primer movimiento del juego, juega cerca del centro (apertura fuerte)
        moves = empty_cells(bd, size)
        if len(moves) == size * size:
            return (size // 2, size // 2)

        # 1) Bridge save: defensa forzada
        forced = find_forced_bridge_save(bd, size, self._player, self._opponent, last_move)
        if forced is not None:
            return forced

        # 2) Win-in-one: si hay jugada que gana inmediatamente, hazla
        for (r, c) in moves:
            bd[r][c] = self._player
            if check_winner(bd, size) == self._player:
                bd[r][c] = 0
                return (r, c)
            bd[r][c] = 0

        # 3) Block-in-one: si el oponente puede ganar en una jugada, bloquea
        for (r, c) in moves:
            bd[r][c] = self._opponent
            if check_winner(bd, size) == self._opponent:
                bd[r][c] = 0
                return (r, c)
            bd[r][c] = 0

        # 4) MCTS con UCT
        deadline = time.monotonic() + self._time_limit * TIME_FRACTION

        # Reusar arbol si es posible
        root = self._reuse_or_new_root(bd, last_move)

        # Loop principal
        iters = 0
        while True:
            # check de tiempo periodico para no medir reloj cada iteracion
            if iters % SIM_BATCH == 0 and time.monotonic() > deadline:
                break

            self._mcts_iteration(root, bd)
            iters += 1

            # corte temprano si ya tenemos suficiente confianza
            if iters >= 5000:
                break

        if not root.children:
            # fallback
            return moves[0]

        # Elegir hijo con MAS visitas (mas robusto que mayor Q/N)
        best = max(root.children, key=lambda ch: ch.N)
        # Guardar root para reusar en el proximo turno
        self._root = best  # avanzar al subarbol que corresponde a nuestra jugada
        return best.move

    # Internals 

    def _reuse_or_new_root(self, board_list, last_move):
        """Intenta reusar el subarbol que corresponde al estado actual.
        Si no es posible, crea uno nuevo."""
        size = self._size
        if (
            self._variant == "classic"
            and self._root is not None
            and last_move is not None
        ):
            # En el turno anterior elegimos un movimiento, _root quedo apuntando al subarbol
            # que correspondia a nuestra jugada. Ahora el oponente jugo last_move desde ahi.
            for child in self._root.children:
                if child.move == last_move:
                    child.parent = None
                    return child

        # Nuevo arbol
        moves = empty_cells(board_list, size)
        random.shuffle(moves)
        return Node(parent=None, move=None, to_move=self._player, untried=moves)

    def _mcts_iteration(self, root, board_template):
        """Una iteracion de MCTS: SELECT - EXPAND - SIMULATE - BACKPROP.
        Trabajamos sobre una copia del tablero para no corromperlo."""
        size = self._size

        # Tablero de trabajo
        bd = [row[:] for row in board_template]

        # 1) SELECT
        node = root
        path = [node]
        while node.is_fully_expanded() and node.children:
            node = node.best_uct_child(UCT_C)
            r, c = node.move
            bd[r][c] = 3 - node.to_move  # node.to_move es el siguiente, asi que la jugada fue del previo
            path.append(node)
            if node.terminal_winner != 0:
                break

        # 2) espand (si no es terminal)
        if node.terminal_winner == 0 and node.untried:
            mv = node.untried.pop()
            r, c = mv
            bd[r][c] = node.to_move
            next_to_move = 3 - node.to_move
            winner = check_winner(bd, size)
            if winner != 0:
                untried_next = []
            else:
                untried_next = empty_cells(bd, size)
                random.shuffle(untried_next)
            child = Node(parent=node, move=mv, to_move=next_to_move,
                         untried=untried_next, terminal_winner=winner)
            node.children.append(child)
            node = child
            path.append(node)

        # 3) Simulate
        if node.terminal_winner != 0:
            reward = 1.0 if node.terminal_winner == self._player else -1.0
        else:
            reward = simulate(bd, size, node.to_move, self._player)

        # 4) Backprop
        # El reward esta en perspectiva del root_player (self._player).
        # Cuando un nodo va a moverlo el oponente, su Q/N debe leer la perspectiva
        # del oponente, asi que invertimos signo segun to_move.
        for n in path:
            n.N += 1
            # Si n.to_move == self._player, este nodo es del oponente cuando se llego (acaba de mover)
            # En la formula Q/N tradicional, Q se lleva en perspectiva del jugador que MOVERA.
            # Para simplificar y evitar inconsistencias, almacenamos Q desde la perspectiva
            # del jugador que ACABA DE MOVER (padre). Como root.move es None (no mueve nadie),
            # usamos un truco: invertimos en la seleccion via signo segun to_move.
            # Aqui acumulamos directamente reward desde perspectiva root y lo signamos al usar.
            # Mejor: guardamos siempre desde perspectiva root.
            if n.to_move == self._player:
                n.Q -= reward 
            else:
                n.Q += reward


def get_strategy() -> Strategy:
    return HexMCTSStrategy()