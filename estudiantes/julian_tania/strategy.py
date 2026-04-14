"""
HexMaster Strategy v3 — MCTS de máximo rendimiento para Hex 11×11

╔════════════════════════════════════════════════════════════════╗
║ ¿QUÉ ES HEXMASTER?                                             ║
║ Una IA que juega Hex usando "Monte Carlo Tree Search" (MCTS):  ║
║ simula miles de partidas aleatorias para encontrar el mejor    ║
║ movimiento en el tiempo disponible.                            ║
╚════════════════════════════════════════════════════════════════╝

COMPONENTES PRINCIPALES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRECÁLCULO (líneas ~30-60)
   • Vecinos de cada celda (lista de listas)
   • Bordes objetivo (celdas en row/col 0,10)
   → Se hacen UNA sola vez al importar el módulo → RÁPIDO

2. ROLLOUT RÁPIDO (líneas ~62-110)
   • Simula el resto de una partida automáticamente
   • Usa BFS con "sellos de generación" para detectar victoria
   • Mucho más rápido que recalcular cada vez → EFICIENTE

3. HEURÍSTICA DIJKSTRA (líneas ~112-132)
   • Evalúa movimientos pequeños usando distancia de camino
   • "¿Qué tan cerca llego a mi objetivo con este movimiento?"
   • Solo se usa con ≤8 candidatos (mucho cálculo)

4. ÁRBOL MCTS (líneas ~134-169)
   Nodo = un movimiento explorando en el árbol
   • move: la jugada que representa
   • wins/visits: estadísticas (cuántas veces ganó/probó)
   • untried: movimientos no explorados aún
   • uct_child(): elige el hijo más prometedor usando fórmula UCT

5. BUCLE PRINCIPAL MCTS (líneas ~171-282)
   Mientras hay tiempo:
   a) SELECCIÓN: navega por el árbol existente
   b) EXPANSIÓN: agrega un nuevo movimiento
   c) ROLLOUT: simula el juego aleatoriamente hasta el fin
   d) BACKPROPAGATION: actualiza estadísticas hacia arriba

6. DARK MODE (líneas ~284-299)
   • En modo oscuro no ves todas las piedras
   • Adivina dónde está el oponente aleatoriamente
   • Crea un tablero completo ficticio para analizar

7. ESTRATEGIA PRINCIPAL (líneas ~301-370)
   play() = el método que toma una decisión cada turno
   • Apertura: juega en el centro (sin gastar MCTS)
   • Determinización: prepara tablero para dark mode
   • MCTS: búsqueda principal con presupuesto de tiempo
   • Fallback: si falla tiempo, usa heurística rápida

OPTIMIZACIONES CLAVE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Arrays planos en lugar de objetos (más rápido)
✓ Reutilización de memoria (BFS, sellos de generación)
✓ Precálculo de vecinos y bordes (O(1) al consultar)
✓ Sets para celdas vacías (discard es O(1))
✓ Análisis inteligente solo cuando hay pocos candidatos
✓ Presupuesto temporal conservador (92% del límite)

VER: README_ESTRATEGIA.md para explicación completa paso a paso
"""

from __future__ import annotations

import random
import time
import math

from strategy import Strategy, GameConfig
from hex_game import (
    shortest_path_distance,
    empty_cells,
    NEIGHBORS,
)

# ---------------------------------------------------------------------------
# Precálculo: VECINOS y BORDES OBJETIVO
# (Se ejecuta una sola vez al importar el módulo)
# ---------------------------------------------------------------------------

_SIZE = 11
_N    = _SIZE * _SIZE  # 121 celdas totales

# Direcciones de vecindad en Hex (6 direcciones)
_DIRS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

def _build_neighbors(size):
    """Para cada celda, calcula sus vecinos hexagonales."""
    nb = [[] for _ in range(size * size)]
    for r in range(size):
        for c in range(size):
            cell = r * size + c
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nb[cell].append(nr * size + nc)
    return nb

_NB = _build_neighbors(_SIZE)

# Identificar bordes objetivo para cada jugador
# Jugador 1: conecta fila 0 (inicio) con fila 10 (fin)
# Jugador 2: conecta col 0 (inicio) con col 10 (fin)
_P1_START = [c < _SIZE for c in range(_N)]          # fila 0
_P1_END   = [c >= _N - _SIZE for c in range(_N)]    # fila 10
_P2_START = [c % _SIZE == 0 for c in range(_N)]      # col 0
_P2_END   = [c % _SIZE == _SIZE - 1 for c in range(_N)]  # col 10

# ---------------------------------------------------------------------------
# ROLLOUT: Simulación rápida del resto de la partida
# ---------------------------------------------------------------------------
# Cuando MCTS necesita saber el resultado de una posición, "simula" 
# automáticamente el resto del juego:
# 1. Llena las celdas vacías: primero turnos realistas, luego al azar
# 2. Usa BFS para detectar si hay camino ganador
# 3. Usa "sellos de generación" para reutilizar memoria (¡muy rápido!)

# Estado persistente (reutilizable entre simulaciones)
_stamp  = [0] * _N
_bfs_q  = [0] * _N
_gen    = 0

def fast_rollout(arr, empty_set, cur_mover, other_mover, eval_player):
    """
    Simula aleatoriamente el resto del juego.
    
    Args:
        arr: estado actual del tablero
        empty_set: índices de celdas vacías
        cur_mover: quién juega ahora
        other_mover: quién juega después
        eval_player: de quién queremos saber si gana (perspectiva)
    
    Returns:
        1.0 si eval_player gana, 0.0 si no
    """
    global _gen

    sim = arr[:]
    moves = list(empty_set)
    random.shuffle(moves)
    # Estrategia de llenado: primeros ~50% con cur_mover, resto con other_mover
    half = (len(moves) + 1) >> 1
    for i in range(len(moves)):
        sim[moves[i]] = cur_mover if i < half else other_mover

    # Seleccionar bordes según quién evaluamos
    if eval_player == 1:
        start_arr = _P1_START
        end_arr   = _P1_END
    else:
        start_arr = _P2_START
        end_arr   = _P2_END

    # BFS con "sello de generación" → evita limpiar arrays
    _gen += 1
    gen = _gen
    qi = 0
    q_end = 0

    # Inicializar BFS desde bordes de inicio
    for cell in range(_N):
        if start_arr[cell] and sim[cell] == eval_player:
            _stamp[cell] = gen
            _bfs_q[q_end] = cell
            q_end += 1

    # Buscar camino a borde final
    nb = _NB
    while qi < q_end:
        cell = _bfs_q[qi]; qi += 1
        if end_arr[cell]:
            return 1.0  # ¡GANÓ!
        for ncell in nb[cell]:
            if sim[ncell] == eval_player and _stamp[ncell] != gen:
                _stamp[ncell] = gen
                _bfs_q[q_end] = ncell
                q_end += 1

    return 0.0  # No hay camino ganador
# ---------------------------------------------------------------------------
# HEURÍSTICA RÁPIDA: Dijkstra para guiar expansión y fallback
# ---------------------------------------------------------------------------

def _expansion_score(arr, size, player, opponent, r, c):
    """
    Evalúa qué tan bueno sería colocar una piedra en (r,c).
    
    Lógica:
    - Si coloco en (r,c), ¿cuál es mi distancia más corta a la victoria?
    - ¿Y la del oponente?
    - Score = distancia_oponente - mi_distancia
    
    Valores altos = buen movimiento (acerca más al oponente a su meta que a mí)
    Valores bajos = malo (yo le acerco a mi meta más que a él)
    
    Se usa:
    1. En expansión MCTS (si hay ≤8 candidatos)
    2. En fallback si se acaba el tiempo
    """
    arr[r * size + c] = player
    board_t = tuple(
        tuple(arr[row * size + col] for col in range(size))
        for row in range(size)
    )
    d_me  = shortest_path_distance(board_t, size, player)
    d_opp = shortest_path_distance(board_t, size, opponent)
    arr[r * size + c] = 0

    if d_opp == float('inf'):
        return  1e9  # Oponente no puede ganar → excelente
    if d_me  == float('inf'):
        return -1e9  # Yo no puedo ganar → terrible
    return d_opp - d_me


# ---------------------------------------------------------------------------
# MCTS: Árbol de Monte Carlo Tree Search
# ---------------------------------------------------------------------------
# Cada nodo representa una posición en el "árbol de posibilidades"
# Mantiene estadísticas de cuántas veces se exploró y cuántas se ganó

class _Node:
    """
    Nodo del árbol MCTS.
    
    Atributos:
        move: la jugada (r,c) que llevó a este nodo
        parent: nodo padre en el árbol
        children: nodos hijos (jugadas expandidas)
        wins: número de simulaciones ganadas desde aquí
        visits: número total de simulaciones
        untried: movimientos no explorados aún
        mover: quién movió para llegar aquí
    """
    __slots__ = ('move', 'parent', 'children', 'wins', 'visits',
                 'untried', 'mover')

    def __init__(self, move, parent, untried, mover):
        self.move     = move
        self.parent   = parent
        self.children = []
        self.wins     = 0.0
        self.visits   = 0
        self.untried  = list(untried)
        self.mover    = mover

    def uct_child(self, c=1.8):
        """
        Selecciona el hijo más prometedor.
        
        Usa la fórmula UCT (Upper Confidence bounds applied to Trees):
        score = (wins/visits) + c * sqrt(ln(parent_visits) / visits)
        
        Balance entre:
        - Explotación: hacer más simulaciones en buenos movimientos
        - Exploración: probar movimientos menos conocidos
        """
        log_n = math.log(self.visits)
        bv = -1.0; bc = None
        for ch in self.children:
            v = ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits)
            if v > bv:
                bv = v; bc = ch
        return bc

    def expand(self, move, untried, mover):
        """Agrega un nuevo hijo al árbol."""
        child = _Node(move, self, untried, mover)
        self.untried.remove(move)
        self.children.append(child)
        return child

    def backup(self, result):
        """Actualiza estadísticas: +1 visita y +resultado en victorias."""
        self.visits += 1
        self.wins   += result


# ---------------------------------------------------------------------------
# BUCLE PRINCIPAL: Monte Carlo Tree Search
# ---------------------------------------------------------------------------

def mcts_search(board, size, player, opponent, time_budget):
    """
    Búsqueda MCTS: simula partidas aleatorias para encontrar el mejor movimiento.
    
    Ciclo principal (mientras hay tiempo):
    1. SELECCIÓN: navega por el árbol usando UCT
    2. EXPANSIÓN: agrega un nuevo movimiento al árbol
    3. ROLLOUT: simula el resto del juego aleatoriamente
    4. BACKPROPAGATION: sube estadísticas hacia la raíz
    
    Args:
        board: tablero actual
        size: dimensión del tablero (11)
        player: quién somos
        opponent: quién es el rival
        time_budget: segundos disponibles para pensar
    
    Returns:
        (r, c): mejor movimiento encontrado
    """
    # Convertir tablero 2D a array lineal (más rápido)
    arr = [0] * (size * size)
    for r in range(size):
        row = board[r]; base = r * size
        for c in range(size):
            arr[base + c] = row[c]

    # Obtener celdas vacías
    empty_idx = [i for i in range(size * size) if arr[i] == 0]
    if not empty_idx:
        return None
    if len(empty_idx) == 1:
        i = empty_idx[0]; return (i // size, i % size)

    # Crear raíz del árbol
    empty_moves = [(i // size, i % size) for i in empty_idx]
    root = _Node(None, None, empty_moves, opponent)

    deadline = time.monotonic() + time_budget

    # ═══════════════════════════════════════════════════════════════
    # BUCLE PRINCIPAL: Mientras hay tiempo, simular más partidas
    # ═══════════════════════════════════════════════════════════════
    while time.monotonic() < deadline:
        node     = root
        sim_arr  = arr[:]
        sim_set  = set(empty_idx)   # Celdas vacías disponibles
        cur      = player

        # ── FASE 1: SELECCIÓN ─────────────────────────────────────
        # Bajamos por el árbol usando UCT hasta nodo no completamente expandido
        while not node.untried and node.children:
            node = node.uct_child()
            r, c = node.move
            idx  = r * size + c
            sim_arr[idx] = cur
            sim_set.discard(idx)
            cur = opponent if cur == player else player

        # ── FASE 2: EXPANSIÓN ─────────────────────────────────────
        # Agregar un nuevo movimiento no probado
        if node.untried:
            ut = node.untried
            if len(ut) <= 8:
                # Pocos candidatos → evaluamos inteligentemente con Dijkstra
                opp_cur = opponent if cur == player else player
                bm, bs = None, -1e18
                for (r, c) in ut:
                    s = _expansion_score(sim_arr, size, cur, opp_cur, r, c)
                    if s > bs: bs = s; bm = (r, c)
                move = bm
            else:
                # Muchos candidatos → elegimos al azar (es más rápido)
                move = random.choice(ut)

            r, c  = move
            idx   = r * size + c
            sim_arr[idx] = cur
            sim_set.discard(idx)
            nxt   = opponent if cur == player else player

            # Expandir: agregar nuevo hijo al árbol
            child_untried = [(i // size, i % size) for i in sim_set]
            node = node.expand(move, child_untried, cur)
            cur  = nxt

        # ── FASE 3: ROLLOUT ───────────────────────────────────────
        # Simular aleatoriamente el resto del juego
        opp_cur = opponent if cur == player else player
        result  = fast_rollout(sim_arr, sim_set, cur, opp_cur, player)

        # ── FASE 4: BACKPROPAGATION ──────────────────────────────
        # Subir por el árbol actualizando estadísticas
        n = node
        while n is not None:
            n.backup(result if n.mover == player else 1.0 - result)
            n = n.parent

    # ═══════════════════════════════════════════════════════════════
    # FIN: Seleccionar el mejor movimiento (el más visitado)
    # ═══════════════════════════════════════════════════════════════
    if not root.children:
        i = random.choice(empty_idx); return (i // size, i % size)

    return max(root.children, key=lambda ch: ch.visits).move


# ---------------------------------------------------------------------------
# DARK MODE: Adivinar posición de piedras ocultas del oponente
# ---------------------------------------------------------------------------

def _determinize(board, size, player, opponent):
    """
    En modo 'dark', no ves todas las piedras del oponente.
    
    Esta función crea un tablero ficticio adivinando dónde están:
    - Contar cuántas piedras nuestras tenemos
    - Asumir que el oponente tiene aproximadamente lo mismo
    - Colocar las piedras ocultas estimadas aleatoriamente en celdas vacías
    
    Esto permite que MCTS analice un tablero completo simulado
    en lugar de quedarse con información incompleta.
    """
    bl = [list(row) for row in board]
    my    = sum(bl[r][c] == player   for r in range(size) for c in range(size))
    known = sum(bl[r][c] == opponent for r in range(size) for c in range(size))
    hidden = max(0, my - known)
    empty  = [(r, c) for r in range(size) for c in range(size) if bl[r][c] == 0]
    if hidden and len(empty) >= hidden:
        for r, c in random.sample(empty, hidden):
            bl[r][c] = opponent
    return tuple(tuple(row) for row in bl)


# ---------------------------------------------------------------------------
# ESTRATEGIA PRINCIPAL: HexMaster
# ---------------------------------------------------------------------------

class HexMasterStrategy(Strategy):
    """
    Estrategia MCTS v3 para Hex 11×11.
    
    Flujo de decisión cada turno:
    1. Apertura: si es el inicio, juega en el centro sin gastar tiempo
    2. Dark mode: si no ves todas las piedras, adivina dónde están
    3. MCTS: busca el mejor movimiento usando simulaciones
    4. Fallback: si se acaba el tiempo, usa heurística rápida
    5. Validación: asegúrate de que el movimiento es válido
    """

    @property
    def name(self) -> str:
        return "HexMaster_julian_tania"

    def begin_game(self, config: GameConfig) -> None:
        """Inicializa variables al comenzar una partida."""
        self._size    = config.board_size
        self._player  = config.player
        self._opp     = config.opponent
        self._variant = config.variant  # "classic" o "dark"
        self._budget  = config.time_limit * 0.93  # Usar 93% del tiempo para ser seguro
        self._moves   = 0
        self._known_opp: set = set()  # Movimientos fallidos del oponente (dark mode)

    def on_move_result(self, move, success):
        """Registra si un movimiento fue exitoso (para dark mode)."""
        if not success:
            self._known_opp.add(move)
        self._moves += 1

    def play(self, board, last_move):
        """
        Decide el próximo movimiento.
        
        Proceso:
        1. Verificar celdas vacías
        2. Detección de apertura (jugar centro)
        3. Preparación dark mode (adivinar piedras)
        4. MCTS con presupuesto de tiempo
        5. Fallback a heurística rápida si no hay tiempo
        6. Validación del movimiento
        """
        t0   = time.monotonic()
        size = self._size

        # Paso 1: Obtener celdas vacías
        moves = empty_cells(board, size)
        if not moves:         return (0, 0)  # Error: no hay celdas
        if len(moves) == 1:   return moves[0]  # Única opción

        # Paso 2: Apertura (primeros movimientos)
        filled = size * size - len(moves)
        if filled <= 1:
            # En la apertura, evitamos análisis costoso
            # Jugamos en el centro
            mid = size // 2
            if board[mid][mid] == 0:
                return (mid, mid)
            # Si el centro está ocupado, cerca del centro
            for dr, dc in NEIGHBORS:
                nr, nc = mid + dr, mid + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == 0:
                    return (nr, nc)

        # Paso 3: Dark mode (si no ves todas las piedras)
        sim = (_determinize(board, size, self._player, self._opp)
               if self._variant == "dark" else board)

        # Paso 4: MCTS con presupuesto de tiempo
        budget = self._budget - (time.monotonic() - t0)
        if budget < 0.05:
            # Sin tiempo suficiente para MCTS → Fallback rápido
            # Usar heurística Dijkstra en muestra pequeña de movimientos
            arr = [0] * (size * size)
            for r in range(size):
                for c in range(size):
                    arr[r * size + c] = board[r][c]
            sample = random.sample(moves, min(8, len(moves)))
            bm, bs = None, -1e18
            for r, c in sample:
                s = _expansion_score(arr, size, self._player, self._opp, r, c)
                if s > bs: bs = s; bm = (r, c)
            return bm or random.choice(moves)

        # Ejecutar MCTS
        move = mcts_search(sim, size, self._player, self._opp, budget)

        # Paso 5: Validar contra tablero real
        # (en dark mode, la simulación puede ser diferente)
        if move is None or board[move[0]][move[1]] != 0:
            return random.choice(moves)
        return move