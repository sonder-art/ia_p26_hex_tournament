# Guia para Equipos

Guia paso a paso para configurar tu equipo y desarrollar una estrategia competitiva de Hex.

## Paso 1: Configuracion Inicial

```bash
# Forkea el repo y clona tu fork
git clone https://github.com/MCCH-7945/ia_p26_hex_tournament.git
cd ia_p26_hex_tournament

# Instala dependencias locales
pip install -r requirements.txt

# Instala Docker (necesario para los tiers MCTS)
# https://docs.docker.com/get-docker/
```

> **Docker es necesario.** Los tiers MCTS (MCTS_Tier_1 a MCTS_Tier_5) son binarios compilados que **solo corren dentro de Docker**. Sin Docker, solo puedes probar contra Random.

## Paso 2: Crea el Directorio de tu Equipo

```bash
cp -r estudiantes/_template estudiantes/nombre_de_tu_equipo
```

Estructura de tu directorio:

```
estudiantes/nombre_de_tu_equipo/
    strategy.py      # <-- UNICO ARCHIVO EVALUADO (todo tu codigo debe estar aqui)
    README.md        # <-- OBLIGATORIO (documenta tu estrategia)
    results/         # Se crea automaticamente para salidas locales
    ...              # Agrega lo que necesites (notebooks, scripts, datos)
```

**Importante:** Solo se ejecuta `strategy.py` durante el torneo. El framework importa unicamente ese archivo. Puedes tener notebooks de analisis, scripts auxiliares, tablas precomputadas y otros archivos en tu directorio para experimentacion local, pero **nada de eso sera accesible durante la evaluacion**. Si necesitas funciones auxiliares, defínelas dentro del mismo `strategy.py`.

## Paso 3: Estudia la Estrategia Random

Antes de escribir tu propia estrategia, estudia `strategies/random_strat.py`. Es la **unica estrategia con codigo fuente visible** — todos los demas tiers (MCTS_Tier_1 a MCTS_Tier_5) son binarios compilados `.so` cuyo codigo no puedes ver.

```python
# strategies/random_strat.py — la estrategia mas simple posible
class RandomStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Random"

    def begin_game(self, config: GameConfig) -> None:
        self._size = config.board_size

    def play(self, board, last_move):
        moves = empty_cells(board, self._size)
        return random.choice(moves)
```

Random simplemente elige una celda vacia al azar. Es el piso minimo — si tu estrategia no puede vencerla consistentemente, hay un bug.

La plantilla en `estudiantes/_template/strategy.py` ya tiene la estructura completa con docstrings detallados para cada metodo. Usala como punto de partida.

## Paso 4: Implementa tu Estrategia

Edita `estudiantes/nombre_de_tu_equipo/strategy.py`:

1. **Renombra** la clase y actualiza la propiedad `name` (debe ser unico: `"NombreEstrategia_nombreequipo"`)
2. **Implementa** tu logica en `play()` — recibe el tablero, devuelve `(row, col)`
3. **Usa** `begin_game(config)` para acceder a la informacion del juego
4. **Implementa** `on_move_result(move, success)` para rastrear colisiones en dark mode

**Tu estrategia debe funcionar para ambas variantes: classic y dark.**

### Informacion disponible en `GameConfig`

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `config.board_size` | `int` | Lado del tablero (11) |
| `config.variant` | `str` | `"classic"` o `"dark"` |
| `config.initial_board` | `tuple[tuple[int,...],...]` | Tablero inicial |
| `config.player` | `int` | Tu jugador: 1 (Negro) o 2 (Blanco) |
| `config.opponent` | `int` | Numero del oponente |
| `config.time_limit` | `float` | Segundos maximos por jugada (15) |

### Metodos de tu estrategia

```python
class MiEstrategia(Strategy):
    @property
    def name(self) -> str:
        return "MiEstrategia_mi_equipo"   # nombre unico

    def begin_game(self, config: GameConfig) -> None:
        # Se llama una vez al inicio de cada partida.
        # Guarda la configuracion y haz precomputaciones aqui.
        # NO consume tu presupuesto de tiempo.
        self._size = config.board_size
        self._player = config.player
        self._variant = config.variant
        self._time_limit = config.time_limit

    def on_move_result(self, move, success):
        # Se llama despues de cada play().
        # success=True: tu piedra se coloco.
        # success=False: colision (dark mode) — perdiste el turno,
        #   pero ahora ves la piedra oculta del oponente.
        pass

    def play(self, board, last_move):
        # board[r][c]: 0=vacio, 1=Negro, 2=Blanco
        # last_move: (row, col) del oponente, o None (siempre None en dark)
        # Devuelve (row, col) de una celda vacia
        moves = empty_cells(board, self._size)
        return moves[0]  # reemplaza con tu logica
```

### Utilidades disponibles

```python
from hex_game import (
    get_neighbors,          # (r, c, size) -> [(nr, nc), ...]
    check_winner,           # (board, size) -> 0, 1, or 2
    shortest_path_distance, # (board, size, player) -> int (Dijkstra)
    empty_cells,            # (board, size) -> [(r, c), ...]
    render_board,           # (board, size) -> str
    NEIGHBORS,              # [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
)
```

**Ejemplos:**
```python
dist = shortest_path_distance(board, 11, player=1)     # distancia Dijkstra para Negro
nbrs = get_neighbors(3, 5, 11)                          # vecinos de (3,5)
winner = check_winner(board, 11)                         # 0=nadie, 1=Negro, 2=Blanco
```

## Paso 5: Prueba Localmente (sin Docker)

Sin Docker solo esta disponible la estrategia Random. Empieza aqui para verificar que tu estrategia funciona.

```bash
# Prueba rapida contra Random (5 partidas, verbose muestra el tablero)
python3 experiment.py --black "MiEstrategia_mi_equipo" --white "Random" --num-games 5 --verbose

# Variante dark contra Random
python3 experiment.py --black "MiEstrategia_mi_equipo" --white "Random" --variant dark --num-games 5 --verbose

# Torneo rapido local (solo Random disponible)
python3 run_all.py
```

Si tu estrategia no vence a Random consistentemente, revisa tu logica antes de continuar.

## Paso 6: Prueba con Docker (contra tiers MCTS)

Los tiers MCTS son binarios compilados para Linux x86_64 que **solo funcionan dentro de Docker**. Aqui es donde realmente mides tu estrategia.

### Experimento individual contra un tier

```bash
# Contra MCTS_Tier_1 (facil)
docker compose run experiment \
  python experiment.py --black "MiEstrategia_mi_equipo" --white "MCTS_Tier_1" \
  --num-games 5 --verbose

# Contra MCTS_Tier_3 (dificil)
docker compose run experiment \
  python experiment.py --black "MiEstrategia_mi_equipo" --white "MCTS_Tier_3" \
  --num-games 5 --verbose

# Variante dark contra un tier
docker compose run experiment \
  python experiment.py --black "MiEstrategia_mi_equipo" --white "MCTS_Tier_2" \
  --variant dark --num-games 3 --verbose
```

### Tambien puedes usar variables de entorno

```bash
BLACK=MiEstrategia_mi_equipo WHITE=MCTS_Tier_3 docker compose run experiment
BLACK=MiEstrategia_mi_equipo WHITE=MCTS_Tier_4 VARIANT=dark docker compose run experiment
```

### Torneo completo: tu equipo vs todos los defaults

```bash
# Tu equipo contra Random + MCTS_Tier_1..5, ambas variantes, liga
TEAM=mi_equipo docker compose up team-tournament
```

Este es el comando mas importante — simula exactamente lo que pasara en la evaluacion real, pero solo con tu equipo y los 6 modelos de referencia.

### Progresion sugerida de pruebas

1. Vence a Random local (sin Docker) — validacion basica
2. Vence a MCTS_Tier_1 en Docker — primer tier real
3. Prueba classic Y dark por separado — asegura que ambas variantes funcionan
4. Corre `team-tournament` — simula la evaluacion completa
5. Sube de tier: T2, T3, T4, T5 — cada uno mas dificil

## Paso 7: Documenta y Entrega

### Escribe tu README

Reemplaza el `README.md` de tu directorio con tu propia documentacion. Debe explicar:

- **Algoritmo**: que tecnica(s) usaste (MCTS, minimax, heuristicas, etc.)
- **Dark mode**: como maneja tu estrategia el fog of war
- **Decisiones de diseno**: que trade-offs hiciste y por que
- **Resultados**: contra que tiers lograste ganar en tus pruebas locales

### Abre un Pull Request

```bash
git add estudiantes/mi_equipo/strategy.py estudiantes/mi_equipo/README.md
git commit -m "add strategy mi_equipo"
git push origin mi_equipo
```

Abre un **Pull Request** de tu branch hacia `main`.

**Tu PR debe contener:**
- `estudiantes/<tu_equipo>/strategy.py` — **obligatorio** (se evalua automaticamente)
- `estudiantes/<tu_equipo>/README.md` — **obligatorio** (documenta tu trabajo)
- Opcionalmente: notebooks, scripts, datos en tu directorio (no seran evaluados)

**NO incluyas:**
- Cambios a archivos fuera de `estudiantes/<tu_equipo>/`
- Archivos grandes (`.pkl`, `.npy`, modelos)
- Resultados (`results/`)

## Opciones de `run_all.py`

```bash
python3 run_all.py                          # rapido (classic, 4 games/pair)
python3 run_all.py --official               # ambas variantes, liga, 4 games/pair
python3 run_all.py --team mi_equipo         # solo tu equipo vs defaults
python3 run_all.py --real                   # evaluacion (10 games/pair)
python3 run_all.py --real --num-games 20    # evaluacion, 20 games/pair
```

## Servicios Docker disponibles

```bash
docker compose up tournament          # Torneo oficial (ambas variantes, liga)
docker compose up real-tournament     # Evaluacion real (10 games/pair)
TEAM=mi_equipo docker compose up team-tournament  # Solo tu equipo vs defaults
docker compose run experiment python experiment.py --help  # Ver todas las opciones
```

## Errores Comunes

### Tu estrategia no aparece
- Verifica: `estudiantes/<tu_equipo>/strategy.py` (nombre exacto).
- Tu clase debe heredar de `Strategy`.
- El directorio **no** debe empezar con `_`.

### Timeout
- El timeout es **estricto**: si `play()` tarda mas de 15 segundos, tu turno se salta.
- Usa `time.monotonic()` para controlar tu presupuesto (deja un margen de ~15%):
  ```python
  import time
  t0 = time.monotonic()
  while time.monotonic() - t0 < self._time_limit * 0.85:
      # una iteracion de MCTS
      ...
  ```

### Movimiento invalido
- `play()` debe devolver `(row, col)` de una celda vacia.
- Celda ocupada o fuera de rango = turno saltado (no pierdes la partida, pero el oponente juega).

### Funciona en classic pero falla en dark
- En dark **solo ves tus piedras** + las descubiertas por colision.
- `last_move` es siempre `None`.
- Implementa `on_move_result()` para rastrear colisiones.
- Las celdas "vacias" pueden tener piedras ocultas.
- **Prueba ambas variantes antes de entregar.**

### Los tiers MCTS no cargan
- Los tiers MCTS son binarios `.so` compilados para Linux x86_64.
- **Solo funcionan dentro de Docker.** Si corres fuera de Docker, solo Random estara disponible.
- Usa `docker compose run experiment ...` para probar contra tiers.

### ImportError
- Solo `numpy` + stdlib. No importes `scipy`, `pandas`, `sklearn`, etc.
- Puedes importar: `from strategy import Strategy, GameConfig` y funciones de `hex_game`.

## Ideas para tu Estrategia

1. **MCTS basico** — Monte Carlo Tree Search con UCT. Usa `time.monotonic()` para el presupuesto.
2. **Rollouts informados** — En vez de rollouts aleatorios, sesga hacia celdas que reducen tu distancia mas corta.
3. **Heuristica dual** — Combina tu `shortest_path_distance` con la del oponente.
4. **Puentes virtuales** — Dos piedras separadas por un gap que el oponente no puede bloquear. Prioriza completarlos.
5. **Early cutoff** — Corta rollouts antes del final y evalua con heuristica de distancia.
6. **Transposition table** — Guarda posiciones evaluadas para reutilizar entre iteraciones.
7. **Determinizacion (dark)** — Estima piedras ocultas del oponente, colocalas aleatoriamente, corre MCTS sobre ese "mundo posible".
8. **ISMCTS (dark)** — Information Set MCTS: mantiene un arbol sobre conjuntos de informacion.
9. **Exploracion de colisiones (dark)** — Colisionar revela informacion. Juega deliberadamente donde sospechas piedras ocultas.

## Consejos

- **Empieza simple.** La plantilla ya tiene la estructura completa. Construye sobre eso.
- **Vence a Random primero.** Si no puedes, tu estrategia tiene un bug.
- **Prueba ambas variantes.** Classic y dark se evaluan por separado. Tu calificacion depende de ambas.
- **Controla tu tiempo.** 15 segundos por jugada es generoso, pero estricto. Usa `time.monotonic()`.
- **Usa los tiers como benchmark.** Si vences a MCTS_Tier_3 (dificil), vas bien.
- **Estudia las utilidades.** `shortest_path_distance` y `get_neighbors` son herramientas poderosas.
- **Dark mode es diferente.** Necesitas razonar sobre informacion oculta. Determinizacion e ISMCTS son tecnicas clave.
