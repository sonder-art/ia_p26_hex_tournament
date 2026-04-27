# FogBridge_MALIK_RUBEN

## Resumen

La estrategia final del equipo vive en [strategy.py](/Users/malikcorverachoi/Documents/New project/estudiantes/MALIK_RUBEN/strategy.py) y se llama **`FogBridge_MALIK_RUBEN`**.

El archivo implementa una estrategia **unificada** para las dos variantes del torneo:

- `classic`: informacion perfecta
- `dark`: fog of war con colisiones

La idea central fue dejar de forzar el mismo algoritmo en ambos modos. `classic` y `dark` comparten utilidades geométricas de Hex, pero el problema que resuelven es distinto:

- en `classic` conviene búsqueda adversarial sobre un tablero completo
- en `dark` conviene razonar sobre estados ocultos y riesgo de colisión

Por eso el `strategy.py` final separa explícitamente ambas capas.

## Arquitectura General

La estrategia tiene dos módulos principales:

1. **Motor `classic`**
2. **Módulo `dark` con belief state**

Ambos comparten:

- representación de tablero `11x11`
- utilidades de vecinos hexagonales
- evaluación por distancias tipo Dijkstra
- detección de puentes
- heurísticas posicionales centradas en el eje objetivo

## Modo Classic

### Idea

En `classic`, el tablero es completamente observable, así que usamos un motor determinista con búsqueda adversarial.

La tubería es:

1. `opening_book_move()`
2. `candidate_moves()`
3. `iterative deepening`
4. `alpha-beta`
5. `tabla de transposición`
6. `evaluate()`

### 1. Opening book

La apertura es pequeña y geométrica, no una biblioteca teórica profunda.

Busca:

- tomar el centro cuando es razonable
- responder al centro con diagonales fuertes
- avanzar en el eje propio
- bloquear amenazas tempranas cerca de los bordes fuente/meta

### 2. Candidate generation

No evaluamos todas las casillas con el mismo peso.

`candidate_moves()` usa:

- ruta mínima propia
- ruta mínima rival
- segunda ruta rival
- vecinos de piedras ya colocadas como red de seguridad

La detección de pertenencia a una ruta usa condición bidireccional:

- `fwd[r][c] + bwd[r][c] - cost(r,c) <= md + slack`

Además:

- el slack del rival es adaptativo
- se considera también la segunda ruta rival en el mismo grafo bloqueado

Eso hace que el shortlist tenga más sentido táctico que una selección solo local.

### 3. Evaluación

`evaluate()` combina cuatro señales:

- **path score**: diferencia de distancia mínima a conexión
- **second-route score**: calidad relativa de la segunda ruta
- **bridge score**: puentes activos
- **position score**: centralidad y estructura

La jerarquía actual es:

- `path` con peso dominante
- `second route` como desempate fuerte
- `bridges`
- `position`

Con esto la estrategia prioriza:

- progreso real hacia su eje
- capacidad de cortar el corredor rival
- robustez posicional

### 4. Alpha-beta + TT

La búsqueda usa:

- `iterative deepening`
- poda `alpha-beta`
- `TranspositionTable`
- `Zobrist hashing`

Detalles importantes:

- el tablero interno usa `try/finally` alrededor de cada `place()/undo()`
- esto evita corrupción del estado si se agota el tiempo
- la TT se indexa por `(zhash, root_player)`

### 5. Presupuesto de tiempo

El framework da `15s` por jugada, pero la estrategia reserva margen.

En `begin_game()`:

- `self._time_limit = config.time_limit * 0.88`

Esto deja colchón para:

- overhead del proceso
- serialización
- variación del entorno Docker

## Modo Dark

## Idea

En `dark`, un `0` no significa “vacío real”, sino **“aparentemente vacío”**.

Por eso en esta variante no usamos minimax directamente sobre el tablero observado. El módulo de `dark` trata el problema como uno de **estado oculto**:

- mantenemos lo que sabemos
- estimamos lo que no sabemos
- sampleamos tableros completos plausibles
- evaluamos jugadas sobre esas determinizaciones

### 1. BeliefState

La clase `BeliefState` mantiene:

- `confirmed_rival`: celdas rivales reveladas por colisión
- `n_succeeded`: jugadas nuestras exitosas
- `n_collisions`: jugadas nuestras que chocaron
- `last_rival_heat`: mapa de calor para piedras ocultas rivales

Su propósito es separar:

- conocimiento seguro
- incertidumbre residual

### 2. Estimación de piedras ocultas

`estimate_n_hidden()` usa:

- cuántos intentos propios llevamos
- si movemos primero o segundo
- cuántas piedras rivales ya confirmamos

La idea es aproximar cuántos turnos rivales debieron ocurrir y restar lo ya revelado.

No es un conteo perfecto, pero funciona como una cota superior útil para muestrear estados plausibles.

### 3. Rival heat

`update_rival_heat()` corre Dijkstra desde la perspectiva del rival sobre la vista parcial actual.

Las celdas que parecen más relevantes para sus caminos cortos reciben más peso. Ese calor se usa después para samplear dónde podrían estar sus piedras ocultas.

### 4. Determinización

`sample_determinization()`:

1. copia la vista parcial
2. toma celdas inciertas
3. samplea `n_hidden` posiciones ponderadas por `rival_heat`
4. coloca ahí piedras rivales ocultas hipotéticas

Eso produce un tablero completo plausible.

### 5. Selección de jugada en dark

`_dark_best_move()`:

1. genera candidatas sobre la vista parcial
2. crea múltiples determinizaciones
3. para cada candidata:
   - penaliza si en una determinización la casilla ya estaba ocupada
   - si estaba libre, evalúa el estado resultante con `evaluate()`
4. promedia scores
5. elige la mejor

No es ISMCTS formal, pero sí una forma práctica de determinización con costo controlado.

### 6. Apertura en dark

La apertura en `dark` es más conservadora que en `classic`.

Regla actual:

- si somos primer jugador, preferimos el centro
- si somos segundo jugador, preferimos diagonales fuertes del centro

La motivación es simple:

- evitar colisiones triviales
- construir estructura central útil desde temprano

## Integración con el Framework

La estrategia está alineada con la interfaz real del repo:

- hereda de `Strategy`
- usa `config.time_limit`
- recibe `board` como `tuple[tuple[int,...], ...]`
- convierte internamente la vista a `list[list[int]]` mutable con `_to_grid()`

Métodos implementados:

- `begin_game(config)`
- `play(board, last_move)`
- `on_move_result(move, success)`
- `end_game(board, winner, your_player)`

Además:

- [strategy_2.py](/Users/malikcorverachoi/Documents/New project/estudiantes/MALIK_RUBEN/strategy_2.py) quedó como **wrapper de compatibilidad** para el runner local
- la lógica real vive solo en [strategy.py](/Users/malikcorverachoi/Documents/New project/estudiantes/MALIK_RUBEN/strategy.py)

## Decisiones de Diseño

### Por qué separar classic y dark

Porque el tipo de incertidumbre es distinto:

- en `classic`, el árbol de juego es observable
- en `dark`, el árbol observable no coincide con el estado real

Usar el mismo motor para ambos modos empeoraba más de lo que ayudaba.

### Por qué no MCTS puro en todo

Se consideró, pero se descartó como estrategia única porque:

- `classic` ya tenía muy buena estructura para evaluación determinista
- `dark` pedía manejo explícito de información oculta
- un único MCTS para ambos modos agregaba complejidad de tuning con poco tiempo

El compromiso final fue:

- `classic`: búsqueda determinista fuerte
- `dark`: determinización con evaluación agregada

### Por qué no usar aprendizaje

No se usó ML ni RL porque:

- el reglamento lo prohíbe
- queríamos una estrategia 100% algorítmica

## Pruebas Realizadas

Pruebas verificadas localmente y/o en Docker:

| Escenario | Resultado |
|---|---|
| `classic` vs `Random` | gana `1/1` |
| `dark` vs `Random` | gana `1/1` |
| `dark` vs `MCTS_Tier_5` | gana `1/1` |

Notas:

- la corrida de `classic` contra `MCTS_Tier_5` se lanzó pero no quedó cerrada en la sesión de trabajo
- por honestidad, no la reportamos como resultado confirmado

## Comandos Útiles

### Probar localmente contra Random

```bash
python3 experiment.py --black "FogBridge_MALIK_RUBEN" --white "Random" --team MALIK_RUBEN --variant classic --num-games 1
python3 experiment.py --black "FogBridge_MALIK_RUBEN" --white "Random" --team MALIK_RUBEN --variant dark --num-games 1
```

### Probar en Docker contra `MCTS_Tier_5`

```bash
docker compose run --rm experiment \
  python experiment.py \
  --black "FogBridge_MALIK_RUBEN" \
  --white "MCTS_Tier_5" \
  --team MALIK_RUBEN \
  --board-size 11 \
  --variant dark \
  --num-games 1 \
  --seed 42 \
  --move-timeout 15 \
  --verbose
```

```bash
docker compose run --rm experiment \
  python experiment.py \
  --black "FogBridge_MALIK_RUBEN" \
  --white "MCTS_Tier_5" \
  --team MALIK_RUBEN \
  --board-size 11 \
  --variant classic \
  --num-games 1 \
  --seed 42 \
  --move-timeout 15 \
  --verbose
```

## Limitaciones Conocidas

- En `dark`, la calidad depende mucho de la estimación de piedras ocultas.
- La determinización no modela explícitamente colisiones del rival.
- En `classic`, el costo por nodo sigue siendo alto por el número de Dijkstra.
- Todavía falta una batería más grande de matches contra `Tier_4` y `Tier_5` en ambas variantes para tener una lectura estadística más estable.

## Cierre

La versión final ya no intenta ser una sola idea forzada para todo Hex.

El resultado es un `strategy.py` con dos módulos compatibles pero especializados:

- `classic`: motor adversarial fuerte y estructurado
- `dark`: belief state + determinización pragmática

Ese fue el diseño que mejor equilibró:

- fuerza
- claridad
- costo computacional
- compatibilidad real con el framework del torneo
