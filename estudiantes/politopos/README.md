# Politopos — Estrategia MCTS para Hex 11×11

## Descripción general

**Politopos** es una estrategia basada en Monte Carlo Tree Search (MCTS) diseñada para jugar Hex en un tablero de 11×11. Funciona en ambas variantes del torneo: classic (información perfecta) y dark (fog of war). Su núcleo es un motor de búsqueda paralelo que corre cuatro procesos de MCTS simultáneamente y combina sus resultados por votación, lo que permite explorar el árbol de juego con aproximadamente cuatro veces más iteraciones que una implementación de un solo proceso en el mismo presupuesto de tiempo.

---

## Algoritmo base: MCTS con UCT-RAVE

El algoritmo principal es MCTS estándar con dos mejoras sobre el UCT básico.

Cada iteración del árbol sigue cuatro fases. En la **selección**, se desciende el árbol desde la raíz eligiendo siempre el hijo con mayor valor UCT-RAVE hasta llegar a un nodo con movimientos sin explorar. En la **expansión**, se saca el mejor movimiento no explorado del nodo actual y se crea un nodo hijo. En el **rollout**, se simula el resto de la partida desde ese nodo hasta alcanzar un ganador o el límite de profundidad. Finalmente, en la **backpropagación**, el resultado se propaga hacia arriba por toda la rama, actualizando visitas, victorias y estadísticas RAVE en cada nodo.

### UCT-RAVE

El valor de un nodo se calcula combinando la fórmula UCT estándar con el estimador RAVE (Rapid Action Value Estimation):

```
UCT  = (wins / visits) + C * sqrt(ln(parent_visits) / visits)
RAVE = rave_wins[move] / rave_visits[move]   # del nodo padre
valor = (1 - β) * UCT + β * RAVE
```

donde `β = rave_visits / (rave_visits + visits)`, lo que hace que RAVE domine al inicio (cuando hay pocas visitas reales) y UCT domine al final (cuando la información directa es suficiente). El parámetro de exploración es `C = 1.25`.

RAVE aprovecha la propiedad de Hex de que los movimientos buenos en una parte del tablero tienden a ser buenos en otras partes también: si un movimiento apareció frecuentemente en rollouts ganadores del subárbol, se considera prometedor incluso antes de haberlo probado directamente.

---

## Motor paralelo

Al inicio de cada turno, el proceso principal lanza tres procesos worker adicionales usando `multiprocessing.Pool` con contexto `fork`. Los cuatro procesos (main + 3 workers) corren MCTS independientemente sobre el mismo tablero, cada uno con una semilla aleatoria distinta para diversificar la exploración. Al terminar el tiempo disponible, se suman los conteos de visitas de todos los procesos y se elige el movimiento con más votos en total.

```
movimiento elegido = argmax( visits_main[m] + visits_w1[m] + visits_w2[m] + visits_w3[m] )
```

Este esquema se llama *root parallelization* y es el método estándar para paralelizar MCTS sin necesidad de sincronización entre procesos durante la búsqueda.

---

## Heurísticas de dominio

### Evaluación de movimientos (`_score`)

Cada casilla vacía recibe un puntaje que combina dos factores:

```
score(r, c, player) = 0.7 * progreso + 0.3 * (vecinos_propios / 6)
```

El **progreso** mide qué tan central es la casilla en la dirección del objetivo del jugador: para el jugador 1 (Negro, conecta filas), es la proximidad a la fila central; para el jugador 2 (Blanco, conecta columnas), la proximidad a la columna central. La **conectividad** cuenta cuántos vecinos directos ya tienen una piedra propia, normalizado sobre el máximo posible de 6 vecinos en Hex.

### Evaluación compuesta (`_combined`)

Durante los rollouts, la selección de movimiento usa una métrica que considera tanto el beneficio propio como el daño al oponente:

```
_combined(r, c, player) = score(r, c, player) - 0.7 * score(r, c, opponent)
```

El coeficiente 0.7 en el término del oponente incentiva juego simultáneamente ofensivo y defensivo.

### Rollout con muestreo inteligente

En cada paso del rollout, en lugar de evaluar todas las casillas libres, se muestrean 6 candidatos al azar y se elige el de mayor `_combined`. Esto da un equilibrio entre velocidad (pocos candidatos evaluados) y calidad (no se juega completamente al azar).

### Evaluación de fin de rollout

Cuando el rollout alcanza el límite de profundidad sin ganador, se evalúa la posición mediante Dijkstra. Se calcula la distancia mínima de cada jugador a su borde objetivo (con peso 0 para casillas propias y 1 para casillas vacías, ignorando casillas del oponente). Gana quien tenga menor distancia.

```python
d1 = _distance(board, size, root_player)
d2 = _distance(board, size, opponent)
result = 1.0 if d1 <= d2 else 0.0
```

---

## Poda de candidatos

En lugar de considerar todas las casillas vacías del tablero (~80-100 en partidas medianas), Politopos solo expande movimientos en una vecindad de radio 2 alrededor de las piezas existentes. Esto típicamente reduce los candidatos a 20-30 casillas, haciendo que el árbol explore más densamente la zona relevante del tablero.

En la apertura, cuando no hay piezas en el tablero, los candidatos se restringen a la zona central (radio `size // 3` desde el centro), evitando movimientos inútiles en los bordes al inicio de la partida.

El orden de exploración dentro de los candidatos sigue el `_score` heurístico: los movimientos con mayor puntaje se intentan primero, lo que acelera la convergencia del árbol hacia ramas prometedoras.

---

## Tree reuse (variante classic)

En la variante classic, el árbol construido en el turno anterior no se descarta. En lugar de eso, al inicio del turno siguiente se desciende dos niveles: primero al nodo correspondiente al movimiento que se jugó, y luego al nodo correspondiente a la respuesta del oponente. El subárbol resultante se reutiliza como nueva raíz, conservando todas las estadísticas acumuladas (visitas, victorias, RAVE) de los turnos anteriores.

Esto permite que el MCTS comience cada turno con información ya explorada en lugar de partir desde cero, lo que equivale a obtener iteraciones "gratuitas" de turnos anteriores.

En la variante dark, este mecanismo se deshabilita porque el tablero observado puede cambiar impredeciblemente por colisiones, haciendo que las estadísticas del árbol anterior no sean confiables.

---

## Detección anticipada

Antes de entrar al MCTS, Politopos revisa en O(n) si existe un movimiento que gana inmediatamente o que bloquea una victoria inmediata del oponente. Si se detecta alguno de los dos casos, se retorna ese movimiento directamente sin necesidad de búsqueda.

---

## Gestión eficiente de casillas (`_EmptyPool`)

La clase `_EmptyPool` mantiene el conjunto de casillas vacías con eliminación en O(1) usando la técnica de swap-and-pop: al eliminar un elemento, se intercambia con el último de la lista y se actualiza un diccionario de posiciones. Esto reemplaza la operación `list.remove()` estándar de Python, que es O(n) y se llamaría miles de veces por segundo durante los rollouts.

---

## Dark mode (Fog of War)

En dark mode, cada jugador solo ve sus propias piedras y las que ha descubierto por colisión. Politopos maneja esto con un enfoque de mundos posibles.

### Conocimiento acumulado

Se mantiene `_known_board`, un tablero con toda la información cierta: las propias piedras y las del oponente descubiertas por colisión. Las colisiones también se registran en `_collision_history` para evitar reintentar esas casillas.

### Generación de mundos posibles

Al inicio de cada turno, se generan `NUM_BELIEF_SAMPLES = 4` versiones plausibles del tablero completo. Cada mundo parte de `_known_board` y añade piedras del oponente en casillas aleatorias desconocidas. La cantidad de piedras a colocar se estima como:

```
hidden_opp = max(0, my_stone_count - known_opp_count)
```

Esto asume que el oponente ha jugado aproximadamente los mismos turnos que el propio jugador, lo que es correcto en Hex dado que los turnos se alternan.

### MCTS sobre múltiples mundos

El tiempo disponible se divide entre los 4 mundos. En cada uno se corre MCTS de forma independiente, obteniendo el mejor movimiento para ese mundo. El movimiento elegido al final es el que apareció como mejor con mayor frecuencia entre todos los mundos.

Esta técnica, conocida como determinización múltiple o Information Set MCTS simplificado, permite razonar sobre la incertidumbre sin necesidad de modelar explícitamente el espacio de creencias.

---

## Parámetros

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `EXPLORATION_CONSTANT` | 1.25 | Balance exploración-explotación en UCT |
| `MAX_SIMULATION_DEPTH` | 100 | Profundidad máxima de rollout |
| `NUM_BELIEF_SAMPLES` | 4 | Mundos posibles en dark mode |
| `NUM_WORKERS` | 3 | Procesos worker adicionales |
| `TIME_BUDGET` | 0.92 | Fracción del tiempo límite utilizada |
| `EXPAND_RADIUS` | 2 | Radio de vecindad para poda de candidatos |

---

## Decisiones de diseño

**Paralelismo con 3 workers.** La ganancia más grande en calidad de juego viene de hacer más iteraciones MCTS en el mismo tiempo. Root parallelization con 3 workers adicionales multiplica las iteraciones por ~4 sin necesidad de sincronización entre procesos — cada worker corre MCTS independiente y al final se suman votos. Se eligió 3 workers (no más) porque el torneo tiene 4 núcleos compartidos con el referee y el oponente; usar más podría causar contención.

**Tree reuse de 2 niveles.** Reutilizar el árbol completo sería incorrecto porque el oponente pudo haber jugado en un nodo que no se exploró. Descartar el árbol completamente desperdicia trabajo valioso. Descender exactamente 2 niveles (mi movimiento → respuesta del oponente) es el balance correcto: conserva las estadísticas acumuladas de la rama que efectivamente ocurrió, sin asumir información incorrecta.

**Radio 2 para poda de candidatos.** Radio 1 es demasiado restrictivo y pierde movimientos de bridge virtual (conexiones a distancia 2 que son fundamentales en Hex). Radio 3 incluye demasiadas casillas y diluye la exploración. Radio 2 captura el espacio táctico relevante (~20-30 candidatos) sin explorar casillas claramente irrelevantes.

**Coeficiente 0.7 en `_combined`.** La penalización al progreso del oponente no debe ser igual al beneficio propio (coeficiente 1.0) porque eso llevaría a juego puramente defensivo. Tampoco debe ser cero porque ignoraría las amenazas del oponente. El valor 0.7 equilibra agresividad y defensa: se busca conectar, pero se interrumpe activamente al oponente cuando sus movimientos son muy amenazantes.

**Estimación de piedras ocultas en dark mode.** Una alternativa sencilla sería asignar al oponente un porcentaje fijo de las casillas vacías, pero esto genera mundos estadísticamente imposibles (por ejemplo, 50 piedras oponentes cuando solo han pasado 5 turnos). Estimar `hidden_opp = my_stone_count - known_opp` produce mundos plausibles y hace que el MCTS razone sobre posiciones que realmente podrían ocurrir.

**Muestreo de 6 candidatos en rollout.** Evaluar todas las casillas libres en cada paso del rollout sería O(n) por paso y haría los rollouts demasiado lentos. Evaluar solo 1 sería casi aleatorio. Con 6 candidatos se mantiene un tiempo por rollout razonable mientras la heurística `_combined` tiene suficientes opciones para identificar movimientos claramente malos y evitarlos.

---

## Resultados de pruebas locales

Las pruebas se realizaron localmente usando `experiment.py` contra Random (disponible sin Docker) antes de la evaluación oficial.

**Variante classic — vs Random (10 partidas):**

| Como Negro | Como Blanco | Total |
|---|---|---|
| 5/5 victorias | 5/5 victorias | 10/10 |

**Variante dark — vs Random (10 partidas):**

| Como Negro | Como Blanco | Total |
|---|---|---|
| 4/5 victorias | 5/5 victorias | 9/10 |

La única derrota en dark mode ocurrió jugando como Negro en una partida donde se produjo una serie de colisiones tempranas que reveló la posición del oponente demasiado tarde para bloquear su conexión.

**Tiers MCTS (requieren Docker — pendiente de evaluación oficial):**

Las pruebas contra MCTS_Tier_1 a Tier_5 no pudieron completarse localmente antes de la entrega por limitaciones de entorno. El análisis del algoritmo sugiere que la combinación de motor paralelo (~4× iteraciones) y poda de candidatos (~3-5× exploración más densa) debería ser suficiente para superar Tier_1 y Tier_2 con consistencia. Tier_3 representa la barrera de calidad esperada dada la complejidad del juego y el presupuesto de tiempo de 15 segundos por movimiento.
