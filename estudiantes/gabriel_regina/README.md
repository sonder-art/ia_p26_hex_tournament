# Estrategia: gabriel_regina (V9 / V3)

## Algoritmo principal: MCTS + RAVE con paralelización en raíz e ISMCTS-lite en dark

La estrategia usa **Monte Carlo Tree Search (MCTS)** como núcleo, extendido con **RAVE** (*Rapid Action Value Estimation*) para aprovechar mejor la información de los rollouts y converger más rápido a buenas decisiones. Sobre esa base se añaden tácticas locales (bridges, plantillas de borde, ladder avoidance), un perfil de parámetros distinto por variante, y para dark un esquema **ISMCTS-lite** con múltiples determinizaciones más una penalización de votos por probabilidad de colisión.

---

## Componentes clave

### 1. UCT-RAVE (selección de nodos)

La fórmula de selección mezcla el valor UCT estándar con el valor RAVE del movimiento:

```
score = (1 - β) * UCT + β * RAVE
β = min(sqrt(RAVE_K / (3*visits + RAVE_K)), RAVE_BLEND)
```

Con `RAVE_K = 400` y `RAVE_BLEND = 0.8` en classic. RAVE acumula estadísticas de todos los rollouts donde se jugó un movimiento (sin importar en qué turno), lo que permite estimar la bondad de movimientos no visitados directamente, acelerando la exploración en tableros grandes.

**Perfiles separados por variante** (`_VARIANT_PARAMS`): classic mantiene el tuning estable (UCT_C=1.2, RAVE_K=400, RAVE_BLEND=0.8, CUTOFF_FILL=0.65, NEIGHBOR_P=0.75); dark usa un perfil más conservador (UCT_C=1.1, RAVE_K=300, RAVE_BLEND=0.75, CUTOFF_FILL=0.62, NEIGHBOR_P=0.72) que en pruebas locales rindió 23/24 partidas en dark. `begin_game()` muta los globales antes de crear el pool de workers para que cada subproceso herede los valores correctos por `fork`.

### 2. Paralelización en raíz (*root parallelization*)

Se lanzan **3 procesos worker** adicionales con `multiprocessing` (fork), cada uno ejecutando MCTS independiente sobre el mismo estado raíz. Al terminar, se **suman los conteos de visitas** de todos los procesos (votación por mayoría ponderada) y se elige el movimiento con más votos totales. Esto cuadruplica el número de simulaciones por turno sin necesidad de sincronización.

### 3. Rollout rápido con tácticas locales y bias

Los rollouts no son puramente aleatorios. Antes de los sesgos estocásticos hay un pipeline de prioridades tácticas: cada vez que el rollout decide la siguiente jugada, se evalúan en orden:

1. **Save-bridge** (`_check_save_bridge`): si el oponente acaba de amenazar uno de mis puentes virtuales (dos piezas mías separadas por un gap), juego inmediatamente la celda *carrier* que mantiene la conexión.
2. **Break-bridge** (`_check_break_bridge`): si detecto un puente virtual del oponente, juego una de las celdas carrier para forzarlo a defender.
3. **Plantilla de borde 4-3-2** (`_check_edge_template`, `EDGE_TEMPLATES_P1/P2`): si el oponente arma una amenaza de borde clásica (dos piezas suyas alineadas hacia la fila/columna terminal), bloqueo la respuesta canónica.
4. **Ladder avoidance**: cuando las dos últimas jugadas forman una escalera (movimiento alineado en fila o columna), con probabilidad 5% rompo la escalera con una respuesta perpendicular en lugar de seguirla.

Si nada de lo anterior dispara, se aplican los sesgos estocásticos:

- **Bias de vecindad (75% en classic, 72% en dark)**: con esa probabilidad el siguiente movimiento se elige entre los vecinos de la última celda jugada, imitando el patrón real de cadenas en Hex.
- **Bias direccional (20%)**: con probabilidad 0.20 se muestrean 5 celdas aleatorias y se elige la más avanzada en la dirección ganadora del jugador actual (filas para jugador 1, columnas para jugador 2).
- **Aleatorio** en los casos restantes.

### 4. Evaluación suave (*soft eval*) al corte

Los rollouts se interrumpen cuando el tablero alcanza el **65% de llenado** (`CUTOFF_FILL = 0.65`). En vez de un resultado binario, se usa una **evaluación continua** vía función sigmoide sobre la diferencia de distancias Dijkstra:

```
eval = sigmoid((dist_oponente - dist_propia) * 0.8)
```

Se incluye un ajuste de tempo: si el oponente tiene el turno, su distancia se reduce en 1 (refleja la ventaja de mover primero). Esto da señales de entrenamiento más ricas que gano/pierdo.

### 5. Restricción de candidatos + FPU con Dijkstra bidireccional

En lugar de explorar las ~121 celdas vacías, la expansión solo considera celdas en un **radio de 2** alrededor de piezas existentes (vecindad del tablero ocupado). Para la raíz, los candidatos se ordenan mediante **Dijkstra bidireccional**: las celdas que pertenecen al camino mínimo actual del jugador se colocan al frente de la lista de movimientos a explorar (*First Play Urgency*), priorizando las jugadas más prometedoras.

Adicionalmente:

- **Filtro de celdas muertas** (`_is_dead_cell`): celdas vacías cuyos vecinos son *todos* del oponente (no aportan a ninguna conexión propia) se descartan de la expansión.
- **Bridge-aware Dijkstra** (`_bridge_aware_dijkstra`, `_bridge_cost`): variante de Dijkstra donde cruzar un vacío que forma un puente virtual entre dos piezas propias cuesta 0 en vez de 1. Refleja que esos vacíos son tácticamente equivalentes a piezas conectadas.

### 6. Reutilización del árbol (*tree reuse*)

En la variante **classic**, entre turnos el árbol no se descarta: se desciende 2 niveles (mi último movimiento → movimiento del oponente) para recuperar el subárbol relevante y reutilizar todas las simulaciones previas.

### 7. Tabla de transposición (solo classic)

Se mantiene una tabla hash `{hash_tablero: (visits, wins)}` con un cap de 50 visitas por entrada (`TRANS_CAP`) para evitar priors obsoletos. Cuando se expande un nodo ya visto, se inicializa con las estadísticas guardadas. Los priors se inyectan **una sola vez por nodo** (`child.visits == 0`) para evitar double-counting.

**Deshabilitada en dark**: cada determinización modifica los mismos hashes de tablero con piezas hipotéticas distintas, lo que contamina los priors entre muestras. En dark `self._trans_table = None`.

### 8. Atajos tácticos antes de MCTS

Antes de lanzar el MCTS, en cada turno se comprueban dos atajos baratos sobre las celdas vacías:

- **Victoria inmediata**: si alguna celda hace `check_winner == self._player`, se juega.
- **Bloqueo inmediato**: si alguna celda permitiría al oponente ganar en el siguiente turno, se juega para bloquear.

Si el deadline se agota durante estos chequeos, se devuelve un **fallback greedy** (`_greedy_fallback`) precomputado con Dijkstra bidireccional: una celda que está sobre el camino mínimo del jugador. Esto garantiza que aunque el budget se queme antes de tiempo, la estrategia nunca devuelve un movimiento absurdo.

### 9. Libro de apertura (classic)

Para 11x11 hay un pequeño libro de apertura hardcodeado (`_OPENING_BOOK`):

- Como **Negro** (primer movimiento): `(1, 9)` (cerca del centro-superior-derecha).
- Como **Blanco** respondiendo a aperturas comunes (`(5,5)`, `(5,4)`, `(5,6)`, `(1,9)`, `(3,7)`, `(7,3)`).

El libro se mantiene **deliberadamente conservador**: en una iteración previa expandir el libro con "(5,5) a todo" causó dos derrotas (vs Tier_4 y Tier_2), así que solo se conservan las entradas que validó V2. El libro está deshabilitado en dark.

### 10. Gestión de tiempo adaptativa

- `TIME_BUDGET = 0.93`, ajustado dinámicamente: `budget = min(0.97, 0.93 + 0.06/(1 + move_count*0.25))`. Las primeras jugadas (más críticas) reciben hasta 97% del budget; las siguientes se acercan al 93%.
- **`SAFETY_TAIL = 0.20`** (200 ms): margen duro restado del `time_limit` para garantizar que `play()` retorna antes del timeout estricto del referee.
- Los workers reciben `worker_duration = duration - 0.15` para dejar margen al `map_async.get()`.

---

## Manejo de la variante Dark (fog of war)

En Dark Hex el tablero del oponente es parcialmente invisible. La estrategia usa **ISMCTS-lite vía determinización múltiple**:

1. **Movimientos fallidos propios**: si un movimiento que intenté colocar falló, significa que el oponente ya tenía esa celda. Se guardan en `_hidden_opp` (y se cuentan en `_collision_count`) y se restauran en el tablero en cada turno.
2. **Estimación de piezas ocultas restantes**: corregida para considerar colisiones y color del jugador. `estimated_hidden = max(0, (my_moves + collisions) - known_opp - offset)` donde `offset = 1 if player == 1 else 0` (el segundo jugador ya vio una de las piezas del primero al inicio).
3. **NUM_DETERMINIZATIONS = 4 muestras únicas** (`NUM_WORKERS + 1`): el thread principal usa una determinización y los 3 workers usan las otras tres. Cada worker corre MCTS independiente sobre su propio "mundo posible"; al final se suman los conteos de visita ponderados (ver punto 6).
4. **Colocación ponderada (35/30/35)**: las piezas estimadas se distribuyen sobre celdas vacías (excluyendo `_failed_moves`) con un peso compuesto:
   - **35% centro** (`max(1, size - dist_manhattan_al_centro)`): los jugadores hábiles tienden al centro.
   - **30% eje ganador del oponente** (`max(1, size - |r-center|)` si el oponente conecta filas, análogo en columnas): el oponente prioriza su dirección de victoria.
   - **35% cluster cerca de piedras ocultas reveladas** (`max(1, size - dist_min_a_known_opp)`): observado en Tier_2 dark — el rival tiende a continuar clusters cerca de donde ya colocó.
5. **Sin tree reuse en dark**: `self._root = None` en cada turno porque el tablero determinizado cambia entre jugadas.
6. **Penalización de votos por P(colisión)** (variance robustness): cada candidato se escala por la fracción de determinizaciones en las que aparecía vacío. Un candidato que aparece vacío en pocas muestras probablemente oculta una piedra rival; bajar su voto reduce las jugadas-kamikaze sin descartar el movimiento (con piso `1/N` para no eliminarlo si todas las muestras lo tachan).
7. **Tabla de transposición deshabilitada**: cada determinización contamina los priors de los mismos hashes (ver §7).

---

## Decisiones de diseño importantes

| Decisión | Razón |
|---|---|
| RAVE sobre UCT puro | Hex tiene alta correlación entre "jugar X en algún momento" y "X es bueno ahora"; RAVE explota eso |
| Root parallelization sobre tree parallelization | Más simple, sin locks; funcional con `fork` en Linux |
| Soft eval con Dijkstra | Las distancias de camino mínimo son la heurística más informativa en Hex; cortar rollouts al 65% y evaluar con sigmoid reduce ruido |
| Bias de vecindad en rollouts | Las cadenas locales dominan la táctica en Hex; rollouts puramente aleatorios son demasiado ruidosos |
| Tácticas locales (bridges, edge templates) en rollout | Forzar respuestas tácticas correctas reduce drásticamente la varianza del rollout y produce evaluaciones más realistas |
| Parámetros distintos por variante | Dark tiene mucho más ruido que classic; bajar UCT_C/RAVE_K y CUTOFF_FILL favorece explotación más estable bajo incertidumbre |
| Determinización múltiple (ISMCTS-lite) sobre IS-MCTS completo | IS-MCTS canónico es caro; 4 determinizaciones independientes con votación ponderada captura buena parte del beneficio sin la complejidad |
| Penalización de votos por P(empty) en dark | Una jugada que sólo "se ve vacía" en pocos mundos posibles probablemente colisiona; multiplicar el voto por la fracción empty es una corrección de varianza barata |
| Pesos de cluster en determinización (35%) | Observación empírica en Tier_2 dark: el rival tiende a continuar clusters cerca de donde ya colocó |
| Libro de apertura conservador | Expandir el libro causó dos derrotas en pruebas locales (Tier_4, Tier_2); sólo se conservan las entradas validadas por V2 |
| TT solo en classic | En dark, cada determinización modifica los mismos hashes con piezas hipotéticas distintas → priors contaminados |
| Filtro de celdas muertas | Celdas con todos los vecinos del oponente no aportan a ninguna conexión propia; filtrarlas reduce el branching factor sin perder información |
| Atajos de victoria/bloqueo + greedy fallback | Garantizan que la estrategia juega una jugada sensata aunque el budget se agote; nunca devuelve un movimiento absurdo |

---

## Resultados de pruebas locales

| Oponente | Resultado |
|---|---|
| Random | Victoria consistente (>95%) |
| MCTS_Tier_1 | Victoria consistente |
| MCTS_Tier_2 | Victoria consistente |
| MCTS_Tier_3 | Victoria consistente |
| MCTS_Tier_4+ | Victoria consistente |

---

## Versiones anteriores y evolución

- **V1–V3**: MCTS básico, sin RAVE, sin paralelismo.
- **V4**: Se añadió RAVE y tabla de transposición.
- **V5**: Fix de apertura, tiempo dinámico, FPU bidireccional, tree reuse.
- **V6**: Root parallelization (4 cores), soft eval con sigmoide continua en lugar de evaluación binaria.
- **V7**: Tácticas locales en rollout (save-bridge, break-bridge, plantillas de borde 4-3-2), ladder avoidance, filtro de celdas muertas, bridge-aware Dijkstra, atajos de victoria/bloqueo + greedy fallback, fix de double-counting en TT, validación de hash en tree reuse.
- **V8**: Determinización dark más sofisticada (peso por centro + eje ganador), corrección de la fórmula de `estimated_hidden` para considerar colisiones y color del jugador.
- **V9 (actual)**: Perfiles de parámetros separados por variante (classic vs dark), TT deshabilitada en dark, **ISMCTS-lite** con 4 determinizaciones distribuidas entre main + 3 workers, **penalización de votos por P(empty)** en dark, prior de determinización con término de cluster (35/30/35), libro de apertura conservador validado.

---

## Estructura del código (`strategy.py`)

| Sección | Funciones clave |
|---|---|
| Parámetros y perfiles | `_VARIANT_PARAMS`, `BRIDGE_PATTERNS`, `BRIDGE_ENDPOINTS`, `EDGE_TEMPLATES_P1/P2`, `_OPENING_BOOK` |
| Estructuras de datos | `_EmptyPool` (selección/eliminación O(1)), `_Node` (UCT-RAVE) |
| Heurísticas Dijkstra | `_soft_eval`, `_bridge_aware_dijkstra`, `_bridge_cost`, `_full_dijkstra`, `_fpu_order`, `_greedy_fallback` |
| Filtros y candidatos | `_is_dead_cell`, `_neighborhood_empties`, `_candidates` |
| Tácticas de rollout | `_check_save_bridge`, `_check_break_bridge`, `_check_edge_template`, `_fast_rollout` |
| MCTS | `_mcts_select`, `_mcts_expand`, `_mcts_backpropagate`, `_build_root` |
| Paralelización | `_worker_run` (root parallelization standalone) |
| Estrategia | `MiEstrategiaV3` (`name = "gabriel_regina_v3"`): `begin_game`, `play`, `on_move_result`, `_opening_book_move`, `_reset_tree`, `_descend_root`, `_determinize` |
