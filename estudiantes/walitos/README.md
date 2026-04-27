# walitos — Hex con MCTS+PUCT y red de resistencia eléctrica

**Estrategia:** `HexMCTSPUCT_walitos`
**Variantes:** Classic + Dark
**Tablero:** 11×11
**Dependencias:** `numpy` + stdlib

---

## Tabla de contenidos

1. [Algoritmo](#1-algoritmo)
2. [Dark mode](#2-dark-mode)
3. [Decisiones de diseño](#3-decisiones-de-diseño)
4. [Resultados](#4-resultados)

---

## 1. Algoritmo

### 1.1 Visión general

La estrategia combina **Monte Carlo Tree Search (MCTS) con selección PUCT**
(la fórmula de AlphaZero) sobre un **prior calculado por una red de
resistencia eléctrica**. Los rollouts se aceleran con una estructura
**Union-Find** que detecta victorias en tiempo cuasi-constante por movida.

El pipeline general en cada turno es:

```
play(board, last_move):
    1. Verificar movida ganadora inmediata → jugarla
    2. Verificar amenaza ganadora del rival → bloquearla
    3. Calcular prior de resistencia eléctrica (cacheado por hash)
    4. Dispatch a _play_classic o _play_dark según variante
    5. Filtro de seguridad: garantizar movida legal
```

Las dos variantes comparten el mismo motor MCTS pero difieren
significativamente en sus etapas previas (ver secciones 1.7 y 2).

### 1.2 ¿Por qué MCTS en lugar de minimax?

El espacio de búsqueda de Hex es prohibitivo para minimax:

- **Branching factor:** hasta 121 movidas iniciales
- **Profundidad:** hasta ~120 movidas por partida
- **Estados:** 3¹²¹ ≈ 10⁵⁷

Minimax con poda alpha-beta requeriría una función de evaluación muy
precisa para podar agresivamente, y desarrollar esa función para Hex es
en sí mismo un problema complejo. MCTS resuelve esto delegando la
evaluación a estadísticas sobre muchos rollouts aleatorios, sin necesidad
de una función de evaluación explícita.

### 1.3 La fórmula PUCT

La selección estándar de UCB en MCTS es:

```
score = Q + C · sqrt(ln(N_padre) / N)
```

Esta fórmula no aprovecha información a priori sobre la calidad de las
movidas. Usamos en su lugar **PUCT** (Predictor + UCB), introducida en
AlphaGo Zero:

```
score(child) = Q(child) + C · P(child) · sqrt(N(parent)) / (1 + N(child))
```

Donde:

- `Q(child) = W/N` es el promedio de victorias en simulaciones a través
  de este hijo (componente de explotación)
- `P(child)` es el **prior**: un score a priori de qué tan buena se
  estima la movida antes de visitarla, calculado por la red de
  resistencia (sección 1.5)
- `C = 1.2` es el coeficiente de exploración (estándar en literatura)
- `N(parent)` y `N(child)` son los conteos de visitas

El segundo término decae con `N(child)`: al inicio el prior domina y
guía la exploración hacia movidas estratégicamente prometedoras; con
suficientes visitas el `Q` empírico toma el control. Esto es
matemáticamente análogo a Thompson sampling con prior bayesiano sobre
la value function.

### 1.4 First Play Urgency (FPU)

Un problema con UCB1 estándar: para nodos no visitados (`N=0`), el
término de exploración es infinito, lo que fuerza visitar todas las
hijas antes de profundizar en cualquiera. Con branching factor de 121,
esto es desastroso — se gastan todas las iteraciones en exploración
superficial.

**FPU** asigna a los nodos no visitados un valor virtual:

```python
fpu_value = parent_Q − FPU_REDUCTION   # FPU_REDUCTION = 0.20
```

Es decir, los hijos no visitados se evalúan como "ligeramente peores
que el promedio del padre". Esto permite **profundizar en las movidas
con prior alto** en lugar de explorar todas las hijas con peso igual.

Implementación en `Node.puct_score`:

```python
def puct_score(self, parent_N, C, fpu):
    u = C * self.prior * math.sqrt(parent_N)
    if self.N == 0:
        return fpu + u           # Nodo no visitado: usa FPU
    return self.W / self.N + u / (1 + self.N)
```

### 1.5 Prior de red de resistencia eléctrica

Esta es la parte conceptualmente más importante de la estrategia.
Pregunta fundamental: **dado un tablero, ¿qué celda vacía es más
valiosa para el jugador 1?**

#### Intuición física

Modelamos el tablero como un circuito eléctrico:

- Los bordes superior e inferior (para el jugador 1) son terminales con
  voltaje fijo: `V=1` arriba, `V=0` abajo
- Cada celda es un nodo cuya conductancia depende de su contenido:
  - **Celda propia:** `g = 10⁶` (cable casi perfecto)
  - **Celda vacía:** `g = 1` (resistencia normal)
  - **Celda del rival:** `g = 0` (aislante)
- Los aristas del grafo de Hex conectan celdas vecinas con conductancia
  igual al mínimo de las dos celdas

Cuando se aplica voltaje, la corriente fluye por el camino de menor
resistencia. **Las celdas con alta corriente son cuellos de botella
estratégicos** — son donde el flujo eléctrico (y por analogía, la
conexión ganadora) debe pasar.

#### Formulación matemática

Por las leyes de Kirchhoff, los voltajes satisfacen `A · V = b`
donde `A` es la matriz Laplaciana del grafo:

```
A[i,i] = Σⱼ g(i,j)         (suma de conductancias incidentes al nodo i)
A[i,j] = −g(i,j)           (negativo de la conductancia entre i y j)
```

Imponemos las condiciones de borde reescribiendo las filas SRC y SNK
como ecuaciones de identidad:

```python
A[SRC,:] = 0;   A[SRC,SRC] = 1;   b[SRC] = 1
A[SNK,:] = 0;   A[SNK,SNK] = 1;   b[SNK] = 0
```

Resolvemos con `numpy.linalg.lstsq` (robusto a singularidades
del Laplaciano que pueden ocurrir si hay celdas aisladas).

#### Cálculo del prior

Una vez tenemos los voltajes `V[i]` de cada celda, calculamos la
corriente local de cada celda vacía:

```python
prior[r,c] = Σⱼ |g(i,j) · (V[j] − V[i])|     # i = idx(r,c)
```

donde `j` itera sobre los vecinos de `(r,c)`. Normalizamos a `Σ prior = 1`
para obtener una distribución de probabilidad sobre celdas vacías.

#### ¿Por qué funciona?

Este prior captura **simultáneamente ofensiva y defensiva**:

- **Ofensiva:** las celdas con alta corriente extienden la conexión
  propia hacia los bordes
- **Defensiva:** las celdas con alta corriente del rival son donde
  bloquear su flujo (corta su conexión virtual)

Es una **generalización continua** de la heurística *two-distance* de
Anshelevich (que cuenta saltos de BFS al borde más lejano). El prior
de resistencia es estrictamente más informativo porque considera
**múltiples caminos paralelos** y los pondera por su capacidad
relativa, no solo el camino más corto.

#### Costo

El cálculo es `O(n³)` por la inversión de la matriz Laplaciana, donde
`n = size² + 2`. Para 11×11, esto son ~123 nodos → ~8ms por evaluación.
Cacheamos por hash del estado completo del tablero, así que mientras
el rival no juegue se reutiliza.

### 1.6 Rollouts con Union-Find

La parte más costosa de MCTS son los rollouts: simulaciones aleatorias
hasta el fin de partida. Una implementación naïve detecta el ganador
con DFS al final, costo `O(n²)` por movida.

Usamos **Union-Find** con path compression y union by rank
(`O(α(n))` amortizado, prácticamente constante) para mantener
componentes conexas durante el rollout:

- Mantenemos dos UF separados, uno por jugador
- Cada UF tiene nodos ficticios SRC y SNK conectados a las celdas de
  los bordes correspondientes
- Cada vez que se juega una piedra, unimos los componentes de sus
  vecinos del mismo color
- El juego termina cuando `uf.find(SRC) == uf.find(SNK)` para algún jugador

```python
class _UF:
    __slots__ = ['p', 'rk']  # path parents, ranks

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]   # Path compression
            x = self.p[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return
        if self.rk[px] < self.rk[py]:        # Union by rank
            px, py = py, px
        self.p[py] = px
        if self.rk[px] == self.rk[py]:
            self.rk[px] += 1
```

**Resultado medido:** ~2500 iteraciones de MCTS/segundo en 11×11,
comparado con ~250-500 con BFS/DFS. Esto se traduce directamente en
~10× más profundidad de búsqueda en el mismo tiempo.

#### Sesgo local en simulación

Los rollouts puramente aleatorios son demasiado divergentes — no
capturan que en Hex las movidas tienden a ser locales (extensiones de
cadenas existentes). Usamos un sesgo simple:

```python
if last_r >= 0 and random.random() < 0.30:
    # 30% de probabilidad: jugar adyacente a la última movida
    for nr, nc in _get_neighbors(last_r, last_c, size):
        if sim[nr][nc] == 0:
            move = (nr, nc); break
```

Sin costo computacional significativo, pero hace los rollouts más
realistas y mejora la calidad estadística de la evaluación.

### 1.7 Tree reuse correcto

Entre turnos podemos preservar la sub-árbol que corresponde a la
posición actual, en lugar de reconstruir desde cero. La sub-árbol
correcta está **dos niveles abajo** del root anterior:

```
       root (antes de mi movida)
      /  \
     A    B    ← mis posibles movidas
    / \   / \
   ...  ... ...  ← respuestas del rival
```

Si yo jugué A y el rival jugó X, el nuevo root debe ser el nodo
`A → X` — descenso de **dos niveles**, no uno.

Un bug temprano descendía solo un nivel:

```python
# BUG: descendía solo a nuestro hijo
return next((ch for ch in self.root.children
             if ch.move == self._our_last_move), None)
```

Esto dejaba como root un nodo cuyos hijos eran las movidas del rival.
MCTS retornaba una de esas movidas como propia → forfeit por movida
inválida (la celda ya estaba ocupada por el rival).

La versión corregida verifica que el camino completo exista:

```python
def _get_reused_root(self, opp_last_move):
    if self.root is None or self._our_last_move is None:
        return None
    our_node = next((ch for ch in self.root.children
                     if ch.move == self._our_last_move), None)
    if our_node is None: return None
    if opp_last_move is None: return None  # No fallback peligroso
    opp_node = next((ch for ch in our_node.children
                     if ch.move == opp_last_move), None)
    return opp_node  # None si no se exploró → árbol fresco
```

Si el rival jugó algo que MCTS no había explorado en el árbol previo,
construimos un árbol fresco. Esto es sub-óptimo pero seguro.

### 1.8 Filtro de seguridad final

Como red de seguridad, antes de retornar la movida elegida verificamos
que esté en el conjunto de movidas legales:

```python
move = ...   # Resultado de MCTS
if move not in set(legal):
    move = random.choice(legal) if legal else (0, 0)
return move
```

Esto previene forfeits incluso si:

- El opening book contiene una movida obsoleta
- Tree reuse retorna un nodo con movida ya jugada
- Cualquier otro bug introduce inconsistencias

No debería activarse en operación normal, pero garantiza robustez.

---

## 2. Dark mode

Dark Hex (Phantom Hex) es estructuralmente distinto de classic. Cada
jugador ve solo sus propias piedras; las del rival son invisibles a
menos que intentemos jugar en una celda ocupada (lo que provoca una
notificación de colisión).

Esto invalida muchas heurísticas de classic — el prior de resistencia,
por ejemplo, calcula sobre un tablero que no refleja la realidad.

### 2.1 Filosofía: NO modelar al rival

Probamos múltiples versiones de modelado del rival en dark (paranoid
greedy, IS-MCTS con determinización, two-distance Anshelevich,
multi-sample expected resistance). Detalles en sección 3.3. **Todas
fueron inferiores** al pipeline simplificado descrito en 2.2.

La razón empírica: modelar al rival con paranoid asume que juega
óptimamente contra nosotros, pero los Tier baselines juegan según su
propio plan (extender chain hacia los bordes), no contra nosotros. El
modelo paranoid predice movidas erróneas que producen movidas reactivas
en lugar de constructivas.

La estrategia final **no modela al rival** — confía en la apertura
ladder para establecer conexión virtual y usa MCTS sobre un espacio
restringido para resolver la endgame.

### 2.2 Pipeline en cuatro etapas

```python
def _play_dark(...):
    # Etapa 1: Defensa de puentes visibles
    if bridge_threats:
        return defense

    # Etapa 2: Bridge ladder opening (primeras 6 movidas)
    if own_count < 6 and ladder_intact:
        return ladder[own_count]

    # Etapa 3: Restricción estructural
    if own_stones:
        legal = [m for m in legal
                 if m in connected_cells | bridge_cells]

    # Etapa 4: MCTS+PUCT sobre legal restrictado
    return mcts_search(legal, ...)
```

### 2.3 Bridge ladder opening

**La idea clave del dark mode: el rival no puede atacar lo que no ve.**

Diseñamos una secuencia hardcoded de 6 piedras que cubren borde-a-borde,
donde cada par consecutivo forma un puente:

- **Player 1** (top→bottom):
  `(0,7) → (2,6) → (4,5) → (6,4) → (8,3) → (10,2)`
- **Player 2** (left→right):
  `(7,0) → (6,2) → (5,4) → (4,6) → (3,8) → (2,10)`

#### Verificación de que sí son puentes

Tomemos `(0,7)` y `(2,6)`:

- Vecinos de `(0,7)`: `(0,6), (0,8), (1,6), (1,7)`
- Vecinos de `(2,6)`: `(1,6), (1,7), (2,5), (2,7), (3,5), (3,6)`
- **Vecinos comunes:** `(1,6)` y `(1,7)`

Si en classic el rival jugara `(1,6)`, defenderíamos con `(1,7)` y
viceversa. Esto preserva la conexión virtual entre `(0,7)` y `(2,6)`
con costo de exactamente 2 movidas (1 ataque + 1 defensa).

#### En dark, el rival no las ve

Por lo tanto **no las puede atacar proactivamente**. Las 6 movidas de
la ladder establecen una conexión virtual completa borde-a-borde en
exactamente 6 movidas determinísticas — una eficiencia que es
imposible en classic.

#### Robustez ante captura accidental

La ladder solo se usa si las movidas anteriores siguen siendo nuestras:

```python
own_count = sum(1 for r in range(size) for c in range(size)
                if board_list[r][c] == player)
prev_intact = all(board_list[r][c] == player
                  for r, c in ladder[:own_count])
if prev_intact:
    next_move = ladder[own_count]
    if next_move in legal_set:
        return next_move
```

Si el rival captura accidentalmente una celda de la ladder (porque
estaba probando esa celda y le tocó), abandonamos la ladder y delegamos
a MCTS. Esto ocurre en ~10-20% de los juegos según observación empírica.

### 2.4 Restricción estructural

**Problema observado:** MCTS sin información del rival juega celdas
dispersas. En classic, las piedras visibles del rival actúan como
restricciones estructurales — los muros bloquean opciones malas, MCTS
converge hacia cadenas focalizadas. En dark, sin esa estructura visible,
MCTS sin restricción se dispersa en movidas inconexas.

Verificación empírica: corrimos 16 movidas en dark sin restricción
estructural, partiendo de tablero vacío. Resultado: cluster de 5×3 en
filas 3-7, sin tocar bordes (pierde por incapacidad de conectar).

**Solución:** filtrar movidas legales a:

```python
own_stones = [(r, c) for r, c in cells if board_list[r][c] == player]
if own_stones:
    connected_cells = set()
    for r, c in own_stones:
        for nr, nc in _get_neighbors(r, c, size):
            if board_list[nr][nc] == 0:
                connected_cells.add((nr, nc))

    bridge_cells = set(_bridge_build_moves(board_list, size, player))

    restricted = [m for m in legal if m in connected_cells | bridge_cells]
    if restricted:
        legal = restricted
```

Esto reemplaza la "estructura del rival" con una **estructura artificial
de chain building**. MCTS ahora elige solo entre extensiones legítimas
de la cadena, garantizando coherencia estructural.

Verificación empírica con la restricción: 16 movidas dark cubren filas
0-10 (todo el tablero), cadena coherente, conexión borde-a-borde.

### 2.5 Tree reuse imposible en dark

En classic, `last_move` del rival es visible y permite navegar el árbol
preservado. En dark, `last_move` es siempre `None` desde nuestra
perspectiva (no vemos al rival). Cada turno reconstruye el árbol desde
cero.

Esto es aceptable porque:

- El espacio de búsqueda restricto (sección 2.4) es pequeño (<20 movidas
  típicamente)
- MCTS converge en pocos miles de iteraciones, no decenas de miles
- Con 12s de tiempo por movida tenemos ~30k iteraciones, suficientes
  para convergencia

### 2.6 Manejo de colisiones

Cuando intentamos jugar en una celda ocupada por el rival, el motor
nos notifica vía `on_move_result(move, success=False)`. Registramos
estas celdas en `self.revealed_opponent`:

```python
def on_move_result(self, move, success):
    if not success:
        self.revealed_opponent.add(move)
```

Estas celdas se excluyen de las movidas legales en futuras evaluaciones:

```python
def _legal_moves(self, board):
    dead = _dead_cells(board, self.size)
    return [(r, c) for r in range(self.size) for c in range(self.size)
            if board[r][c] == 0
            and (r, c) not in self.revealed_opponent
            and (r, c) not in dead]
```

Aunque no usamos las colisiones para reconstruir explícitamente el
estado del rival (sería un modelado paranoid implícito), sí evitamos
intentar jugar dos veces en la misma celda ocupada.

---

## 3. Decisiones de diseño

### 3.1 Mantener el pipeline simple

La filosofía guía: **cada componente debe demostrar ganancia empírica
o ser eliminado**. Probamos extensiones sofisticadas y las descartamos
cuando introdujeron regresiones, incluso si teóricamente debían ayudar.

Componentes finales mantenidos en classic:

- Win/block check (ganancia obvia)
- Opening book (ganancia probada)
- MCTS+PUCT con prior de resistencia (núcleo)

Componentes finales en dark:

- Bridge defense
- Bridge ladder opening
- Filtro estructural
- MCTS+PUCT

### 3.2 Componentes descartados en classic

Implementamos pero descartamos:

#### Virtual connection (VC) search

**Idea:** búsqueda explícita de conexiones virtuales más allá de
bridges (ladders, edge templates, escaleras forzadas). Genera
"fortalezas" garantizadas — caminos que el rival no puede romper sin
importar qué juegue.

**Por qué se descartó:** la implementación tenía bugs sutiles que
ocasionalmente producían movidas ilegales o cadenas VC inválidas.
Cuando MCTS retornaba una celda señalada por el VC solver y la celda
ya estaba ocupada, causaba forfeit. El esfuerzo de debuggear excedía
la ganancia esperada.

#### Endgame alpha-beta solver

**Idea:** cuando quedan pocas celdas vacías (≤20), invocar búsqueda
exhaustiva alpha-beta en lugar de MCTS. En endgame el espacio es
manejable y la búsqueda exhaustiva debería encontrar la respuesta
óptima.

**Por qué se descartó:** el solver agotaba el tiempo en posiciones
donde MCTS encontraba la respuesta rápidamente. La condición de
terminación temprana (memoization, transposition tables) requería
más implementación de la disponible.

### 3.3 Aproximaciones descartadas en dark

| Aproximación | Idea | Resultado |
|--------------|------|-----------|
| **Paranoid greedy** | Colocar piedras "ocultas" del rival en sus celdas de mayor prior de resistencia (peor caso) | Cluster sin extender; perdía 0/5 vs Tier 2 |
| **IS-MCTS** | MCTS con determinización aleatoria por iteración, promediando sobre múltiples mundos posibles | Inconsistente; tree convergía mal |
| **Two-distance Anshelevich** | Heurística clásica `score = -(d_top + d_bot)` | Empate en celdas centrales — no diferenciaba dirección de extensión |
| **Two-distance max-based** | Variante `score = -(max(d_top,d_bot) + 0.1·min(d_top,d_bot))` para forzar extensión hacia el borde más lejano | Mejor que sum, pero todavía clusters al consolidar |
| **Multi-sample expected resistance** | Promediar prior sobre 4 determinizaciones para suavizar | Computacionalmente caro, sin mejora medible |
| **Threat detection** | Sumar prior del rival en celdas visibles para identificar amenazas | Causaba colisiones (jugábamos donde el rival ya estaba) |
| **Bonus de conectividad multiplicativo** | Bonus `1 + 0.4·adj_own` para favorecer celdas conectadas | Favorecía CENTRO del cluster sobre frontera → consolidaba en lugar de extender |
| **Bonus de conectividad binario** | Bonus fijo `1.25` si hay ≥1 vecino propio, sin escalar con número de vecinos | Mejor balance, pero todavía MCTS con tantos parámetros era inconsistente |

**Lección general:** combinar muchas heurísticas con pesos manuales
crea inconsistencias difíciles de calibrar. Todas las heurísticas
intentan optimizar el mismo objetivo (construir cadena ganadora) pero
tienen modos de falla diferentes. La composición de las fallas era
peor que la suma de las ganancias individuales.

La solución final (bridge ladder + filtro estructural + MCTS) tiene
**solo dos hyperparámetros** (las posiciones de la ladder y la
restricción estructural). Es robusta porque cada componente tiene una
función clara y aislada.

### 3.4 Trade-offs principales

#### Tiempo vs. precisión en MCTS

Con `time_limit = 12s`, sustraemos `0.5s` de margen para conversiones
finales (`deadline = monotonic + time_limit - 0.5`). Esto da ~30k
iteraciones en 11×11 — suficiente para convergencia razonable.

Podríamos haber gastado más tiempo en heurísticas determinísticas
(VC search, endgame solver), pero la inversión en MCTS más profundo
demostró ser más confiable.

#### Cache de prior vs. memoria

Cacheamos el prior de resistencia por hash del estado. El cache
acumula entradas a lo largo del juego pero se invalida en `end_game()`.
En la práctica, el cache rara vez excede ~30 entradas por partida
(un par por turno), uso de memoria insignificante.

#### Restricción estructural en dark vs. flexibilidad

La restricción a celdas conectadas/puente reduce dramáticamente el
espacio de búsqueda en dark. Trade-off: si la restricción es demasiado
agresiva, podríamos perder movidas estratégicas no-locales (por ejemplo,
crear una cabeza de puente lejana).

Solución: solo aplicamos la restricción cuando hay piedras propias.
La primera movida (apertura) tiene espacio completo. Después, la
ladder hardcoded toma control. Solo después de la ladder (o si la
ladder se rompe) entra la restricción estructural.

### 3.5 Determinismo y robustez

La estrategia es **determinista dado el mismo estado del tablero y la
misma semilla de `random`**. No tiene dependencias externas (no llama
a APIs, no usa GPUs, no carga modelos), así que su comportamiento es
reproducible.

Esto es importante para debugging: si una partida tiene un comportamiento
extraño, podemos reproducirla exactamente con el mismo seed y
diagnosticarla.

### 3.6 Cumplimiento con el reglamento

- **No usa ML/RL en runtime:** ningún componente aprendido se evalúa
  en runtime. La estrategia contiene solo constantes hardcoded
  (`C_PUCT=1.2`, `FPU=0.20`, posiciones de ladder, etc.).
- **No depende de bibliotecas externas:** solo `numpy` y la stdlib de
  Python.
- **Maneja correctamente movidas ilegales:** filtro de seguridad final
  garantiza que nunca se retorna una celda ocupada (sección 1.8).
- **Tiempo de respuesta acotado:** cada movida respeta el `time_limit`
  proporcionado en `GameConfig`, restando 0.5s de margen para
  conversiones finales.

---

## 4. Resultados

### 4.1 Evaluación contra todos los baselines

Evaluación en 48 partidas (4 por matchup, 6 oponentes × 2 variantes,
2 partidas como negro y 2 como blanco):

| Oponente    | Classic      | Dark         |
|-------------|--------------|--------------|
| Random      | **4/4** (100%) | **4/4** (100%) |
| MCTS_Tier_1 | **4/4** (100%) | **4/4** (100%) |
| MCTS_Tier_2 | **4/4** (100%) | 2/4 (50%)    |
| MCTS_Tier_3 | 3/4 (75%)    | 3/4 (75%)    |
| MCTS_Tier_4 | 3/4 (75%)    | **4/4** (100%) |
| MCTS_Tier_5 | **4/4** (100%) | 2/4 (50%)    |
| **Total**   | **22/24 (91.7%)** | **19/24 (79.2%)** |

**Win rate global: 41/48 = 85.4%.**

### 4.2 Tiers vencidos contundentemente

Definiendo "vencer" como ≥75% winrate (3 de 4 partidas):

**Classic:** vence Random, Tier 1, Tier 2, Tier 3, Tier 4, Tier 5 (los 6)
**Dark:** vence Random, Tier 1, Tier 3, Tier 4 (4 de 6)

Total de modelos vencidos: 10 de 12.

### 4.3 Análisis de partidas

- **Largo promedio:** 24-46 movidas según matchup, con promedio global
  de ~30 movidas. Esto indica que la estrategia construye conexiones
  en mid-game, no en endgame profundo.
- **Classic vs Tier 3 (75%):** las partidas duran 45.8 movidas en
  promedio (las más largas), sugiriendo que Tier 3 obliga a endgame
  pero MCTS resuelve correctamente la mayoría.
- **Dark vs Tier 4 (100%):** sorprendentemente fuerte — el bridge
  ladder + filtro estructural manejan bien la estrategia de Tier 4.
- **Dark vs Tier 2/5 (50%):** los puntos débiles. Hipótesis: estos
  baselines tienen una estrategia que ocasionalmente coloca piedras
  en celdas que rompen la apertura ladder por casualidad (no por
  diseño, ya que no la ven).