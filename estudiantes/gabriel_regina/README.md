# Estrategia: gabriel_regina

## Algoritmo principal: MCTS + RAVE con paralelización en raíz

La estrategia usa **Monte Carlo Tree Search (MCTS)** como núcleo, extendido con **RAVE** (*Rapid Action Value Estimation*) para aprovechar mejor la información de los rollouts y converger más rápido a buenas decisiones.

---

## Componentes clave

### 1. UCT-RAVE (selección de nodos)

La fórmula de selección mezcla el valor UCT estándar con el valor RAVE del movimiento:

```
score = (1 - β) * UCT + β * RAVE
β = min(sqrt(RAVE_K / (3*visits + RAVE_K)), RAVE_BLEND)
```

Con `RAVE_K = 400` y `RAVE_BLEND = 0.8`. RAVE acumula estadísticas de todos los rollouts donde se jugó un movimiento (sin importar en qué turno), lo que permite estimar la bondad de movimientos no visitados directamente, acelerando la exploración en tableros grandes.

### 2. Paralelización en raíz (*root parallelization*)

Se lanzan **3 procesos worker** adicionales con `multiprocessing` (fork), cada uno ejecutando MCTS independiente sobre el mismo estado raíz. Al terminar, se **suman los conteos de visitas** de todos los procesos (votación por mayoría ponderada) y se elige el movimiento con más votos totales. Esto cuadruplica el número de simulaciones por turno sin necesidad de sincronización.

### 3. Rollout rápido con bias

Los rollouts no son puramente aleatorios; tienen dos sesgos:

- **Bias de vecindad (75%)**: con probabilidad 0.75 el siguiente movimiento se elige entre los vecinos de la última celda jugada, imitando el patrón real de cadenas en Hex.
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

### 6. Reutilización del árbol (*tree reuse*)

En la variante **classic**, entre turnos el árbol no se descarta: se desciende 2 niveles (mi último movimiento → movimiento del oponente) para recuperar el subárbol relevante y reutilizar todas las simulaciones previas.

### 7. Tabla de transposición

Se mantiene una tabla hash `{hash_tablero: (visits, wins)}` con un cap de 50 visitas por entrada para evitar priors obsoletos. Cuando se expande un nodo ya visto, se inicializa con las estadísticas guardadas.

---

## Manejo de la variante Dark (fog of war)

En Dark Hex el tablero del oponente es parcialmente invisible. La estrategia usa **determinización**:

1. **Movimientos fallidos propios**: si un movimiento que intenté colocar falló, significa que el oponente ya tenía esa celda. Se guardan en `_hidden_opp` y se restauran en el tablero en cada turno.
2. **Estimación de piezas ocultas restantes**: se estima la cantidad de piezas del oponente que aún no se conocen como `max(0, mis_movimientos - piezas_oponente_conocidas - 1)`.
3. **Colocación ponderada por distancia al centro**: las piezas estimadas se distribuyen en celdas vacías con probabilidad proporcional a la cercanía al centro (`peso = size - distancia_manhattan_al_centro`), ya que jugadores hábiles tienden a ocupar el centro.
4. El árbol MCTS **se reinicia en cada turno** en dark mode (no hay tree reuse) porque el tablero determinizado cambia en cada jugada.

---

## Decisiones de diseño importantes

| Decisión | Razón |
|---|---|
| RAVE sobre UCT puro | Hex tiene alta correlación entre "jugar X en algún momento" y "X es bueno ahora"; RAVE explota eso |
| Root parallelization sobre tree parallelization | Más simple, sin locks; funcional con `fork` en Linux |
| Soft eval con Dijkstra | Las distancias de camino mínimo son la heurística más informativa en Hex; cortar rollouts al 65% y evaluar con sigmoid reduce ruido |
| Bias de vecindad en rollouts | Las cadenas locales dominan la táctica en Hex; rollouts puramente aleatorios son demasiado ruidosos |
| Determinización en dark | IS-MCTS (Information Set MCTS) completo es costoso; determinización simple es una aproximación práctica que funciona bien en la práctica |
| No hay apertura hardcodeada | El centro (5,5) se prioriza automáticamente por FPU; hardcodear aperturas generó problemas en versiones anteriores |

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
- **V6 (actual)**: Root parallelization (4 cores), soft eval con sigmoide continua en lugar de evaluación binaria.
