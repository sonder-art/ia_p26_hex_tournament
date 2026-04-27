# EquipoTochiaXalli — Estrategia de Hex

## Resumen

Esta estrategia implementa **MCTS (Monte Carlo Tree Search) con UCT** combinado con varias heurísticas específicas de Hex para superar a un MCTS vanilla con más presupuesto de iteraciones.

## Algoritmo

### 1. Núcleo: MCTS con UCT

El motor central es MCTS clásico con las cuatro fases (Selección, Expansión, Simulación, Retropropagación). La selección usa la fórmula UCT (módulo 17, UCB1 aplicado a árboles):

```
UCT(v) = Q(v)/N(v) + c * sqrt(ln(N_padre) / N(v))
```

Usamos `c = 1.2` (ligeramente menor que sqrt(2) ≈ 1.41) porque en Hex 11×11 con tiempo limitado preferimos un poco más de explotación. La elección final del movimiento es por número de visitas del hijo (no por Q/N), criterio estándar más robusto.

### 2. Rollouts informados por distancia

En lugar de rollouts puramente aleatorios, con probabilidad 0.85 elegimos jugadas que minimizan la distancia más corta de Dijkstra del jugador a turno (usando `shortest_path_distance` del motor) penalizando ligeramente con la del oponente. Esto hace que las simulaciones terminen mucho más rápido y con mejor señal (clave para que el árbol crezca con información útil dentro del presupuesto de 15 segundos).

### 3. Bridge pattern (puente Hex)

Antes de invocar MCTS, revisamos si el oponente acaba de invadir uno de los dos *carriers* de un puente nuestro (patrón clásico: dos piedras propias separadas en diagonal-hex con dos celdas vacías que las conectan). Si es así, jugamos el otro carrier para mantener la conexión virtual. Esto no requiere búsqueda y captura una intuición fundamental de Hex.

### 4. Win-in-one / Block-in-one

Antes del MCTS chequeamos si hay jugada ganadora inmediata o si tenemos que bloquear una jugada ganadora del oponente. Esto evita que el MCTS desperdicie iteraciones en posiciones ya decididas tácticamente.

### 5. Tree reuse

Entre llamadas a `play()` reciclamos el subárbol correspondiente al estado actual. Esto duplica efectivamente el presupuesto de iteraciones útiles a partir del segundo turno.

### 6. Apertura

Si es la primera jugada del juego (tablero vacío), jugamos el centro `(5, 5)` directamente — heurística estándar en Hex 11×11.

## Variante Dark (fog of war)

En `dark`:

- `last_move` siempre es `None` y solo vemos nuestras propias piedras.
- Tratamos la vista del jugador como una **determinización**: corremos MCTS sobre el tablero parcialmente observable como si fuera el verdadero (las celdas que aparecen como 0 las tratamos como vacías reales — lo cual estadísticamente es razonable porque no sabemos dónde están las piedras ocultas del oponente).
- Cuando ocurre una colisión (`on_move_result(success=False)`), guardamos esa celda como conocida-ocupada-por-oponente e invalidamos el árbol (la información cambió).
- Como `last_move` es `None`, el bridge save no se activa en dark — pero todo lo demás sí funciona.

Esta es una determinización simple pero efectiva: ISMCTS o muestreo de mundos múltiples sería más sofisticado, pero la determinización fija es más estable bajo restricciones de tiempo.

## Decisiones de diseño y trade-offs

| Decisión | Por qué |
|---|---|
| `c = 1.2` en UCT | Hex 11×11 tiene mucho branching → preferimos más explotación dentro del budget |
| Rollouts informados (no puros) | Rollouts puros son demasiado ruidosos en 11×11 — Dijkstra da señal direccional |
| Selección final por visitas, no por Q/N | Más robusta a outliers (un hijo con 1 victoria de 1 visita NO debe ser el preferido) |
| 85% del tiempo | Margen de seguridad para evitar timeouts |
| Determinización fija en dark | Más simple y estable que ISMCTS bajo presupuesto limitado |
| Solo `numpy` + stdlib | Cumple restricciones del torneo |

## Resultados de pruebas locales



## Estructura de archivos

- `strategy.py` — implementación completa (único archivo evaluado)
- `README.md` — este documento