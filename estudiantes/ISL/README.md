# ISL Strategy

## Algoritmo
MCTS (Monte Carlo Tree Search) con heurística semántica basada en distancia Dijkstra.

## Componentes principales
- **Apertura**: jugada al centro del tablero
- **Quick win/block**: detecta victorias o bloqueos inmediatos en 1 jugada
- **MCTS con UCT**: selección via Upper Confidence Bound
- **Rollouts sesgados**: 70% mejor movimiento de muestra, 30% aleatorio
- **Ordenamiento de movimientos**: por distancia Dijkstra propia y rival

## Dark mode
Manejo de colisiones via `on_move_result` — registra celdas del oponente descubiertas.

## Resultados locales
- ✅ Vence a Random consistentemente
- ✅ Vence a MCTS_Tier_1
