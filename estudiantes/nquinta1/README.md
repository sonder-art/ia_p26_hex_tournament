# NQuinta_Strategy_v2 — Hex Agent

## Descripción

`NQuinta_Strategy_v2` es un agente heurístico para Hex 11x11 diseñado para competir en el torneo del curso.

La estrategia no usa búsqueda Monte Carlo completa, sino una evaluación heurística de cada movimiento legal basada en:

- reducción de la distancia de conexión propia
- aumento de la distancia de conexión del rival
- ocupación de casillas valiosas para bloquear al oponente
- preferencia por jugadas conectadas con piedras propias
- control del centro en etapas tempranas

## Idea principal

Para cada jugada legal, el agente simula:

1. cómo cambia su distancia mínima de conexión
2. cómo cambia la distancia mínima de conexión del rival
3. qué tan peligrosa sería esa casilla si la ocupara el oponente

Con esa información asigna un puntaje y elige la mejor jugada.

## Heurísticas usadas

- **Centro al inicio:** si el centro está libre, se prioriza.
- **Victoria inmediata:** si una jugada gana la partida, se toma de inmediato.
- **Bloqueo preventivo:** se favorecen casillas que también serían muy fuertes para el rival.
- **Conectividad local:** se premian jugadas adyacentes a piedras propias.
- **Balance ofensivo-defensivo:** se combina avance propio con contención del oponente.

## Archivo principal

- `estudiantes/nquinta1/strategy.py`

## Comentarios

Es una estrategia heurística ligera, pensada para mantenerse dentro del límite de tiempo por jugada y ofrecer un desempeño mejor que un agente aleatorio.
