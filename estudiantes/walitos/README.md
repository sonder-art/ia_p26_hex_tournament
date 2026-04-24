# Team Template

Copia este directorio para crear tu equipo:

```bash
cp -r estudiantes/_template estudiantes/nombre_de_tu_equipo
```

## Estructura

```
estudiantes/nombre_de_tu_equipo/
    strategy.py      # <-- UNICO ARCHIVO EVALUADO (todo tu codigo aqui)
    README.md        # <-- OBLIGATORIO (reemplaza este con tu documentacion)
    ...              # Agrega lo que necesites (notebooks, scripts, datos)
```

## Inicio rapido

1. **Estudia** `strategies/random_strat.py` — es la unica estrategia con codigo fuente visible. Todos los tiers MCTS son binarios compilados `.so` cuyo codigo no puedes ver. Random es tu punto de partida y tu piso minimo.

2. **Edita** `strategy.py` — cambia el nombre de la clase, la propiedad `name`, e implementa `play()`.

3. **Prueba** contra Random (sin Docker):
   ```bash
   python3 experiment.py --black "TuNombre_equipo" --white "Random" --num-games 5 --verbose
   ```

4. **Prueba** contra tiers MCTS (requiere Docker):
   ```bash
   docker compose run experiment \
     python experiment.py --black "TuNombre_equipo" --white "MCTS_Tier_3" \
     --num-games 5 --verbose
   ```

5. **Corre un torneo completo** (tu equipo vs todos los defaults):
   ```bash
   TEAM=nombre_de_tu_equipo docker compose up team-tournament
   ```

6. **Documenta** tu estrategia reemplazando este README.

7. **Entrega** via Pull Request.

## Tu README debe explicar

**Reemplaza este archivo** con tu propia documentacion. Debe incluir:

- **Algoritmo**: que tecnica(s) usaste (MCTS, minimax, heuristicas, etc.)
- **Dark mode**: como maneja tu estrategia el fog of war (determinizacion, ISMCTS, tracking de colisiones, etc.)
- **Decisiones de diseno**: que trade-offs hiciste y por que
- **Resultados**: contra que tiers lograste ganar en tus pruebas locales

## Reglas

- Tu estrategia debe funcionar para **ambas** variantes: `classic` y `dark` (fog of war).
- **15 segundos** max por jugada (timeout estricto — exceder = turno saltado, no pierdes la partida).
- **8 GB** de memoria, **numpy + stdlib** unicamente.
- El `name` debe ser unico: `"NombreEstrategia_nombreequipo"`.
- Tu estrategia corre en un **proceso separado** — no puedes acceder al motor del juego ni al oponente.

## Calificacion

Tu calificacion depende de cuantos de los 6 modelos de referencia vences en los standings combinados (classic + dark):

| Modelos vencidos | Calificacion |
|------------------|-------------|
| 0 | 0 |
| 1 | 5 |
| 2 | 6 |
| 3 | 7 |
| 4 | 8 |
| 5 | 9 |
| 6 | 10 |

"Vencer" = tus puntos totales ≥ puntos totales del modelo (empate te favorece).
Top 3 estudiantes por puntos totales = 10 automatico.

## Documentacion completa

- **[Guia para equipos](../../docs/team_guide.md)** — paso a paso detallado
- **[Reglas del torneo](../../docs/rules.md)** — mecanica, restricciones, juego limpio
- **[README principal](../../README.md)** — overview, calificacion con ejemplo, estructura del repo

## Utilidades disponibles

```python
from hex_game import (
    get_neighbors,          # (r, c, size) -> lista de vecinos
    check_winner,           # (board, size) -> 0, 1, o 2
    shortest_path_distance, # (board, size, player) -> int (Dijkstra)
    empty_cells,            # (board, size) -> [(r, c), ...]
    render_board,           # (board, size) -> str
    NEIGHBORS,              # [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
)
```
