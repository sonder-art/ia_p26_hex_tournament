# Compus Hybrid MCTS

## Resumen

`CompusHybridMCTS_compus` implementa una estrategia hibrida para Hex 11x11 sin redes neuronales:

1. Pre-chequeo tactico (ganar o bloquear en 1 jugada).
2. Generacion de candidatos `top-K` guiada por heuristica.
3. MCTS guiado por priors (PUCT-lite).
4. Rollouts cortos con politica epsilon-greedy heuristica.
5. Manejo de incertidumbre para variante `dark` usando memoria de colisiones.
6. Precomputacion de mapas y patrones en `begin_game` para acelerar evaluaciones.

## Flujo por turno (`play`)

1. Construye `board_mut` y lista de jugadas aparentes legales (`empty_cells`).
2. Actualiza memoria en `dark` (`_refresh_dark_knowledge` y ajuste de riesgo).
3. Busca:
   - jugada ganadora inmediata,
   - jugada de bloqueo inmediato del oponente.
4. Genera `root_candidates`:
   - score rapido local,
   - refinamiento con mejora de `shortest_path_distance` para el root.
5. Ejecuta MCTS hasta `soft_deadline`.
6. Retorna hijo raiz con mayor `visits`.
7. Si algo sale fuera de lo esperado, usa fallback heuristico seguro.

## Heuristicas

El score rapido combina:

- soporte local (vecinos propios),
- presion (vecinos del rival),
- potencial de puente local,
- sesgo central,
- progreso por eje de conexion segun color,
- bonus por cercania a foco tactico (`last_move` o ultimo propio),
- en `dark`: penalizacion por riesgo + pequeno bonus de exploracion local.

Adicionalmente se usa un mapa precomputado de patrones de puente (distancia 2 con dos conectores comunes) para valorar conexiones virtuales de forma mas precisa.

La evaluacion de estado (`_evaluate_board`) combina:

- diferencia de camino minimo (`shortest_path_distance`),
- conectividad local,
- densidad de puentes potenciales,
- fragmentacion por componentes conectados.

## MCTS

Cada nodo guarda:

- tablero,
- jugador al que le toca,
- `prior`,
- acumulado `value_sum`,
- `visits`,
- `children`,
- lista `unexpanded`.

Seleccion:

`score = exploit + explore`, donde:

- `exploit` usa `q` (invertido cuando decide el oponente),
- `explore = c_puct * prior * sqrt(N_parent)/(1 + N_child)`.

Expansion:

- toma una accion de `unexpanded`,
- crea tablero hijo,
- genera candidatos para el siguiente jugador con branching menor.

Simulacion:

- rollout corto (profundidad fija),
- politica epsilon-greedy con score heuristico.

Backpropagation:

- valor en perspectiva del jugador raiz (self),
- actualiza `visits` y `value_sum` en todo el camino.

## Variante Dark

Estado interno:

- `known_opp`: celdas ya confirmadas del rival,
- `collision_cells`: historial de colisiones,
- `risk_map`: calor por probabilidad de colision,
- `attempt_heat`: cuantas veces se insistio en zonas.

`on_move_result(move, success)`:

- `success=True`: reduce riesgo local en esa zona.
- `success=False`: marca colision y aumenta riesgo alrededor.

El score en `dark` castiga zonas riesgosas y evita repetir intentos en hotspots.

## Tiempo y seguridad

- Presupuesto dinamico por fase (`open/mid/late/end`) en vez de una fraccion fija.
- Cuando hay mas tiempo permitido por jugada, tambien sube branching (`top-K`) y profundidad efectiva de rollout.
- Todo el ciclo principal de MCTS corta por deadline.
- Si no hay salida valida de MCTS, se usa fallback legal.

## Precomputacion

En `begin_game` se construyen estructuras estaticas para no recalcularlas en cada nodo:

- `self._neighbors`: vecinos de cada celda.
- `self._center_score_map`: sesgo central por celda.
- `self._axis_score_map`: sesgo por eje de conexion para cada color.
- `self._bridge_patterns`: patrones de puente por celda.
- `self._cells`: lista plana de celdas para iteraciones rapidas.

## Parametros tunables

En `begin_game` y helpers:

- `self._c_puct`
- `self._rollout_depth`
- `self._rollout_epsilon`
- limites de candidatos (`_root_candidate_limit`, `_tree_candidate_limit`)
- pesos de riesgo (`_compute_risk_weight`)

Tambien se pueden ajustar por variables de entorno (sin tocar codigo):

- `COMPUS_C_PUCT`
- `COMPUS_ROLLOUT_DEPTH`
- `COMPUS_ROLLOUT_EPSILON`
- `COMPUS_RISK_LOW`
- `COMPUS_RISK_MID`
- `COMPUS_RISK_HIGH`
- `COMPUS_BUDGET_OPEN`
- `COMPUS_BUDGET_MID`
- `COMPUS_BUDGET_LATE`
- `COMPUS_BUDGET_END`
- `COMPUS_ROOT_LIMIT_SCALE`
- `COMPUS_TREE_LIMIT_SCALE`
- `COMPUS_ROLLOUT_DEPTH_BONUS`
- `COMPUS_FIXED_SEED` (opcional, para reproducibilidad)

## Tuning automatico con Docker

Script incluido:

- `estudiantes/compus/tune_docker.py`
- `estudiantes/compus/overnight_train.py` (bucle robusto con checkpoint/reanudacion)

Ejemplo rapido:

```bash
python estudiantes/compus/tune_docker.py --team compus --strategy "MiEstrategia_mi_equipo" --opponent "MCTS_Tier_3" --trials 8 --num-games 2 --variants classic,dark --both-colors
```

Aplicar automaticamente la mejor configuracion a los defaults de `strategy.py`:

```bash
python estudiantes/compus/tune_docker.py --team compus --strategy "MiEstrategia_mi_equipo" --opponent "MCTS_Tier_3" --trials 12 --num-games 2 --variants classic,dark --both-colors --apply-best
```

El script:

1. prueba distintas combinaciones de hiperparametros,
2. ejecuta `docker compose run experiment ...`,
3. mide winrate y penaliza forfeits,
4. guarda ranking y mejor configuracion en `estudiantes/compus/results/tuning_summary_*.json`.
5. con `--apply-best`, escribe esa mejor configuracion en `DEFAULT_TUNING` dentro de `strategy.py`.

## Entrenamiento nocturno robusto

Comando sugerido para dejar corriendo toda la noche:

```bash
python estudiantes/compus/overnight_train.py --team compus --strategy "MiEstrategia_mi_equipo" --run-name overnight_main --runner docker --opponents MCTS_Tier_3,MCTS_Tier_4 --variants classic,dark --both-colors --num-games 1 --move-timeout 8 --seed-batches 1 --hours 10 --max-evals 999999 --apply-best-every 5
```

Resume automaticamente si vuelves a correr el mismo `--run-name` (usa el checkpoint existente).

Archivos clave del run:

- `estudiantes/compus/results/<run-name>/checkpoint.json`
- `estudiantes/compus/results/<run-name>/history.jsonl`
- `estudiantes/compus/results/<run-name>/summary.json`
- `estudiantes/compus/results/<run-name>/best_params.json`
- `estudiantes/compus/results/<run-name>/best_params.env`
- `estudiantes/compus/results/<run-name>/errors.log` (si hubo fallos)

Opciones utiles:

- `--fresh`: iniciar run nuevo (archiva el anterior con timestamp).
- `--no-apply-best`: no tocar `strategy.py` durante el entrenamiento.
- `--no-apply-best-on-improve`: evita escritura en cada mejora.
- `--apply-best-every N`: aplica mejor config cada N evaluaciones.

## Consideraciones

- El algoritmo esta hecho para ser robusto en `classic` y `dark`.
- No depende de archivos externos ni librerias fuera de stdlib.
- Todo vive en `estudiantes/compus/strategy.py`, como pide la evaluacion.
