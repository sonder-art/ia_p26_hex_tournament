# Torneo de Hex — IA Primavera 2026 ITAM

Torneo de estrategias de Hex para el curso de Inteligencia Artificial.

Tu equipo implementa **una sola estrategia** que juega Hex en un tablero de **11x11** en dos variantes: **classic** y **dark** (fog of war). El framework descubre todas las estrategias, ejecuta un torneo en formato **liga** (round-robin) con aislamiento de procesos, y genera calificaciones automaticamente.

---

## Documentacion

| Documento | Descripcion |
|-----------|-------------|
| **[Guia para equipos](docs/team_guide.md)** | Paso a paso: setup, implementacion, pruebas, entrega |
| **[Reglas del torneo](docs/rules.md)** | Mecanica del juego, formato, restricciones, juego limpio |

---

## Setup rapido

```bash
# 1. Forkea el repo y clona tu fork
git clone https://github.com/MCCH-7945/ia_p26_hex_tournament.git
cd ia_p26_hex_tournament

# 2. Instala Docker (necesario para correr los tiers MCTS)
#    https://docs.docker.com/get-docker/

# 3. Instala dependencias locales
pip install -r requirements.txt

# 4. Corre un torneo rapido de prueba
python3 run_all.py
```

> **Nota**: Los tiers MCTS (MCTS_Tier_1 a MCTS_Tier_5) son binarios compilados que **solo corren dentro de Docker**. Contra Random puedes probar sin Docker; para todo lo demas, usa Docker.

---

## Que tienes que hacer

1. **Crea tu equipo**: `cp -r estudiantes/_template estudiantes/mi_equipo`
2. **Edita** `estudiantes/mi_equipo/strategy.py` — una clase que hereda de `Strategy`
3. **Estudia** `strategies/random_strat.py` — es la unica estrategia con codigo visible, usala como referencia
4. **Prueba** contra Random localmente, luego contra los tiers MCTS en Docker
5. **Documenta** tu estrategia en `estudiantes/mi_equipo/README.md`
6. **Entrega** via Pull Request

Tu estrategia debe funcionar para **ambas variantes** (classic y dark).

Para la guia detallada paso a paso, consulta **[docs/team_guide.md](docs/team_guide.md)**.

---

## Comandos principales

### Sin Docker (solo Random)

```bash
# Tu estrategia contra Random
python3 experiment.py --black "MiEstrategia_mi_equipo" --white "Random" --num-games 5 --verbose

# Torneo rapido
python3 run_all.py
```

### Con Docker (tiers MCTS)

```bash
# Tu estrategia contra un tier especifico
docker compose run experiment \
  python experiment.py --black "MiEstrategia_mi_equipo" --white "MCTS_Tier_3" \
  --num-games 5 --verbose

# Torneo completo: tu equipo vs todos los defaults
TEAM=mi_equipo docker compose up team-tournament

# Torneo oficial (ambas variantes, liga)
docker compose up tournament
```

### Opciones de `run_all.py`

```bash
python3 run_all.py                          # rapido (classic, 4 games/pair)
python3 run_all.py --official               # ambas variantes, liga, 4 games/pair
python3 run_all.py --team mi_equipo         # solo tu equipo vs defaults
python3 run_all.py --real                   # evaluacion (10 games/pair)
```

---

## Calificacion

### Formato del torneo

El torneo se juega en formato **liga round-robin** (todos contra todos) en **dos variantes** del juego: classic y dark.

#### Paso 1: Se juegan dos ligas

- **Liga classic**: todos contra todos en Hex clasico (informacion perfecta).
- **Liga dark**: todos contra todos en Dark Hex (fog of war).
- Cada par juega **4 partidas por variante** (2 como Negro, 2 como Blanco — balance de color perfecto).
- **Victoria = 1 punto, derrota = 0 puntos.** No hay empates en Hex.

#### Paso 2: Se suman los puntos de ambas ligas

Tus puntos de la liga classic + tus puntos de la liga dark = tus **puntos totales**.

#### Paso 3: Se determina que modelos venciste

Hay **6 modelos** de dificultad creciente en el torneo. Tu estrategia juega contra todos ellos (y contra las demas estrategias de estudiantes). Al final, se comparan los **puntos totales** de tu estrategia contra los de cada modelo:

- **Si tus puntos totales ≥ puntos totales del modelo → lo venciste.**
- **Si tus puntos totales < puntos totales del modelo → no lo venciste.**
- **Empate (mismos puntos) = lo venciste** (el empate favorece al estudiante).

#### Paso 4: Se calcula tu calificacion

Tu calificacion se basa en **cuantos modelos venciste**, no en cuales especificamente:

| Modelos vencidos | Calificacion | Ejemplo |
|------------------|-------------|---------|
| **0** | **0** | No venciste a ninguno |
| **1** | **5** | Venciste a 1 modelo (probablemente Random) |
| **2** | **6** | Venciste a 2 modelos |
| **3** | **7** | Venciste a 3 modelos |
| **4** | **8** | Venciste a 4 modelos |
| **5** | **9** | Venciste a 5 modelos |
| **6** | **10** | Venciste a todos los modelos |

**Formula**: `calificacion = 4 + cantidad_de_modelos_vencidos` (si venciste al menos 1; si no venciste a ninguno, es 0).

**Excepcion**: Los **top 3** estudiantes por puntos totales obtienen automaticamente **10 puntos** (requiere minimo 3 estudiantes).

### 6 modelos de referencia

| Modelo | Dificultad |
|--------|------------|
| **Random** | Trivial — juega al azar |
| **MCTS_Tier_1** | Facil |
| **MCTS_Tier_2** | Media |
| **MCTS_Tier_3** | Dificil |
| **MCTS_Tier_4** | Muy dificil |
| **MCTS_Tier_5** | Experto |

Los algoritmos son **opacos** — binarios compilados cuyo codigo no puedes ver. Solo puedes estudiar su comportamiento jugando contra ellos.

**Nota sobre el conteo**: Como la calificacion se basa en la **cantidad** de modelos vencidos (no en sus nombres), no importa si un modelo facil termina con mas puntos que uno dificil. Si MCTS_Tier_4 le gana a MCTS_Tier_5 en el torneo, ambos siguen contando como un modelo vencido cada uno.

### Ejemplo completo

Supongamos un torneo con 1 estudiante y los 6 modelos. Cada par juega 4 partidas en classic + 4 en dark. Standings finales:

```
  LIGA CLASSIC                    LIGA DARK
  Rank  Estrategia     Pts       Rank  Estrategia     Pts
  ──────────────────────────      ──────────────────────────
   1    MCTS_Tier_5     30        1    MCTS_Tier_5     28
   2    Tu_equipo       25        2    Tu_equipo       26
   3    MCTS_Tier_4     24        3    MCTS_Tier_4     25
   4    MCTS_Tier_3     20        4    MCTS_Tier_3     18
   5    MCTS_Tier_2     15        5    MCTS_Tier_2     14
   6    MCTS_Tier_1     10        6    MCTS_Tier_1      9
   7    Random           3        7    Random           2

  STANDINGS COMBINADOS
  Estrategia        Classic  Dark  Total
  ──────────────────────────────────────
  MCTS_Tier_5          30     28     58
  Tu_equipo            25     26     51   ← tus puntos totales
  MCTS_Tier_4          24     25     49
  MCTS_Tier_3          20     18     38
  MCTS_Tier_2          15     14     29
  MCTS_Tier_1          10      9     19
  Random                3      2      5
```

**Calculo**:
1. Tus puntos totales: **51** (25 classic + 26 dark)
2. Comparas contra cada modelo:
   - Random (5 pts total): 51 ≥ 5 → **vencido**
   - MCTS_Tier_1 (19 pts total): 51 ≥ 19 → **vencido**
   - MCTS_Tier_2 (29 pts total): 51 ≥ 29 → **vencido**
   - MCTS_Tier_3 (38 pts total): 51 ≥ 38 → **vencido**
   - MCTS_Tier_4 (49 pts total): 51 ≥ 49 → **vencido**
   - MCTS_Tier_5 (58 pts total): 51 < 58 → **no vencido**
3. Modelos vencidos: **5**
4. Calificacion: 4 + 5 = **9**

### Resumen rapido

| Lo que cuenta | Detalle |
|---------------|---------|
| **Que se juega** | Liga round-robin, classic + dark, 4 partidas/par/variante |
| **Que se mide** | Puntos totales (victorias classic + victorias dark) |
| **Que se compara** | Tus puntos totales vs los de cada modelo |
| **Condicion de victoria** | Tus pts ≥ pts del modelo (empate = vences) |
| **Calificacion** | 0 modelos = 0, luego 4 + N modelos vencidos (max 10) |
| **Excepcion** | Top 3 estudiantes = 10 automatico |

Para las reglas completas del juego, restricciones de recursos y criterios de descalificacion, consulta **[docs/rules.md](docs/rules.md)**.

---

## Entrega

```bash
git add estudiantes/mi_equipo/strategy.py estudiantes/mi_equipo/README.md
git commit -m "add strategy mi_equipo"
git push origin mi_equipo
```

Abre un **Pull Request** de tu branch hacia `main`.

**Tu PR debe contener:**
- `estudiantes/<tu_equipo>/strategy.py` — **obligatorio** (se evalua automaticamente)
- `estudiantes/<tu_equipo>/README.md` — **obligatorio** (documenta tu estrategia)
- Opcionalmente: notebooks, scripts, datos en tu directorio (no seran evaluados)

**Tu `README.md` debe explicar:**
- Que algoritmo(s) usaste (MCTS, minimax, heuristicas, etc.)
- Como maneja tu estrategia la variante dark (fog of war)
- Decisiones de diseno importantes y por que las tomaste
- Resultados de pruebas locales (contra que tiers lograste ganar)

**NO incluyas:**
- Cambios a archivos fuera de `estudiantes/<tu_equipo>/`
- Archivos grandes (`.pkl`, `.npy`, modelos)
- Resultados (`results/`)

---

## Estructura del repositorio

```
ia_p26_hex_tournament/
├── run_all.py              # Un comando para todo
├── strategy.py             # Clase base Strategy + GameConfig
├── hex_game.py             # Motor del juego (tablero, BFS, Dijkstra)
├── referee.py              # Referee con aislamiento de procesos
├── strategy_worker.py      # Subprocess que ejecuta cada estrategia
├── tournament.py           # Torneo liga con calificaciones
├── experiment.py           # Pruebas individuales (verbose)
├── strategies/             # Defaults compilados
│   ├── random_strat.py     #   Random (unico .py, LEELO como referencia)
│   └── mcts_tier_*_strat.*.so  #   MCTS_Tier_1..5 (binarios, solo Docker)
├── estudiantes/            # <-- AQUI VA TU ESTRATEGIA
│   ├── _template/          #     Template para copiar
│   └── <tu-equipo>/
│       ├── strategy.py     #     Tu estrategia (UNICO archivo evaluado)
│       └── README.md       #     Tu documentacion (obligatorio)
├── docs/
│   ├── team_guide.md       #     Guia paso a paso para equipos
│   └── rules.md            #     Reglas del torneo
├── results/                # Salidas del torneo
│   ├── runs/<timestamp>/   #     Resultados completos por ejecucion
│   ├── latest -> runs/...  #     Symlink al mas reciente
│   └── history.jsonl       #     Historial de ejecuciones
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

---

## Outputs del torneo

| Archivo | Descripcion |
|---------|-------------|
| `results/runs/<timestamp>/config.json` | Parametros del torneo |
| `results/runs/<timestamp>/games.jsonl` | Una linea JSON por partida (con move log completo) |
| `results/runs/<timestamp>/classic_league.json` | Standings liga classic |
| `results/runs/<timestamp>/dark_league.json` | Standings liga dark |
| `results/runs/<timestamp>/combined_standings.json` | Standings combinados |
| `results/runs/<timestamp>/grades.json` | Calificaciones (JSON) |
| `results/runs/<timestamp>/grades.csv` | Calificaciones (CSV para spreadsheet) |
| `results/runs/<timestamp>/summary.txt` | Resumen legible |
| `results/latest` | Symlink al mas reciente |
| `results/history.jsonl` | Historial de ejecuciones |
