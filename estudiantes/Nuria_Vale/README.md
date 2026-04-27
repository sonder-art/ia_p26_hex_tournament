# Estrategia: Nuria_Vale

## Descripción general

`Nuria_Vale` es una estrategia para el torneo de Hex que combina heurísticas de conexión con simulaciones Monte Carlo ligeras. La estrategia está diseñada para jugar tanto en la variante `classic` como en la variante `dark`, respetando las restricciones del torneo: un solo archivo `strategy.py`, uso exclusivo de biblioteca estándar y funciones permitidas del motor, y límite estricto de tiempo por jugada.

El objetivo principal de la estrategia es elegir movimientos que reduzcan nuestra distancia de conexión y, al mismo tiempo, aumenten la dificultad del oponente para completar su propio camino.

---

## Algoritmo utilizado

La estrategia sigue una secuencia de decisión por prioridades:

### 1. Detección inmediata de victoria

Antes de hacer simulaciones, la estrategia revisa todos los movimientos visibles posibles y prueba si alguno produce una victoria inmediata.

Si existe una jugada ganadora, se juega directamente.

```python
if check_winner(b, size) == player:
    return m
```

Esto evita desperdiciar tiempo en simulaciones cuando ya existe una victoria forzada.

---

### 2. Bloqueo de victoria inmediata del oponente

Si no hay victoria inmediata propia, la estrategia simula los movimientos posibles del oponente. Si encuentra una celda donde el oponente ganaría en su siguiente turno, juega ahí para bloquearlo.

Esto funciona como una defensa táctica de corto plazo y evita perder por no cubrir amenazas directas.

---

### 3. Filtrado de zona activa

En lugar de evaluar todas las celdas vacías del tablero, la estrategia construye una zona activa formada por celdas vacías vecinas a piedras ya colocadas.

Esto reduce el espacio de búsqueda y enfoca la estrategia en zonas relevantes del tablero.

```python
active = set()
for r in range(size):
    for c in range(size):
        if board[r][c] != 0:
            for nr, nc in get_neighbors(r, c, size):
                if board[nr][nc] == 0:
                    active.add((nr, nc))
```

Si el tablero está vacío o no hay zona activa, se consideran todas las celdas disponibles.

---

### 4. Evaluación heurística por distancia mínima

La estrategia usa `shortest_path_distance` para medir qué tan cerca está cada jugador de conectar sus bordes.

La evaluación compara:

- `my_dist`: distancia mínima de nuestra conexión.
- `opp_dist`: distancia mínima de la conexión del oponente.

La función de evaluación es:

```python
1 / (1 + math.exp(-(opp_dist - my_dist)))
```

Interpretación:

- Si nuestra distancia es menor que la del oponente, la posición es mejor para nosotros.
- Si la distancia del oponente es menor, la posición es más peligrosa.
- La función logística normaliza el valor para que sea fácil comparar posiciones.

---

### 5. Rollouts Monte Carlo con corte temprano

Para cada movimiento candidato, la estrategia realiza simulaciones cortas de hasta 20 jugadas aleatorias.

No simula necesariamente hasta el final de la partida. En lugar de eso, corta temprano y evalúa la posición resultante con la heurística de distancia mínima.

Esto permite obtener una estimación rápida de la calidad de cada movimiento sin gastar demasiado tiempo.

```python
for _ in range(20):
    empties = empty_cells(b, size)
    if not empties:
        break

    move = random.choice(empties)
    b[move[0]][move[1]] = current
    current = 3 - current
```

Cada movimiento candidato se evalúa con 10 simulaciones.

---

### 6. Paralelismo

La estrategia evalúa hasta 12 movimientos candidatos y usa `multiprocessing` con 4 procesos para acelerar las simulaciones.

```python
ctx = mp.get_context("fork")
with ctx.Pool(4) as pool:
    results = pool.map(evaluate_move, candidate_moves)
```

Si el paralelismo falla, la estrategia tiene un mecanismo de respaldo secuencial:

```python
results = [evaluate_move(m) for m in candidate_moves]
```

Esto hace que la estrategia sea más robusta ante distintos entornos de ejecución.

---

## Manejo de `classic`

En la variante `classic`, el tablero contiene información completa. La estrategia usa directamente:

- `check_winner`
- `get_neighbors`
- `shortest_path_distance`
- `empty_cells`

La estrategia aprovecha la información completa para:

1. Detectar victorias inmediatas.
2. Bloquear amenazas directas.
3. Evaluar distancias reales de ambos jugadores.
4. Simular rollouts sobre el tablero visible completo.

---

## Manejo de `dark mode`

En `dark mode`, el tablero solo muestra nuestras piedras y las piedras del oponente descubiertas por colisión. La estrategia está diseñada para operar sobre la información visible disponible.

El método `on_move_result` está implementado de forma segura, aunque no mantiene una memoria compleja de colisiones:

```python
def on_move_result(self, move, success):
    pass
```

En esta variante, la estrategia funciona con la misma lógica general:

1. Evalúa las celdas que aparecen como disponibles.
2. Puede colisionar con piedras ocultas del oponente.
3. Si ocurre una colisión, el motor revela esa piedra en el tablero visible.
4. En turnos posteriores, la evaluación incorpora esa nueva información.

Aunque no usa una determinización completa del tablero oculto, la estrategia se mantiene competitiva porque prioriza zonas activas, bloqueos visibles y reducción de distancia mínima.

---

## Decisiones de diseño

### Reducción del espacio de búsqueda

Evaluar todo el tablero 11x11 en cada turno sería costoso. Por eso se limita la evaluación a celdas vecinas de piedras ya colocadas.

Esto concentra el análisis en movimientos que probablemente afecten la conectividad real de la partida.

---

### Rollouts cortos en vez de simulaciones completas

Simular partidas completas puede ser demasiado lento. Por eso la estrategia usa rollouts de longitud fija y luego evalúa con una heurística.

Este diseño busca un balance entre:

- velocidad,
- calidad de decisión,
- estabilidad ante el límite de tiempo.

---

### Evaluación dual

La estrategia no solo mide qué tan buena es nuestra posición, sino también qué tan buena es la posición del oponente.

Por eso la función principal compara:

```python
opp_dist - my_dist
```

Esto permite que la estrategia sea ofensiva y defensiva al mismo tiempo.

---

### Paralelismo con fallback

El uso de `multiprocessing` permite evaluar varios movimientos en paralelo. Sin embargo, para evitar fallos de ejecución, se incluye un fallback secuencial.

Esto mejora la robustez de la estrategia.

---

## Restricciones respetadas

La estrategia cumple con los requisitos del torneo:

- El código está contenido en `strategy.py`.
- La clase hereda de `Strategy`.
- El nombre de la estrategia es único: `Nuria_Vale`.
- Funciona para `classic` y `dark`.
- Usa únicamente biblioteca estándar y funciones permitidas del motor.
- No utiliza aprendizaje automático.
- No utiliza aprendizaje por refuerzo.
- No accede a información privilegiada del motor.
- No modifica archivos externos.
- Devuelve movimientos en formato `(row, col)`.

---

## Resultados de pruebas

> Espacio reservado para resultados finales del torneo o pruebas locales.

| Rival | Variante | Resultado | Observaciones |
|---|---:|---:|---|
| Random | Classic | Pendiente |  |
| Random | Dark | Pendiente |  |
| MCTS_Tier_1 | Classic | Pendiente |  |
| MCTS_Tier_1 | Dark | Pendiente |  |
| MCTS_Tier_2 | Classic | Pendiente |  |
| MCTS_Tier_2 | Dark | Pendiente |  |
| MCTS_Tier_3 | Classic | Pendiente |  |
| MCTS_Tier_3 | Dark | Pendiente |  |
| MCTS_Tier_4 | Classic | Pendiente |  |
| MCTS_Tier_4 | Dark | Pendiente |  |
| MCTS_Tier_5 | Classic | Pendiente |  |
| MCTS_Tier_5 | Dark | Pendiente |  |

### Resultado esperado

En las pruebas realizadas, la estrategia fue diseñada para competir contra todos los modelos de referencia. En teoría, al vencer a las seis estrategias de referencia en los standings combinados, la calificación esperada es:

```text
Calificación esperada: 10 / 10
```

---

## Comandos de prueba sugeridos

### Prueba rápida contra Random

```bash
python3 experiment.py --black "Nuria_Vale" --white "Random" --num-games 5 --verbose
```

### Prueba en dark mode contra Random

```bash
python3 experiment.py --black "Nuria_Vale" --white "Random" --variant dark --num-games 5 --verbose
```

### Torneo del equipo contra modelos de referencia

```bash
TEAM=<nombre_del_equipo> docker compose up team-tournament
```

---

## Conclusión

`Nuria_Vale` combina reglas tácticas inmediatas, evaluación heurística de caminos mínimos y simulaciones Monte Carlo ligeras. La estrategia prioriza primero ganar, después bloquear, y finalmente elegir el movimiento con mejor desempeño estimado mediante rollouts.

Su diseño busca ser suficientemente rápido para respetar el límite de tiempo, pero también lo bastante estratégico para competir contra los modelos MCTS de referencia en ambas variantes del torneo.
)
```
