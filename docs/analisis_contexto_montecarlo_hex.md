# Análisis de contexto: módulo de Monte Carlo Search para Hex

## Alcance y limitación de acceso

Se intentó acceder directamente a las páginas y notebooks listados en `sonder.art`, pero desde este entorno los requests HTTP/HTTPS a ese dominio responden `403 Forbidden` (bloqueo de red/proxy). Por lo tanto, este análisis se construye con:

1. Los títulos/estructura de las URLs proporcionadas.
2. El contexto técnico del repositorio actual (motor de Hex, API de estrategias y torneo).
3. Conocimiento estándar de MCTS/UCT aplicado a Hex.

---

## Lectura estructural del temario (inferida por la secuencia de páginas)

La progresión de contenidos sugiere este hilo pedagógico:

1. **Más allá de Minimax**
   - Motivación: minimax puro no escala en Hex 11x11.
   - Cambio de paradigma: búsqueda estadística por muestreo (rollouts).

2. **Hex**
   - Reglas, topología hexagonal, propiedad sin empates.
   - Objetivo de conexión (P1 vertical, P2 horizontal).

3. **Hex y rollouts (notebook)**
   - Simulaciones aleatorias desde estados intermedios para estimar valor de jugadas.
   - Primer baseline tipo Monte Carlo flat (sin árbol profundo sofisticado).

4. **MCTS**
   - Bucle de 4 pasos: *selection → expansion → simulation → backpropagation*.
   - Árbol por estados/movimientos con actualización incremental.

5. **MCTS paso a paso (notebook)**
   - Instrumentación del árbol: visitas, victorias, frontera de expansión.
   - Validación de consistencia de estadísticas por nodo.

6. **UCT**
   - Fórmula UCB1 aplicada a selección de nodos.
   - Balance exploración/explotación mediante constante `c`.

7. **MCTS en acción**
   - Evaluación práctica en partidas reales.
   - Sensibilidad a presupuesto de tiempo y número de simulaciones.

8. **UCT y experimentos (notebook)**
   - Barrido de hiperparámetros (`c`, rollout policy, presupuesto).
   - Curvas de desempeño (winrate / estabilidad).

9. **Más allá**
   - Extensiones: RAVE, progressive bias, transposition tables, rollout guiado.

10. **Torneo (notebook de aplicación)**
    - Integración final: estrategia bajo restricciones reales de tiempo y entorno competitivo.

---

## Cómo este contexto encaja con ESTE repositorio

### 1) Motor de juego y reglas
El repositorio ya implementa correctamente las bases de Hex para este flujo:
- Vecindad hexagonal de 6 direcciones.
- Ganador por conectividad BFS.
- Variante `classic` y variante `dark` con colisiones y visibilidad parcial.

Esto habilita directamente MCTS y también variantes para información imperfecta. 

### 2) API de estrategia compatible con MCTS
La interfaz `Strategy` y `GameConfig` permite:
- Precálculo en `begin_game`.
- Decisión con límite duro de tiempo en `play`.
- Retroalimentación de colisión en `on_move_result` (clave para dark).

Es una API adecuada para implementar:
- MCTS clásico en `classic`.
- Determinización / belief updates en `dark`.

### 3) Entorno de evaluación tipo torneo
El pipeline del repo replica exactamente lo que se esperaría del notebook de torneo:
- Competencia contra niveles baseline (`Random`, `MCTS_Tier_1..5`).
- Medición por victorias agregadas y tier máximo vencido.
- Restricciones operativas (tiempo, memoria, CPU) coherentes con investigación aplicada.

---

## Implicaciones técnicas para una estrategia competitiva

### A. En `classic`
Recomendación de línea base robusta:
- **Selection**: UCT con `Q/N + c*sqrt(log(Np)/N)`.
- **Expansion**: 1 hijo por iteración (o política de expansión parcial).
- **Simulation**: rollout semi-guiado (no completamente random).
- **Backprop**: recompensa binaria desde perspectiva del jugador raíz.

Mejoras de alto impacto:
- Priorización de celdas cercanas a caminos mínimos (`shortest_path_distance`).
- Reutilización de árbol entre turnos (*tree reuse*).
- Control estricto del presupuesto temporal (corte con margen de seguridad).

### B. En `dark`
Por información imperfecta, MCTS directo sobre estado observado es miope.
Conviene usar:
- **Determinización ligera**: muestrear tableros plausibles de piedras ocultas.
- **Ensemble de búsquedas cortas** sobre múltiples determinizaciones.
- **Penalización de zonas de alta probabilidad de colisión**.

La señal `on_move_result(success=False)` debe incorporarse para actualizar creencias locales.

### C. Rollout policy práctica
Para Hex, rollout totalmente aleatorio suele ser ruidoso. Mejor:
- Sesgo hacia celdas que reduzcan distancia de conexión propia.
- Bloqueo oportunista cuando oponente tenga amenaza de conexión corta.
- Prioridad a vecindad de último movimiento exitoso (si aplica).

---

## Riesgos frecuentes (y cómo mitigarlos)

1. **Agotar tiempo por jugada**
   - Mitigar con bucle temporal usando `time.monotonic()` y margen (`~0.85–0.9` del límite).

2. **Rollouts demasiado lentos**
   - Estructuras ligeras (listas de vacías, actualizaciones in-place con undo simple).

3. **Sobreexploración por mala `c` en UCT**
   - Ajustar por experimentos: barrido pequeño y validación cruzada contra tiers.

4. **Baja robustez en dark**
   - Introducir modelo de incertidumbre mínimo + memoria de colisiones.

---

## Roadmap sugerido para el proyecto

1. Implementar MCTS-UCT funcional y estable para `classic`.
2. Medir contra `Random` y `MCTS_Tier_1/2` con presupuesto fijo.
3. Optimizar rollout policy (heurística de caminos).
4. Añadir reuso de árbol entre turnos.
5. Extender a `dark` con determinización simple + colisiones.
6. Correr mini-torneo repetido y consolidar hiperparámetros.

---

## Resumen ejecutivo

Las páginas compartidas parecen estructurar un camino completo desde fundamentos (rollouts) hasta integración competitiva (torneo) para Hex con MCTS/UCT. El repositorio actual está bien alineado con ese enfoque: ya ofrece motor, API y marco de evaluación apropiados. La estrategia ganadora en este contexto probablemente combine **UCT eficiente + rollouts guiados + disciplina de tiempo + adaptación para dark con incertidumbre explícita**.
