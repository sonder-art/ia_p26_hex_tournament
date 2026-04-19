# ElieStrategy_eliefaya

## Descripción

Esta estrategia para el juego de Hex está basada en una heurística determinista que combina decisiones ofensivas y defensivas para maximizar la probabilidad de victoria en ambas variantes: classic y dark.

---

## Algoritmo

La estrategia sigue un enfoque jerárquico:

1. **Victoria inmediata**
   - Evalúa todas las jugadas posibles.
   - Si alguna produce una victoria directa (`check_winner`), se selecciona inmediatamente.

2. **Bloqueo del oponente**
   - Simula las jugadas del oponente.
   - Si el oponente puede ganar en el siguiente turno, se bloquea esa posición.

3. **Evaluación heurística**
   - Para cada movimiento posible:
     - Se calcula la distancia más corta a la victoria (`shortest_path_distance`).
     - Se calcula la distancia del oponente.
   - Se define un score:

     score = my_dist - opp_dist

4. **Control del centro**
   - Se penaliza la distancia al centro del tablero:
     
     score += distancia_al_centro

   - Esto favorece posiciones con mayor conectividad.

---

## Dark Mode (Fog of War)

La estrategia maneja información incompleta mediante:

- Registro de movimientos fallidos en `on_move_result`
- Uso de un conjunto `_failed_moves` para evitar repetir colisiones

Esto permite mejorar la eficiencia en presencia de información oculta.

---

## Decisiones de diseño

- Se eligió una heurística determinista en lugar de MCTS por simplicidad y control del tiempo.
- Se prioriza velocidad de cálculo para cumplir con el límite de 15 segundos por jugada.
- Se incorporó lógica defensiva explícita (bloqueo) para mejorar robustez.
- Se añadió control espacial (centro) para mejorar calidad de posiciones.

---

## Resultados

Pruebas locales:

- vs Random (classic): 100% victorias
- vs Random (dark): 100% victorias
- Comportamiento consistente en ambas variantes
- Mejora en eficiencia de juego respecto a versiones anteriores

---

## Limitaciones

- Estrategia greedy (no planifica múltiples turnos)
- No utiliza simulaciones tipo Monte Carlo (MCTS)
- No modela incertidumbre avanzada en dark mode

---

## Conclusión

La estrategia logra un equilibrio entre simplicidad, eficiencia y rendimiento, siendo capaz de vencer consistentemente al baseline y adaptarse correctamente a condiciones de información incompleta.