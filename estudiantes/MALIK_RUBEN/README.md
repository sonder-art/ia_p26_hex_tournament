# FogBridge_MALIK_RUBEN

## Resumen

La estrategia implementada en `strategy.py` se llama **`FogBridge_MALIK_RUBEN`** y fue diseñada para jugar en las dos variantes del torneo:

- **classic**: información completa
- **dark**: fog of war, donde solo vemos nuestras piedras y las del oponente descubiertas por colisión

La idea central es combinar una evaluación heurística de conectividad con una búsqueda táctica corta. En `classic` la estrategia hace una revisión de victorias inmediatas, bloqueos inmediatos y después una búsqueda tipo minimax muy poco profunda sobre un conjunto pequeño de jugadas candidatas. En `dark` reutiliza la misma intuición posicional, pero además estima el riesgo de colisión de cada celda aparentemente vacía.

## Algoritmo

La estrategia no usa aprendizaje automático ni RL. Todo está implementado con **stdlib + utilidades del framework**.

### 1. Evaluación principal del tablero

Cada tablero se evalúa con una función heurística que combina:

- **Distancia Dijkstra propia vs rival** usando `shortest_path_distance`
- **Span** sobre el eje objetivo
  - Negro: cuánto se extienden nuestras piedras de arriba hacia abajo
  - Blanco: cuánto se extienden de izquierda a derecha
- **Puentes y soporte a dos saltos**
- **Masa central**
- **Presencia en bordes objetivo**

La señal más importante es la diferencia entre la distancia mínima de conexión propia y la del oponente. Sobre eso se suman rasgos estructurales para favorecer cadenas conectadas, puentes y presencia útil en el centro.

### 2. Modo classic

En `classic` la política sigue este orden:

1. Buscar **victorias inmediatas**
2. Si no existen, buscar **bloqueos inmediatos** de una victoria rival
3. Construir una lista pequeña de candidatas con una evaluación estática
4. Para cada candidata, simular nuestra jugada y luego considerar las mejores respuestas rivales
5. Elegir la jugada con mejor valor combinado

No es un minimax profundo, porque el límite de 15 segundos por jugada obliga a ser conservadores. La búsqueda es intencionalmente corta pero estable.

### 3. Modo dark

En `dark` la estrategia no sabe dónde jugó el oponente salvo cuando ocurre una colisión y esa piedra se revela. Por eso se modelan tres cosas:

- **Valor posicional visible**: igual que en classic, pero sobre el tablero parcial
- **Bono de información**: celdas con vecindad útil y valor de corredor reciben preferencia
- **Riesgo de colisión**: se estima a partir de cuántos turnos del oponente pudieron ya haber ocurrido y cuántas piedras rivales visibles conocemos

La función de riesgo penaliza:

- zonas compatibles con el corredor natural del oponente
- celdas cercanas a piedras rivales visibles
- celdas aisladas respecto a nuestras propias cadenas

Además, se guarda memoria de colisiones anteriores mediante `on_move_result(move, success)`, para no insistir en celdas que ya sabemos ocupadas.

## Cómo maneja dark mode

La estrategia usa una aproximación de **creencia implícita** en vez de una determinización completa o ISMCTS:

- No genera tableros ocultos completos
- No intenta reconstruir exactamente la posición rival
- Mantiene un estimado simple del número de piedras ocultas rivales
- Usa ese estimado para penalizar celdas con alta probabilidad de choque

Esta decisión se tomó porque:

- es mucho más barata computacionalmente
- cabe cómodamente dentro del límite de 15 segundos
- evita una implementación grande y frágil en un solo archivo

## Decisiones de diseño

### Por qué no usar MCTS puro

MCTS sería una opción natural para Hex, pero aquí había varios costos:

- el archivo evaluado es uno solo
- hay que resolver `classic` y `dark`
- en `dark`, una simulación ingenua sin modelo de información imperfecta puede engañarse mucho
- bajo el tiempo disponible era más confiable una heurística bien afinada con búsqueda corta

### Por qué usar `shortest_path_distance`

El framework ya provee una señal muy fuerte para Hex: la distancia de conexión por Dijkstra. Esa señal:

- refleja bien el progreso real en el tablero
- sirve tanto para atacar como para defender
- funciona razonablemente bien incluso sobre la vista parcial en `dark`

### Por qué usar candidatas en lugar de todas las jugadas

En un tablero 11x11 hay hasta 121 jugadas posibles. Explorar todo con profundidad adicional en cada turno es innecesario y caro. Por eso primero se rankean movimientos y solo se estudia un subconjunto prometedor.

## Alternativas consideradas

Se consideraron estas opciones:

- **MCTS clásico**
  - Ventaja: muy natural para Hex
  - Desventaja: más costoso y menos claro para `dark`
- **Minimax más profundo**
  - Ventaja: más táctico
  - Desventaja: demasiado caro para 11x11 con 15 s y sin poda sofisticada
- **Determinización completa para dark**
  - Ventaja: modela mejor la incertidumbre
  - Desventaja: complejidad alta para una entrega de un solo archivo

La versión entregada prioriza robustez, claridad y costo controlado por movimiento.

## Resultados de pruebas locales

Pruebas ejecutadas en este workspace:

- `classic` vs `Random`, 4 partidas: **4 victorias, 0 derrotas**
- `dark` vs `Random`, 4 partidas: **4 victorias, 0 derrotas**

Todavía no quedaron corridas documentadas contra los tiers MCTS dentro de Docker en esta entrega. Los comandos sugeridos para seguir validando son:

```bash
python3 experiment.py --black "FogBridge_MALIK_RUBEN" --white "Random" --num-games 5 --team MALIK_RUBEN
python3 experiment.py --black "FogBridge_MALIK_RUBEN" --white "Random" --variant dark --num-games 5 --team MALIK_RUBEN
TEAM=MALIK_RUBEN docker compose up team-tournament
```

## Restricciones respetadas

- un solo archivo evaluado: `estudiantes/MALIK_RUBEN/strategy.py`
- solo `stdlib` y utilidades del torneo
- compatible con `classic` y `dark`
- usa `on_move_result` para manejar colisiones
- nombre único de estrategia: `FogBridge_MALIK_RUBEN`
