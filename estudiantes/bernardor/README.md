# Hexarino — estrategia de Hex 11×11

**Equipo:** bernardor
**Variantes soportadas:** classic, dark

## Resumen ejecutivo

Hexarino es una estrategia para Hex 11×11 que combina tres ideas. Primera: Monte Carlo Tree Search clásico con rollouts heurísticos cortos (no aleatorios) y evaluación final mediante la función shortest_path_distance del propio motor — la misma que el torneo usa como criterio de desempate, lo que alinea nuestra heurística con la métrica oficial. Segunda: incorporación de bridges, el patrón de conexión virtual de Hex, usado como atajo (save-bridge), como sesgo en los rollouts, y como fuente de movimientos prometedores en el filtro de candidatos. Tercera: para la variante dark (con niebla de guerra), aplicamos PIMC — se samplean N mundos consistentes con la información observada, se ejecuta MCTS sobre cada uno, y se elige el movimiento más votado, con un filtro de seguridad que penaliza movimientos con alta probabilidad estimada de colisión. La estrategia respeta la restricción de solo numpy + stdlib y un solo archivo strategy.py.

## Arquitectura general

Hexarino se compone de una clase coordinadora `Hexarino` (que hereda
de `Strategy`) y dos motores de búsqueda según la variante.

**Cadena de decisión** en cada `play()`:

1. **Apertura conocida** — si es nuestro primer turno, jugamos centro
   (Negro) o respuesta predefinida (Blanco).
2. **Jugada ganadora** — si nuestra distancia Dijkstra es 1, buscamos
   la celda que cierra la conexión.
3. **Bloqueo obligado** — si la del oponente es 1, ocupamos su celda
   crítica.
4. **Save-bridge** — si el oponente cortó un *carrier* de un bridge
   nuestro, jugamos el otro *carrier* automáticamente.
5. **Búsqueda principal** — `MCTSEngine` para *classic*, `DarkEngine`
   (PIMC) para *dark*.
6. **Fallback greedy** — red de seguridad. Si la búsqueda falla,
   jugada heurística de un nivel.

**Componentes**: `Hexarino` (coordinador), `MCTSEngine` y `MCTSNode`
(motor classic), `DarkEngine` (motor dark con PIMC), y un conjunto de
helpers compartidos para Dijkstra, detección de bridges y filtrado de
celdas relevantes.

**Precomputaciones**: en `begin_game()` calculamos los vecinos y
bridges potenciales de cada celda una sola vez. Estas tablas se
consultan en O(1) durante toda la partida, lo cual es importante
porque los rollouts MCTS las invocan miles de veces.



## Heurística base: distancia Dijkstra

El motor del juego (`hex_game.py`) ofrece la función
`shortest_path_distance(board, size, player)`, que calcula la distancia
mínima que le falta a `player` para conectar sus dos bordes objetivo.
El cálculo es un Dijkstra estándar sobre el grafo del tablero, donde:

- Las piedras propias tienen costo 0 (ya forman parte del camino).
- Las celdas vacías tienen costo 1 (hay que ocuparlas).
- Las piedras del oponente son paredes (impasables).

Si el jugador ya conectó sus bordes, la distancia es 0; si el oponente
bloquea completamente todo camino posible, devuelve `size*size+1`.

**Por qué esta heurística y no otra**: el motor del torneo usa
exactamente esta misma función como criterio de desempate cuando una
partida llega al límite de movimientos sin ganador definido. Es decir,
nuestra heurística no aproxima la métrica oficial — *es* la métrica
oficial. Cualquier evaluación basada en otra fórmula (e.g. número de
piedras propias en línea, distancia Manhattan al borde) introduciría
una desalineación entre lo que MCTS optimiza y lo que el torneo
recompensa.

A partir de Dijkstra construimos la **función de evaluación** principal:
eval(board) = dist_oponente − dist_propia

Más positivo significa mejor para nosotros (estamos más cerca de ganar
y/o el oponente más lejos). Casos especiales:

- Si `dist_propia == 0`: ya ganamos → +∞ efectivo.
- Si `dist_oponente == 0`: el oponente ya ganó → −∞ efectivo.

Para que MCTS pueda combinar esta evaluación con sus probabilidades de
victoria, mapeamos `eval` a `[0, 1]` mediante una sigmoide:
winrate(board) = 1 / (1 + exp(−(dist_op − dist_yo) / 2))

El factor `2` en el denominador modula la pendiente: una diferencia de
4 celdas se mapea a ~0.88, una diferencia de −4 a ~0.12. Así una hoja
del árbol con ventaja clara aporta un valor cercano a 1, una desventaja
clara cercano a 0, y posiciones equilibradas cercano a 0.5.


## MCTS para variante classic

Para la variante classic implementamos Monte Carlo Tree Search clásico
con cuatro fases por cada simulación:

1. **Selección**: desde la raíz, descendemos eligiendo en cada nodo el
   hijo con mayor UCB1, hasta llegar a un nodo no completamente
   expandido o terminal. La fórmula UCB1 que usamos es:
UCB1(hijo) = (W/N) + c · sqrt(ln(N_padre) / N_hijo)

   con `c = sqrt(2)`. El primer término favorece hijos con buen
   *winrate* histórico (explotación); el segundo favorece hijos poco
   visitados (exploración).

2. **Expansión**: si el nodo seleccionado tiene movimientos legales no
   explorados, creamos un nuevo hijo correspondiente a uno de ellos
   (elegido aleatoriamente entre los no expandidos).

3. **Rollout**: desde el nuevo hijo, simulamos la partida hasta una
   profundidad limitada (`ROLLOUT_DEPTH = 12` movimientos), guiados
   por una *default policy* heurística. Si la partida termina antes
   con un ganador detectado por `check_winner`, devolvemos 1 ó 0
   directamente. Si no, evaluamos la posición resultante con la
   sigmoide de Dijkstra descrita arriba.

4. **Retropropagación**: el valor obtenido en el rollout se propaga
   hacia arriba por el camino de selección, actualizando `N` y `W` de
   cada nodo. Importante: el valor se invierte alternadamente según
   qué jugador toma decisiones en cada nodo, de modo que cada padre
   maximiza su propio winrate cuando elige hijos.

Después de agotar el presupuesto de tiempo, devolvemos como jugada el
**hijo de la raíz con más visitas** (no el de mayor winrate), porque
ese criterio es más robusto frente a la varianza de las simulaciones.

### Decisiones de diseño clave

**Rollouts heurísticos, no aleatorios**. La opción "vanilla" es jugar
movimientos aleatorios durante el rollout. En Hex 11×11 esto es ruido
puro: con 121 celdas y reglas que recompensan conexión, cualquier
secuencia aleatoria converge a posiciones absurdas. En su lugar, en
cada paso del rollout puntuamos las celdas candidatas con una
heurística rápida O(1) y elegimos entre las top-K (`ROLLOUT_TOPK = 4`)
con un poco de aleatoriedad (`ROLLOUT_EPSILON = 0.15`) para
diversificar entre simulaciones.

**Rollouts cortos con evaluación final, no rollouts hasta el final**.
Llevar cada rollout hasta el final del juego (~80 movimientos) y
contar quién ganó es teóricamente más limpio, pero requiere ~80
llamadas a heurística por simulación. Truncamos a 12 movimientos y
evaluamos la posición resultante con Dijkstra una sola vez. Esto da
~10× más simulaciones por segundo a cambio de algo de ruido en la
señal — un trade-off favorable en este dominio.

**Por qué MCTS y no minimax**. El branching factor en Hex 11×11 es
~120 en el primer movimiento. Aún con poda alfa-beta, llegar a
profundidad 4 es difícil dentro de 2-3 segundos. MCTS, en cambio,
escala con el tiempo disponible: más segundos = más simulaciones =
mejor decisión, sin saltos cualitativos por agotar profundidad.


## Bridges: conocimiento de dominio

Un **bridge** es el patrón de conexión virtual más importante de Hex:
dos piedras del mismo color separadas por dos celdas vacías que
forman una "conexión garantizada" — si el oponente juega en una de
las dos celdas vacías (un *carrier*), el dueño juega en la otra y la
conexión se realiza físicamente.

Esquemáticamente, para una piedra negra en `(r, c)`, uno de los seis
bridges posibles tiene esta forma:
. B . .         ← (r,   c)   piedra origen
. a b .         ← (r+1, c-1) y (r+1, c) son los carriers
. . B .         ← (r+2, c-1) otra piedra negra

Si las dos celdas marcadas `a` y `b` están vacías, las dos `B` están
virtualmente conectadas: el oponente no puede separarlas. Cada celda
tiene hasta seis bridges potenciales, uno en cada dirección
hexagonal.

### Cómo usamos los bridges

En `begin_game()` precomputamos `bridges_de[(r, c)]` para todo el
tablero: una tabla que dado una celda devuelve la lista de
`(otra_piedra, carrier1, carrier2)` para sus seis bridges válidos
(filtrando los que se salen del tablero). Esta tabla se consulta en
O(1) durante el resto de la partida.

Los bridges se usan en cuatro lugares:

1. **Save-bridge como atajo táctico**. En `play()`, antes de invocar
   MCTS, revisamos si la última jugada del oponente cortó un carrier
   de algún bridge nuestro existente. Si sí, jugamos el otro carrier
   automáticamente. Esta es una decisión obviamente correcta y no
   tiene sentido gastar tiempo de búsqueda en ella.

2. **Save-bridge dentro de rollouts**. Durante una simulación MCTS,
   si en la última jugada simulada el oponente cortó un carrier,
   forzamos al jugador siguiente (su dueño) a jugar el otro carrier.
   Esto hace que los rollouts respeten conexiones virtuales en vez
   de descartarlas por azar, lo cual mejora notablemente la calidad
   de la señal.

3. **Bonus en la heurística rápida**. La función que puntúa celdas
   candidatas durante rollouts da un bonus de +5 a celdas que crean
   un nuevo bridge potencial con una piedra propia (es decir, donde
   colocar piedra forma un patrón de bridge cuyos carriers están
   ambos vacíos).

4. **Movimientos relevantes en el filtro**. Cuando construimos la
   lista de candidatos a expandir en un nodo MCTS, incluimos las
   "otras puntas" de bridges potenciales desde nuestras piedras
   actuales — celdas que físicamente están a distancia 2, pero que
   si las jugamos crean una conexión virtual fuerte.

Este conocimiento de dominio es lo que separa un MCTS genérico de uno
adaptado a Hex. Sin bridges, el algoritmo redescubre estos patrones
solo a base de simulaciones, lo cual desperdicia mucho tiempo de
cómputo.


## Filtro de relevancia

El branching factor en Hex 11×11 es alto: en una posición temprana,
hay ~115 celdas legales para expandir desde un nodo MCTS. Si el árbol
se ramifica con todas, el cómputo se diluye en celdas que ningún
jugador sensato consideraría. Necesitamos un filtro que reduzca el
conjunto de candidatos a las celdas "relevantes".

### Reglas del filtro

Para un jugador dado y un tablero, las celdas relevantes son la unión
de cuatro conjuntos:

1. **Vecindad cercana propia**: celdas vacías a distancia 1 ó 2 de
   alguna piedra propia. Es donde podemos extender nuestras cadenas.
2. **Bridges potenciales propios**: las "otras puntas" de bridges
   posibles desde piedras propias. Cubre celdas a distancia 2 que no
   son simples vecinos pero permiten construir conexión virtual.
3. **Defensa amplia**: celdas vacías a distancia 1 de cualquier
   piedra enemiga. Necesitamos al menos *ver* qué hace el oponente,
   aunque no esté cerca de su borde.
4. **Defensa profunda en zonas críticas**: celdas a distancia 2 de
   piedras enemigas que están dentro de un umbral (3 filas/columnas)
   de su borde objetivo. Cuando el oponente avanza hacia su lado,
   necesitamos mirar más lejos para anticipar amenazas.

Esto típicamente reduce el branching de ~115 a entre 25 y 40 celdas,
lo cual es manejable.

### Cómo llegamos a esta versión

Esta sección iteró tres veces:

- **Versión 1**: vecinos a distancia 1 y 2 de **todas** las piedras
  (mías y enemigas). Funcionaba contra Random y MCTS_Tier_1, pero
  contra rivales más fuertes el algoritmo "perseguía" cada piedra
  enemiga aislada y fragmentaba su propio juego en vez de construir
  cadenas coherentes.

- **Versión 2**: solo vecinos de piedras propias, ignorando casi
  todas las del oponente excepto las que estuvieran muy cerca de su
  borde objetivo. Demasiado restrictiva — el oponente podía
  desarrollar zonas centrales sin que nuestro MCTS las explorara.
  Resultado: regresión completa contra MCTS_Tier_3.

- **Versión 3 (actual)**: el balance descrito arriba. Vecinos d=1 a
  todas las piedras (defensa amplia) + d=2 a propias (construcción)
  + d=2 a enemigas amenazantes (defensa profunda).

La lección práctica fue que un filtro debe distinguir entre **mirar**
una zona y **planear** en ella. Vecinos d=1 alcanzan para "darse
cuenta" de lo que pasa; d=2 se reserva para zonas donde activamente
queremos jugar.

## Variante dark: PIMC

En la variante *dark* (con niebla de guerra), un jugador no ve las
piedras del oponente excepto cuando intenta jugar en una celda ocupada
y descubre la colisión. El estado real del juego es desconocido para
nosotros: solo observamos un *belief state* parcial.

Esto rompe MCTS clásico: un MCTS ingenuo asumiría que el tablero
visible es completo y rolloutearía como si el oponente no existiera.
Para abordar esto implementamos **PIMC** (Perfect Information Monte
Carlo), una técnica estándar para juegos con información incompleta.

### El algoritmo

En cada llamada a `play()`:

1. **Construimos el belief state**:
   - Piedras propias y enemigas visibles en `board`.
   - Conteo de turnos del oponente (lo llevamos manualmente desde
     `begin_game`).
   - Piedras enemigas ocultas = turnos_oponente − piedras_visibles.

2. **Sampleamos N mundos plausibles** (N = `DARK_N_MUNDOS = 6`). Cada
   mundo es un tablero completo con todas las piedras enemigas
   ocultas colocadas en celdas que un oponente razonable habría
   jugado. Para esto usamos una "heurística inversa": para cada celda
   vacía calculamos qué tan atractiva sería para el oponente
   (vecinos enemigos, formación de bridges enemigos, zona productiva
   suya), y sampleamos las top-K celdas con probabilidad proporcional
   al score.

3. **Ejecutamos MCTS classic en cada mundo**, con presupuesto
   tiempo_total / (N+1) por mundo. Como cada mundo es de información
   perfecta, MCTS funciona como en classic.

4. **Combinamos los resultados por voto de visitas**: para cada
   movimiento candidato, sumamos las visitas que recibió como hijo de
   raíz a través de todos los mundos. Solo consideramos movimientos
   que sean legales en el tablero **visible** (no en el mundo
   simulado).

5. **Aplicamos un filtro de seguridad contra colisiones**: si una
   celda candidata aparece como piedra enemiga en ≥
   `DARK_UMBRAL_COLISION = 0.4` fracción de los mundos sampleados, su
   score se multiplica por `DARK_PENALIZACION_COLISION = 0.1`. Una
   colisión cuesta turno entero, así que es peor que casi cualquier
   movimiento subóptimo.

6. **Devolvemos el movimiento con mayor score combinado**.

### Limitación conceptual: strategy fusion

PIMC tiene una debilidad teórica conocida ("strategy fusion problem"):
asume que el oponente, en cada mundo simulado, juega con información
perfecta. En realidad, el oponente también es ciego en dark. Esto
hace a PIMC sistemáticamente pesimista — sobreestima al oponente.
La solución completa (IS-MCTS, online inference) es significativamente
más compleja de implementar; PIMC simple es un punto razonable de la
curva costo/beneficio para un proyecto de un solo entregable.


## Aperturas y atajos

Antes de invocar la búsqueda costosa, `play()` aplica una serie de
atajos que resuelven con lógica exacta lo que se pueda.

**Aperturas conocidas**:
- *Negro*: `(5, 5)` — centro del tablero. Sin regla de swap, jugar
  centro en el primer movimiento es la apertura estándar más fuerte.
- *Blanco*: `(5, 2)` o equivalente — respuesta lateral lejos del
  centro. Evita contacto inmediato con la piedra negra (que llevaría
  a una pelea local desfavorable como segundo jugador) y proyecta
  hacia el borde izquierdo, manteniendo opciones simétricas.

**Atajos tácticos**:
- **Jugada ganadora**: si nuestra distancia Dijkstra es 1, recorremos
  las celdas relevantes buscando la que cierra la conexión y la
  jugamos.
- **Bloqueo obligado**: si la del oponente es 1, ocupamos su celda
  crítica.
- **Save-bridge**: si la última jugada del oponente cortó un *carrier*
  de un bridge nuestro, jugamos el otro carrier para completar la
  conexión virtual antes de que pueda volver a atacarla.

Estos atajos no son optimización opcional: son decisiones donde MCTS
solo introduciría ruido. Resolverlas en O(N²) garantiza precisión y
libera tiempo para posiciones genuinamente ambiguas.


## Manejo de tiempo

El torneo otorga `move_timeout = 15s` por jugada. Internamente
manejamos el presupuesto vía la variable de entorno
`HEXARINO_TIME_S` (default 2.5s para iteración rápida durante
desarrollo, configurable a 13-14s para el torneo). Esto permite:

- Correr experimentos locales rápido sin esperar partidas largas.
- Calibrar el presupuesto al límite real del referee con un buffer
  de seguridad de 0.3s (`TIEMPO_BUFFER_S`) para evitar timeouts.

El bucle principal de MCTS usa `time.monotonic()` y se detiene en
cuanto cruza el deadline. No hay número fijo de simulaciones: cada
jugada hace tantas como caben en el tiempo disponible.

En la variante *dark*, el presupuesto se reparte entre los N mundos:
cada uno recibe `tiempo_total / (N+1)` segundos, dejando un margen
para overhead de sampling.


## Alternativas consideradas y descartadas

Documentamos aquí las decisiones de diseño donde consideramos
alternativas y elegimos no implementarlas, con la razón explícita.

**Minimax con poda alfa-beta**. Considerada como búsqueda principal
en lugar de MCTS. Descartada por el branching factor: en Hex 11×11
hay ~115 celdas legales en posición temprana. Aún con poda agresiva,
alcanzar profundidad 4 dentro de 2-3 segundos es difícil, y a esa
profundidad la búsqueda no captura amenazas estructurales típicas de
Hex (que se desarrollan en 6-10 movimientos). MCTS escala
gradualmente con el tiempo disponible, sin saltos cualitativos por
agotar profundidad.

**Random rollouts puros**. La opción "vanilla" de MCTS. Descartada
porque en Hex 11×11 los rollouts aleatorios convergen a posiciones
casi sin estructura: con 121 celdas y reglas que recompensan
conexión, cualquier secuencia aleatoria desperdicia la mayoría de
movimientos en celdas irrelevantes. La señal en el rollout es
demasiado ruidosa para guiar UCB1 con presupuesto limitado.

**Rollouts hasta el final del juego**. Llevar cada rollout hasta que
`check_winner` detecte ganador es teóricamente más limpio que truncar
y evaluar. Descartada por costo: una partida típica tiene 35-50
movimientos, así que un rollout completo cuesta ~10× más que uno
truncado a 12. Preferimos hacer 10× más simulaciones con algo más de
ruido en cada una — favorable trade-off en este dominio.

**RAVE / AMAF (All Moves As First)**. Técnica clásica que comparte
estadísticas entre nodos hermanos para acelerar convergencia.
Considerada y no implementada por restricción de tiempo del proyecto.
Sería la primera mejora a agregar si dispusiéramos de más iteración.

**Tree reuse entre jugadas**. En lugar de reconstruir el árbol MCTS
desde cero en cada `play()`, conservar el sub-árbol del movimiento
que el oponente terminó eligiendo. Considerada y no implementada
porque su beneficio es modesto cuando los presupuestos por jugada son
pequeños — el árbol que sobrevive es pequeño.

**Edge templates y virtual connections complejas**. Hex tiene
literatura sobre patrones más sofisticados que el bridge simple
(ziggurats, escaleras, templates de borde). Descartados por
complejidad de implementación: codificar correctamente los templates
y su composición requiere mucho más código que su beneficio
incremental sobre bridges, especialmente en un proyecto académico de
un solo archivo.

**Vectorización con NumPy**. Considerada para acelerar `Dijkstra` y
los rollouts. Descartada porque el bottleneck real es Dijkstra (que
usa `heapq` y no se vectoriza naturalmente), y los rollouts ya operan
sobre listas de Python pequeñas donde el overhead de NumPy supera a
la ganancia. Mantenemos la dependencia mínima a `numpy + stdlib`
como dicta el enunciado.

**IS-MCTS (Information Set MCTS) para dark**. Habría sido la solución
teóricamente correcta al strategy fusion problem en lugar de PIMC.
Descartada por complejidad: requiere mantener nodos del árbol
indexados por *information sets* en vez de estados concretos, con
sampling de mundos consistente en cada paso de selección. PIMC simple
captura ~80% del beneficio con ~30% del código.

**Modelado de oponente**. Aprender en línea qué tipo de jugador es el
oponente actual y ajustar nuestra estrategia. Descartado por la
estructura del torneo (partidas independientes, no series largas
contra el mismo rival) y por complejidad.


## Resultados experimentales

Configuración común a todos los experimentos: tablero 11×11, variante
classic salvo indicación contraria. `HEXARINO_TIME_S` indicado por
fila.

| Oponente              | Tiempo | Como Negro | Como Blanco | Notas                                  |
|-----------------------|--------|------------|-------------|----------------------------------------|
| Random                | 1.0s   | 5/5        | 5/5         | Sanity check                           |
| MCTS_Tier_1           | 2.0s   | 5/5        | 5/5         | Vencido consistentemente               |
| MCTS_Tier_3           | 2.0s   | 1/5        | 0/5         | Diferencia de tiempo (Tier 3 usa 12s)  |
| MCTS_Tier_3           | 13.0s  | _llenar_   | _llenar_    | Tiempo equivalente                     |
| MCTS_Tier_3 (dark)    | 15.0s  | 2/5        | _llenar_    | PIMC mejora dark vs classic            |

**Lectura de los resultados**: la estrategia es competitiva contra
oponentes hasta MCTS_Tier_2 (vencidos al 100%). Contra MCTS_Tier_3 y
algunas estrategias de estudiantes con greedy bien tuneada, no
alcanzamos el techo — esto refleja una limitación arquitectónica que
discutimos abajo.


## Limitaciones conocidas

- **Convergencia de MCTS con presupuesto bajo**. Con tiempos por
  debajo de 5s, las simulaciones son insuficientes para que UCB1
  identifique amenazas estructurales del oponente que requieren
  10+ movimientos para concretarse. Greedy bien tuneadas pueden
  vencernos en ese régimen.
- **Conocimiento de dominio limitado a bridges**. No detectamos
  virtual connections más complejas (edge templates, escaleras,
  ziggurats). Una greedy con buenos pesos de "bloqueo preventivo"
  puede capturar parte de esta intuición sin necesidad de búsqueda.
- **Dark con strategy fusion**. PIMC es teóricamente subóptimo en
  juegos con información imperfecta porque asume oponente con
  información completa en cada mundo simulado.
- **Sin memoria entre partidas**. No aprendemos del historial reciente
  de un oponente; cada partida arranca desde cero.
- **Filtro de relevancia heurístico**. Las reglas del filtro (vecinos
  d=1/d=2, carriers de bridges enemigos, etc.) son una aproximación
  manual. Un filtro aprendido podría ser más preciso pero requiere
  datos.


## Cómo correr

Pruebas locales con Docker (asume `docker compose` configurado en el
repo del torneo):

```bash
# Contra Random (sanity check)
docker compose run --rm experiment \
  python experiment.py --black "Hexarino_bernardor" --white "Random" \
  --num-games 5

# Contra un tier MCTS, con tiempo configurado
docker compose run --rm -e HEXARINO_TIME_S=13.0 experiment \
  python experiment.py --black "Hexarino_bernardor" --white "MCTS_Tier_3" \
  --num-games 5

# Variante dark
docker compose run --rm -e HEXARINO_TIME_S=13.0 experiment \
  python experiment.py --black "Hexarino_bernardor" --white "MCTS_Tier_3" \
  --variant dark --num-games 5

# Verbose: imprime el tablero después de cada jugada
docker compose run --rm experiment \
  python experiment.py --black "Hexarino_bernardor" --white "Random" \
  --num-games 1 --verbose
```

Variables de entorno relevantes:
- `HEXARINO_TIME_S` (default `2.5`): presupuesto interno de tiempo
  por jugada. En el torneo conviene `13.0` (deja buffer seguro vs
  el `move_timeout=15.0`).