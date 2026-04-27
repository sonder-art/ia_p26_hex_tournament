# Estrategia de nuestro equipo: julianTania2 con estrategia HexForge

**HexForge** es una estrategia para el juego Hex 11×11 diseñada para competir tanto en la variante **classic** como en la variante **dark**. La idea central de la estrategia es combinar búsqueda tipo Monte Carlo con una capa táctica y heurística más fuerte que la de nuestra versión inicial, con el objetivo de tomar mejores decisiones en menos tiempo y reducir errores en posiciones críticas.

A diferencia de una estrategia puramente basada en simulaciones aleatorias, HexForge intenta concentrar el presupuesto de tiempo en jugadas relevantes del tablero: amenazas inmediatas, conexiones importantes, bloqueos urgentes y movimientos que fortalecen estructuras favorables. El resultado es una estrategia más agresiva, más consistente y mejor adaptada a partidas reales del torneo.

---

## 1. Versión inicial: HexMaster

Nuestra primera versión, **HexMaster**, utilizaba como núcleo principal un esquema de **Monte Carlo Tree Search (MCTS)**. La idea era simular muchas partidas posibles dentro del tiempo disponible y elegir la jugada con mejores estadísticas acumuladas. Esta versión ya incluía varias optimizaciones de rendimiento para poder explorar más posiciones por turno.

### 1.1 Funcionamiento general de HexMaster

La versión inicial seguía este flujo:

1. Revisar si había celdas vacías.
2. Jugar de forma simple en la apertura, priorizando el centro.
3. En dark mode, construir una determinización sencilla del tablero adivinando posiciones ocultas del rival.
4. Ejecutar MCTS mientras hubiera tiempo disponible.
5. Si el tiempo se agotaba, usar una heurística rápida basada en distancia Dijkstra.

### 1.2 Componentes de HexMaster

#### Apertura simple
Durante los primeros movimientos, HexMaster evitaba gastar demasiado tiempo de búsqueda y prefería jugar en el centro o cerca del centro para obtener una posición razonable de arranque.

#### MCTS clásico
La estrategia principal era MCTS con sus cuatro fases típicas:
- selección,
- expansión,
- rollout,
- backpropagation.

Los nodos se elegían con una fórmula UCT para balancear exploración y explotación. En expansión, si había pocos candidatos se usaba una heurística con Dijkstra; si había muchos, se elegía uno aleatoriamente. 

#### Rollouts rápidos
Las simulaciones del resto de la partida se hacían de forma rápida para maximizar la cantidad de iteraciones. Para eso se reutilizaba memoria y se usaba una detección eficiente de victoria mediante BFS. 

#### Dark mode por determinización simple
En la variante dark, HexMaster generaba un tablero ficticio colocando piedras ocultas del rival en posiciones vacías estimadas de manera probabilística. Esto permitía correr MCTS sobre una versión “completa” del tablero, aunque con una aproximación relativamente cruda del estado real.

### 1.3 Fortalezas y limitaciones de HexMaster

HexMaster funcionó como una primera base sólida porque:
- era rápido,
- aprovechaba bien el tiempo por jugada,
- y permitía explorar muchas posibilidades.

Sin embargo, su principal limitación era que dependía demasiado de simulaciones relativamente genéricas. En posiciones tácticas o de alta sensibilidad, esa aproximación podía dejar pasar amenazas inmediatas, puentes importantes o respuestas críticas. Además, en dark mode la determinización era útil, pero demasiado simple para capturar bien la incertidumbre real del juego. 

---

## 2. Nueva versión: HexForge

**HexForge** es la evolución directa de HexMaster. Mantiene la idea de usar búsqueda y simulación, pero reemplaza una parte importante del comportamiento anterior por una estrategia más guiada y más táctica.

El objetivo del rediseño fue doble:

1. **mejorar la calidad de las jugadas**, especialmente en posiciones de conflicto;
2. **reducir el desperdicio de tiempo** en movimientos poco relevantes.

En vez de confiar tanto en rollouts amplios y relativamente neutros, HexForge intenta decidir mejor desde el principio qué movimientos merecen atención.

---

## 3. Estrategia actual de HexForge

### 3.1 Capa táctica previa a la búsqueda

Antes de arrancar la búsqueda principal, HexForge ejecuta una fase táctica corta cuyo objetivo es detectar situaciones en las que una sola jugada decide la partida o cambia por completo el equilibrio local. Esta fase se ejecuta sobre un conjunto reducido de movimientos prioritarios para no gastar tiempo revisando todo el tablero cuando no hace falta.

La primera decisión importante es **qué conjunto revisar tácticamente**. Si quedan más de **24 casillas vacías**, la verificación táctica se hace solo sobre el conjunto de jugadas candidatas que ya fueron priorizadas por la heurística de raíz. Si quedan **24 o menos**, entonces sí se revisan todas las casillas vacías. La razón es que en el midgame temprano el costo de verificar 40, 50 o más celdas sería demasiado alto, pero en posiciones más cerradas sí conviene revisar exhaustivamente.

Sobre ese conjunto táctico se aplican dos pruebas exactas:

1. **Victoria inmediata propia**: para cada candidato se coloca temporalmente nuestra piedra y se verifica si ya existe un camino ganador. Si esto ocurre, la jugada se devuelve inmediatamente sin pasar por MCTS.
2. **Victoria inmediata del rival**: si no tenemos una victoria inmediata, se repite el mismo chequeo pero colocando temporalmente una piedra del oponente. Si el rival podría ganar en una jugada concreta, entonces esa casilla se juega como bloqueo obligatorio.

Ejemplo numérico. Supongamos que somos jugador 1 y quedan 18 celdas vacías. Como **18 ≤ 24**, la revisión táctica se hace sobre las 18. Si entre ellas la casilla `(5, 6)` completa una conexión superior-inferior para nosotros, se juega `(5, 6)` de inmediato. Si no existe tal jugada, pero la casilla `(4, 7)` le daría la victoria inmediata al rival si él la jugara en su turno, entonces HexForge responde con `(4, 7)` antes de iniciar cualquier simulación.

Esta capa no usa poda alfa-beta todavía; aquí no se está explorando un árbol profundo, sino resolviendo tácticas de **profundidad 1**. La ventaja es que evita que el MCTS desperdicie iteraciones redescubriendo algo que ya era forzado desde el principio.

---

### 3.2 Búsqueda Monte Carlo más guiada

La búsqueda principal de HexForge sigue siendo **Monte Carlo Tree Search**, pero no en su forma más genérica. No se construye un árbol grande en profundidad como en un MCTS clásico con expansión de muchos nodos internos; en cambio, se hace una **búsqueda fuerte en la raíz** con selección tipo **UCB** y con una fuerte priorización heurística de movimientos.

El procedimiento es este:

1. Se calculan puntuaciones heurísticas para las casillas vacías.
2. Solo se conserva un subconjunto de candidatas llamado `root_moves`.
3. Sobre esas candidatas se reparte el presupuesto de simulación usando una fórmula UCB con sesgo progresivo.

El tamaño de `root_moves` no es fijo. Depende del número de vacías:

- si hay más de **80** vacías: se conservan **10** candidatas;
- si hay entre **56 y 80**: se conservan **12**;
- si hay entre **36 y 55**: se conservan **14**;
- si hay entre **21 y 35**: se conservan **16**;
- si hay **20 o menos**: se conservan hasta **24**.

Esto ya es una forma de poda, aunque no sea poda alfa-beta. En vez de explorar todo el branching factor del tablero, se reduce desde el principio a un conjunto pequeño de jugadas plausibles.

Luego, en cada iteración del MCTS de raíz, la selección del movimiento se hace con una variante de **UCB con progressive bias**:

\[
UCB(i)=\bar{X}_i + c\sqrt{\frac{\ln(N+1)}{n_i}} + \frac{0.20\cdot prior_i}{n_i+1}
\]

donde:

- \(\bar{X}_i\) es la media de victorias del movimiento \(i\),
- \(N\) es el número total de simulaciones en la raíz,
- \(n_i\) es el número de veces que se ha probado ese movimiento,
- `prior_i` es una prioridad heurística normalizada entre 0 y 1,
- y \(c\) vale:
  - **1.28** si quedan más de **28** vacías,
  - **0.95** si quedan **28 o menos**.

Esto significa que en el juego temprano y medio la estrategia explora un poco más, mientras que en el juego tardío explota más lo que ya parece bueno.

Ejemplo numérico. Supongamos que quedan 40 casillas vacías, así que el tope en raíz es **14** candidatos. Imaginemos que dos movimientos A y B tienen:

- A: media \(0.63\), \(n_A=12\), `prior=0.90`
- B: media \(0.58\), \(n_B=4\), `prior=0.70`

Si \(N=50\), entonces:
- A recibe un bono exploratorio menor porque ya fue visitado más veces,
- B recibe más exploración por tener \(n_B\) más pequeño,
- pero A mantiene una ventaja estructural porque parte de una media mejor y además un `prior` mayor.

En consecuencia, HexForge no elige solo por win rate ni solo por heurística, sino por una combinación de ambas.

**No se usa alfa-beta como método principal del MCTS.** La búsqueda global sigue siendo Monte Carlo guiada por UCB. La única parte donde sí se usa **negamax con poda alfa-beta** es en el final exacto, como se explica más abajo.

---

### 3.3 Priorización de conexiones y estructuras locales

El cambio más fuerte respecto a HexMaster está en la función de puntuación de candidatos. En HexForge, cada casilla vacía recibe una puntuación compuesta por varios términos estructurales. La estrategia ya no trata todas las casillas como “igual de simulables”, sino que intenta estimar cuáles realmente mueven la partida.

La heurística mezcla cinco grupos de señales:

#### a) Distancia propia y distancia rival

Para cada jugador se calculan mapas de distancia tipo **0-1 BFS** hacia ambos bordes objetivo. Las casillas ya ocupadas por el jugador cuentan con costo 0 y las vacías con costo 1. Con esto se obtiene:

- `my_best`: mejor distancia global de nuestro camino,
- `opp_best`: mejor distancia global del rival,
- `my_through(idx)`: distancia si nuestro camino pasa por `idx`,
- `opp_through(idx)`: distancia si el del rival pasa por `idx`.

Luego se suma:

- **24.0 × (2·size − my_through)** para premiar casillas que abaratan mucho nuestro camino,
- **8.0 × max(0, my_best − (my_through−1))** para premiar mejoras reales sobre nuestro mejor camino actual,
- **22.0 × (2·size − opp_through)** para jugadas que interfieren trayectorias críticas del rival,
- **7.0 × max(0, opp_best − opp_through)** para premiar bloqueos que empeoran de verdad el camino contrario.

Ejemplo. En tablero 11×11, \(2\cdot size = 22\). Si una casilla da `my_through = 9`, aporta:

\[
24(22-9)=24\cdot13=312
\]

solo en el término principal de conexión propia. Si además `my_best = 10`, entonces:

\[
8\cdot \max(0,10-(9-1))=8\cdot2=16
\]

extra por mejorar el mejor camino conocido. Una casilla así tiene mucho peso aunque no sea central geométricamente.

#### b) Adyacencia a grupos propios y enemigos

Se construyen componentes conexas con una estructura tipo **DSU/Union-Find**. Para cada candidata se mide:

- cuántos vecinos propios toca,
- cuántos vecinos rivales toca,
- cuántas componentes distintas propias conecta,
- cuántas componentes distintas rivales interfiere.

Los términos exactos son:

- **7.0** por cada vecino propio adyacente,
- **5.0** por cada vecino rival adyacente,
- **14.0 × (número de componentes propias distintas tocadas − 1)**,
- **12.0 × (número de componentes rivales distintas tocadas − 1)**.

Esto quiere decir que una casilla que une **dos grupos propios desconectados** recibe al menos un bono extra de **14**, mientras que una que toca **tres grupos propios** recibe **28** por ese solo concepto.

#### c) Contacto con bordes objetivo

Cada componente lleva un bitmask que indica qué borde relevante toca. Si al jugar una casilla el conjunto unido toca ambos lados de nuestro objetivo, el bono es muy alto:

- **+120.0** si la unión propia resultante toca ambos bordes,
- **+90.0** si la jugada corta o intercepta una estructura rival que toca ambos bordes.

Si solo se toca un borde, el bono es parcial:
- **+12.0** por cada borde propio alcanzado,
- **+10.0** por cada borde rival implicado.

Esto es importante porque obliga al algoritmo a valorar más una casilla que “conecta de verdad” que otra que simplemente está cerca de piedras amigas.

#### d) Centralidad y alineación axial

Además del valor táctico, se añade un sesgo posicional:

- **1.6 × center(idx)**, donde `center` mide cercanía al centro hexagonal,
- **2.3 × axis(idx)**, donde `axis` depende del jugador:
  - para jugador 1 se favorece la cercanía al eje vertical útil,
  - para jugador 2 se favorece la cercanía al eje horizontal útil.

Ejemplo: una casilla central con `center=10` y `axis=9` aporta:

\[
1.6\cdot10 + 2.3\cdot9 = 16 + 20.7 = 36.7
\]

antes de contar cualquier otra señal. No es suficiente para ganar sola, pero sí sirve como desempate entre dos jugadas tácticamente parecidas.

#### e) Bridges y conexiones virtuales locales

HexForge incorpora un bono específico para patrones tipo **bridge**. Si una casilla está entre dos piedras propias no adyacentes pero con una estructura de conexión local plausible, se añade:

- **+6** si el otro punto auxiliar del bridge está vacío,
- **+10** si ese punto ya es nuestro,
- **+8** si el punto está ocupado por el rival, porque sigue siendo una zona crítica de conflicto.

La estrategia también suma:
- **1.0 × bridge_bonus propio**
- **0.85 × bridge_bonus rival**

Es decir, valora ligeramente más construir su propio bridge que destruir el del oponente, pero toma ambas cosas en cuenta.

En conjunto, esta heurística hace que la búsqueda de raíz no empiece “ciega”, sino muy bien informada estructuralmente.

---

### 3.4 Rollouts menos ingenuos

HexMaster dependía mucho de la cantidad de rollouts. En HexForge, el rollout se volvió más sesgado y más corto, con evaluación heurística al final.

En lugar de simular hasta el final del tablero siempre, HexForge usa una política de profundidad limitada:

- si quedan más de **70** vacías: profundidad **10**
- si quedan entre **46 y 70**: profundidad **14**
- si quedan entre **26 y 45**: profundidad **18**
- si quedan entre **15 y 25**: profundidad **22**
- si quedan **14 o menos**: simula hasta el final

Dentro del rollout, cada jugada no se elige al azar puro. En cada paso se muestrean hasta **5 candidatos** aleatorios de las vacías y se selecciona el mejor de esos 5 con esta puntuación:

\[
8\cdot same - 6\cdot enemy + 1.7\cdot center + 1.9\cdot axis + 0.6\cdot empty
\]

más:
- **+6** si tiene al menos 2 vecinos propios,
- **+5** si tiene al menos 2 vecinos rivales,
- y un pequeño ruido aleatorio de hasta **1.8** para romper empates.

Aquí:
- `same` = vecinos propios,
- `enemy` = vecinos rivales,
- `empty` = vecinos vacíos.

Ejemplo. Si una casilla en rollout tiene:
- 2 vecinos propios,
- 1 vecino rival,
- 3 vecinos vacíos,
- `center=8`,
- `axis=9`,

su score sería:

\[
8(2)-6(1)+1.7(8)+1.9(9)+0.6(3)+6 = 16-6+13.6+17.1+1.8+6 = 48.5
\]

Si otra candidata sale con 35 o 40, el rollout jugará la de 48.5. Esto vuelve los rollouts más “humanos” y menos erráticos.

Además, cada cierto número de pasos se verifica si ya hay una victoria explícita, para cortar temprano si la simulación ya quedó decidida.

---

### 3.5 Mejor manejo de dark mode

La mejora en dark mode no consiste solo en “adivinar mejor”. La estrategia ahora modela explícitamente la incertidumbre en tres niveles.

#### a) Estimación del número de piedras ocultas

HexForge estima cuántas piedras del rival podrían existir sin ser visibles. Para eso usa:

- `opp_turns`: número estimado de turnos que ya tuvo el oponente,
- `visible_opp`: piedras rivales visibles en el tablero.

Luego calcula:

\[
base = \max(0, opp\_turns - visible\_opp)
\]
\[
hidden\_count = round(0.72 \cdot base)
\]

Ese factor **0.72** amortigua la estimación para no sobrepoblar el tablero con piedras ocultas ficticias, ya que en dark también existen colisiones y jugadas fallidas.

Ejemplo. Si se estima que el rival ya tuvo 10 turnos y solo vemos 6 piedras:

\[
base = 10 - 6 = 4
\]
\[
hidden\_count = round(0.72 \cdot 4)=3
\]

Entonces la estrategia asume aproximadamente **3 piedras ocultas**.

#### b) Distribución no uniforme de probabilidad oculta

No todas las vacías son igualmente probables como escondite del rival. HexForge primero puntúa las casillas **desde la perspectiva del oponente** usando la heurística de candidatos en modo classic. Luego normaliza esos valores y asigna:

\[
p(idx)=0.15 + 0.85 \cdot \frac{s(idx)-s_{min}}{s_{max}-s_{min}}
\]

Así, incluso la peor casilla conserva probabilidad base **0.15**, pero las más plausibles se acercan a **1.0**.

#### c) Ensemble de varios mundos plausibles

En lugar de una sola determinización, HexForge evalúa varios mundos posibles:

- si quedan más de **70** vacías: **3 mundos**
- si quedan entre **36 y 70**: **4 mundos**
- si quedan **35 o menos**: **2 mundos**

En cada mundo se toma un `hidden_count` ligeramente distinto entre:
- `hidden_count - 1`,
- `hidden_count`,
- `hidden_count + 1`.

Luego se generan determinizaciones muestreando desde un pool de las casillas más plausibles, con tamaño:

\[
\min(\text{vacías}, \max(18, 5\cdot hidden\_count))
\]

y pesos:
\[
0.05 + p(idx)
\]

Cada mundo propone una jugada con su propia búsqueda de raíz, y al final se agregan los votos. La jugada final es la que gana más mundos plausibles.

Ejemplo. Si `hidden_count = 3` y quedan 50 vacías, HexForge puede evaluar 4 mundos con conteos \([2,3,4,2]\) y luego quedarse con la jugada que aparece más veces como mejor respuesta.

Esta técnica no es alfa-beta ni minimax clásico; es un **ensemble de determinizaciones** con MCTS guiado en cada una.

---

### 3.6 Mejor uso del tiempo

HexForge ajusta mejor el presupuesto según el tipo de posición. En general usa aproximadamente el **88%** del límite por jugada para dejar margen de seguridad.

Además, el reparto no es uniforme:

- en apertura, se evita gastar demasiado porque la respuesta suele salir de un repertorio local alrededor del centro;
- en posiciones con muchas vacías, se reduce el número de candidatos raíz para no dispersar la exploración;
- en finales pequeños, se abandona la simulación y se cambia a búsqueda exacta.

La apertura, por ejemplo, intenta una lista fija de hasta **7 posiciones** alrededor del centro:
- centro,
- arriba y abajo del centro,
- izquierda y derecha,
- y dos diagonales cercanas.

Esto evita gastar MCTS en posiciones donde una apertura central razonable ya es suficientemente fuerte.

La optimización más importante del final es que, si la partida está en classic y quedan **9 o menos vacías**, se activa un **negamax exacto con poda alfa-beta**. Esta es la única parte del algoritmo que sí usa alfa-beta de forma explícita.

El solver devuelve:
- **+1** si el jugador al turno tiene victoria forzada,
- **−1** si tiene derrota forzada.

Para ordenar las ramas del negamax, primero reutiliza la misma heurística de candidatos, lo que hace que la poda alfa-beta corte antes y sea más efectiva.

Ejemplo. Si quedan 7 casillas vacías, el branching factor máximo bruto sería \(7!\) en una exploración ingenua, pero con:
- ordenamiento heurístico,
- poda alfa-beta,
- memoización por `(tablero, jugador)`,

el árbol real se reduce muchísimo y permite resolver exactamente posiciones que MCTS solo aproximaría.

---

## 4. Cambios principales respecto a HexMaster

Los cambios más importantes respecto a HexMaster fueron los siguientes.

### Cambio 1: de expansión amplia a poda fuerte en la raíz
HexMaster distribuía más trabajo entre muchas jugadas posibles y dependía de que los rollouts separaran las buenas de las malas.  
HexForge reduce explícitamente el branching factor con top-k en raíz: 10, 12, 14, 16 o 24 candidatos según el número de vacías. Eso hace la búsqueda más profunda y mejor enfocada sobre movimientos relevantes.

### Cambio 2: de táctica implícita a táctica explícita
HexMaster confiaba en que una jugada ganadora o un bloqueo urgente aparecieran como estadísticamente buenos dentro del MCTS.  
HexForge primero revisa de forma exacta victorias inmediatas y defensas obligatorias. Eso elimina errores donde la simulación no alcanzaba a detectar amenazas de una jugada.

### Cambio 3: de evaluación por distancia a evaluación estructural compuesta
HexMaster usaba mucho la señal de distancia Dijkstra.  
HexForge sigue usando mapas de distancia, pero además agrega:
- unión de componentes,
- contacto con bordes,
- número de grupos conectados,
- bridges propios y rivales,
- centralidad y eje útil.

Es decir, ya no pregunta solo “qué tan corto es mi camino”, sino también “qué tan estable, conectado y tácticamente fuerte es”.

### Cambio 4: de una determinización a un ensemble de mundos
HexMaster usaba una sola reconstrucción probabilística del tablero en dark.  
HexForge estima cuántas piedras ocultas hay, distribuye probabilidades no uniformes sobre las vacías y evalúa varios mundos plausibles. Esto reduce mucho el riesgo de comprometer toda la decisión a una sola hipótesis equivocada.

### Cambio 5: de rollouts neutrales a rollouts sesgados
HexMaster simulaba más, pero con menor calidad local.  
HexForge reduce la profundidad de rollout cuando conviene y escoge jugadas intermedias con una política sesgada por vecindad, centralidad y alineación. Esto vuelve cada simulación individual más informativa.

### Cambio 6: incorporación de solver exacto al final
HexMaster se mantenía en modo heurístico incluso en finales pequeños.  
HexForge cambia a **negamax con poda alfa-beta y memoización** cuando quedan 9 o menos vacías en classic. Ese cambio fue importante para no perder finales tácticos que ya eran resolubles exactamente.

---

## 5. Flujo general de decisión de HexForge

De manera simplificada, el flujo actual de la estrategia es:

```text
Inicio del turno
    ↓
Aplanar tablero y obtener vacías
    ↓
Si es apertura temprana, jugar de una lista local alrededor del centro
    ↓
Generar top-k candidatos estructurales
    ↓
Buscar victoria inmediata propia
    ↓
Buscar victoria inmediata del rival y bloquear si es necesario
    ↓
Si es final pequeño en classic (≤ 9 vacías), usar negamax exacto con poda alfa-beta
    ↓
Si es classic, ejecutar MCTS guiado en raíz con UCB + progressive bias
    ↓
Si es dark, estimar hidden_count, generar varios mundos plausibles y correr búsqueda en cada uno
    ↓
Agregar resultados y elegir la mejor jugada