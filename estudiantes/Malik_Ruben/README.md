# FogBridge_MALIK_RUBEN

## Resumen

La estrategia implementada en `strategy.py` se llama **`FogBridge_MALIK_RUBEN`** y juega Hex 11x11 en las dos variantes del torneo:

- `classic`, con informacion completa
- `dark`, con fog of war donde solo observamos nuestras propias piedras y las colisiones del rival

La version final se apoyo en tres capas principales y un ajuste extra por color:

- en `classic`, una combinacion de heuristicas estructurales de Hex + una busqueda tipo **flat root Monte Carlo**
- en finales de `classic`, un **solver exacto de endgame** para reducir ruido cuando ya quedan pocas casillas
- en `dark`, una estrategia **mucho mas conservadora**, enfocada en construir cadenas solidas y aprovechar el hecho de que el rival no ve nuestras piedras
- en `classic`, cuando juega **Blanco**, una evaluacion en **dos etapas** para filtrar candidatas baratas antes de gastar el costo completo de la evaluacion adversarial

La motivacion principal fue construir una estrategia fuerte dentro de las restricciones reales del torneo:

- un solo archivo evaluado
- maximo 15 segundos por jugada
- solo `numpy` + biblioteca estandar
- compatibilidad obligatoria con `classic` y `dark`

## Idea central

La estrategia intenta maximizar dos cosas al mismo tiempo:

- nuestra conectividad real hacia los bordes objetivo
- la dificultad del rival para cerrar su propia conexion

Para eso, el agente usa como señal base la distancia minima de conexion calculada con `shortest_path_distance`, y luego la corrige con informacion local de forma:

- adyacencias utiles
- fusion de grupos
- puentes
- casillas criticas de corredores cortos
- cercania a los bordes importantes

La intuicion es que en Hex no basta con jugar “cerca”; hay que jugar donde una piedra:

1. mejora tu cadena de verdad
2. o rompe una cadena rival de verdad
3. o idealmente hace ambas cosas

## Algoritmo

## 1. Apertura

La estrategia usa una opening book pequena alrededor del centro.

No intenta ser una biblioteca teorica profunda; solo busca:

- asegurar flexibilidad inicial
- evitar aperturas demasiado pasivas
- partir desde una estructura que pueda expandirse rapido

La apertura es diferente segun el color, para respetar el eje que cada jugador quiere conectar:

- Negro prioriza continuidad vertical
- Blanco prioriza continuidad horizontal

## 2. Corredores criticos con Dijkstra

La pieza mas importante del evaluador es el calculo de **corredores criticos**.

Para cada jugador se computa:

- distancia desde un borde objetivo
- distancia desde el borde opuesto
- un conjunto de casillas vacias que aparecen en rutas cortas o casi cortas

Esto produce un contexto estructural con:

- `best`: la mejor distancia de conexion actual
- `critical`: un score por casilla critica
- `top`: las casillas mas relevantes para construir o cortar
- `mass`: la “masa” total de rutas cortas visibles

Esta informacion se usa tanto para atacar como para defender.

## 3. Seleccion de candidatas en `classic`

En vez de considerar 121 jugadas con el mismo peso, la estrategia construye una bolsa de candidatas anchas.

Las candidatas se forman combinando:

- casillas criticas propias
- casillas criticas del rival
- jugadas de contacto entre grupos
- jugadas que aumentan la distancia rival
- **pure cuts**, es decir, jugadas buenas para cortar aunque no conecten de inmediato con nuestra propia cadena
- vecinos de la ultima jugada, cuando tiene sentido tactico

Esta parte fue importante porque versiones anteriores solo valoraban bien jugadas “dual-purpose”, y eso hacia que a veces la estrategia persiguiera sombras del corredor rival en vez de cortarlo.

## 4. Evaluacion adversarial previa

Antes del Monte Carlo, cada candidata se evalua con un filtro estructural y tactico.

Para cada jugada candidata se mide:

- cuanto baja nuestra distancia
- cuanto sube la distancia rival
- cuanto reduce la masa de rutas cortas del rival
- cuanto fortalece nuestra propia estructura
- que tan buena queda despues de una posible mejor replica rival

Ese ultimo punto es importante: la estrategia incorpora una **senal explicita de respuesta del oponente**. Es una verificacion corta de 1-ply para evitar jugadas que se ven bien localmente pero colapsan en el siguiente turno.

En el checkpoint final, esta parte quedo asimetrica por color:

- como **Blanco**, la raiz hace primero un filtro barato y luego una evaluacion cara solo sobre un shortlist seguro
- como **Negro**, se mantuvo la evaluacion cara tradicional sobre todas las candidatas relevantes

Esa decision fue intencional: el `two-stage scoring` ayudo mas a Blanco que a Negro en las pruebas, asi que se dejo activo solo donde aportaba valor neto.

## 5. Solver exacto de finales

Cuando el tablero ya esta bastante cerrado, los rollouts cortos dejan de ser la mejor herramienta.

Por eso la version final activa un **solver exacto de endgame** en `classic` cuando quedan pocas casillas vacias:

- usa una busqueda tipo `negamax` con poda alpha-beta
- guarda resultados en una transposition table
- ordena jugadas priorizando wins inmediatas, blocks inmediatos y jugadas estructuralmente fuertes
- solo se activa tarde en la partida, para no gastar presupuesto donde todavia conviene explorar en la raiz

Esta capa fue importante porque varias partidas cerradas, sobre todo jugando como Blanco, se estaban decidiendo por ruido de rollout y no por calculo exacto.

## 6. Ajustes especificos para Blanco

La mayor debilidad observada durante el desarrollo fue la asimetria por color.

Como Blanco, varias versiones tempranas tendian a:

- abrir varios planes horizontales a la vez
- seguir forma local bonita sin cortar el corredor correcto del rival
- o defender demasiado tarde corredores verticales estrechos

Para corregir eso, la version final agrega cuatro ideas puntuales en `classic`:

- deteccion explicita de **corredores estrechos** del rival cuando Negro empieza a consolidar una cadena peligrosa
- penalizacion por **sobrecomprometerse** en una banda horizontal demasiado angosta si la jugada no extiende span ni fusiona grupos
- una bonificacion de **conversion** muy tardia y muy controlada, que solo entra cuando Blanco ya va claramente por delante y el rival no esta creando una amenaza real
- un `two-stage scoring` en la raiz: primero filtra candidatas con señales baratas y luego reserva la evaluacion cara para el shortlist

La intencion no fue volver a Blanco agresivo por defecto, sino hacerlo menos ingenuo en medio juego y menos lento para cerrar una ventaja ya ganada.

## 7. Flat root Monte Carlo en `classic`

La decision final en `classic` no la toma solo la heuristica.

Una vez armada la lista de candidatas, la estrategia hace un **flat root search**:

- elige entre aproximadamente 18 y 24 jugadas prometedoras
- para cada una ejecuta simulaciones cortas desde la raiz
- usa casi todo el presupuesto de tiempo restante
- mezcla resultado de rollout + valor heuristico del estado

No es un MCTS completo profundo sobre todo el arbol. Es un root search ancho, deliberadamente simple, porque:

- el branching factor de Hex 11x11 es muy grande
- las restricciones del torneo castigan mucho una busqueda demasiado ambiciosa
- una buena exploracion en la raiz da mejor retorno que expandir mal niveles profundos

## 8. Rollouts sesgados por conectividad

Los rollouts en `classic` no son aleatorios puros.

Las jugadas durante la simulacion se sesgan con señales de conectividad:

- adyacencias propias y rivales
- fusion de grupos
- patrones de puente
- valor de corredor
- cercania a bordes objetivo
- relacion con la ultima jugada

Ademas, los rollouts son deliberadamente cortos. Si no se resuelve la partida rapido, la estrategia cae a una evaluacion heuristica del estado parcial o, si el tablero ya esta suficientemente cerrado, al solver exacto del final. Esto reduce costo y permite usar mejor los 15 segundos en la raiz.

## 9. Modo `dark`

En `dark` el enfoque es distinto.

Aqui la estrategia prioriza:

- construir una columna o fila muy estable y dificil de detectar completamente
- evitar colisiones innecesarias
- registrar las colisiones en `on_move_result`
- explotar el hecho de que el rival puede chocar repetidamente contra nuestras piedras ocultas

La politica de `dark` es mucho mas ligera que la de `classic`, y eso fue intencional.

En esta variante el mayor retorno no vino de una busqueda mas compleja, sino de:

- jugar seguro
- ocupar corredores centrales utiles
- consolidar cadenas invisibles
- evitar repetir errores ya detectados

## Manejo de fog of war

La estrategia modela la incertidumbre de manera simple pero util.

### Riesgo de colision

Se estima una densidad de piedras rivales ocultas usando:

- cuantos turnos del rival debieron haber ocurrido
- cuantas piedras rivales visibles conocemos ya

Luego esta densidad se ajusta por:

- cercania a rutas plausibles del rival
- presencia local de piedras rivales
- aislamiento respecto a nuestras cadenas

### Memoria de colisiones

Cada vez que el motor reporta una colision por medio de `on_move_result(move, success)`, la casilla queda marcada. Esa informacion se reutiliza para no insistir en jugadas ya descubiertas como ocupadas.

### Idea tactica en dark

En la practica, la estrategia intenta construir cadenas robustas que el rival no ve completas. Eso hace que, en varias partidas, el oponente desperdicie turnos chocando contra piedras nuestras ya colocadas.

## Decisiones de diseno

## Por que no usar RL o ML

No se uso aprendizaje automatico ni aprendizaje por refuerzo porque el reglamento lo prohibe explicitamente. La estrategia se mantuvo completamente dentro de enfoques permitidos:

- heuristicas
- busqueda
- simulaciones
- evaluacion estructural

## Por que no MCTS puro

Se considero hacer un MCTS mas clasico, pero se descarto como enfoque unico porque:

- en `classic` hacia falta mucho control sobre el conjunto inicial de jugadas
- en `dark` el modelo de informacion imperfecta complicaba demasiado una version mas formal
- una implementacion completa era mas costosa y fragil bajo 15 segundos por jugada

El compromiso final fue:

- `classic`: root search ancho con rollouts sesgados
- `dark`: heuristica solida, barata y oportunista

## Por que separar tanto `classic` y `dark`

Una leccion importante de las pruebas fue que ambas variantes premian cosas distintas.

- en `classic`, el problema principal es detectar bien corredores rivales y responder con cortes utiles
- en `dark`, el problema principal es informacion imperfecta y manejo de colisiones

Por eso no convenia forzar exactamente el mismo algoritmo en ambas.

## Alternativas consideradas

Durante el desarrollo se probaron varias familias:

- heuristicas puras
- heuristicas + busqueda tactica corta
- heuristicas + caminos criticos
- heuristicas + desempate Monte Carlo
- root search ancho con rollouts sesgados
- una variante mas cercana a `UCT` / bandit en la raiz

Las versiones mas tempranas jugaban razonablemente contra `Random`, pero quedaban cortas contra `MCTS_Tier_2` porque:

- defendian demasiado localmente
- o construian “forma bonita” sin cortar el corredor correcto

Tambien se probo una raiz mas parecida a `UCT`, pero no quedo como version final porque:

- exploraba mas de lo que convenia bajo 15 segundos
- empeoraba la disciplina estructural de Blanco
- en las mini-series directas contra `MCTS_Tier_2` rindio peor que el root search mas sesgado por heuristica

Tambien se probo aplicar el `two-stage scoring` a ambos colores. Esa variante mejoro a Blanco, pero empeoro a Negro. Por eso el compromiso final fue dejar:

- `two-stage` solo para **Blanco**
- evaluacion completa anterior para **Negro**

Las mejoras mas importantes fueron:

- dejar de parchar solo pesos y pasar a una busqueda de raiz mas ancha
- agregar solver exacto para finales cerrados
- introducir defensa especifica de corredores peligrosos cuando Blanco juega
- usar `two-stage scoring` solo en Blanco para ahorrar presupuesto donde mas ayudaba

## Fortalezas observadas

- domina a `Random`
- ya es competitiva de verdad contra `MCTS_Tier_2`
- como Negro, muestra una columna vertebral muy fuerte
- los finales cerrados en `classic` son mucho mas estables por el solver exacto
- Blanco ya reconoce mejor cuando debe cortar un corredor peligroso en vez de solo seguir expandiendo
- Blanco ahora aprovecha mejor el presupuesto de raiz gracias al filtro barato + shortlist caro
- en `dark`, provoca muchas colisiones del rival
- usa casi todo el presupuesto de tiempo en `classic`, pero sin forfeits

## Limitaciones observadas

- sigue siendo mejor como Negro que como Blanco
- en `classic`, algunas derrotas todavia vienen de abrir varios planes horizontales a la vez cuando juega Blanco
- contra rivales muy pasivos, todavia puede convertir ventajas de forma mas lenta de lo ideal
- no modela formalmente conjuntos de informacion en `dark`
- el root search aun depende de una buena generacion inicial de candidatas
- la mejora mas reciente de `two-stage` esta mejor validada en Blanco que en Negro, por eso no se generalizo a ambos colores

La asimetria por color fue una conclusion importante de las pruebas:

- rendimiento total como Negro: mucho mejor
- rendimiento total como Blanco: mas irregular

## Resultados de pruebas locales

Resultados confirmados en este workspace:

- `classic` vs `Random`: 1/1 victoria en humo rapido
- `dark` vs `Random`: 1/1 victoria en humo rapido
- `classic` vs `MCTS_Tier_2`, 10 partidas: **4 victorias, 6 derrotas**
- `dark` vs `MCTS_Tier_2`, 10 partidas: **7 victorias, 3 derrotas**

En una validacion mas reciente del checkpoint final de `classic`, ya con:

- solver exacto de final
- parche de medio juego para Blanco
- `two-stage scoring` solo para Blanco

se obtuvo una mini-serie de **6 partidas** contra `MCTS_Tier_2` con resultado:

- `classic` vs `MCTS_Tier_2`, 6 partidas: **4 victorias, 2 derrotas**
- por color en esa serie: **2-1 como Negro** y **2-1 como Blanco**

Si se combinan esas dos series directas contra `MCTS_Tier_2`, el resultado fue:

- total directo `classic + dark`: **11 victorias, 9 derrotas**

Ademas, por color:

- como **Negro**: **8 victorias, 2 derrotas**
- como **Blanco**: **3 victorias, 7 derrotas**

Esto no garantiza automaticamente la nota del torneo, porque la calificacion final depende de los standings completos del round-robin y no de un head-to-head aislado. Aun asi, es una señal fuerte de que la estrategia ya puede competir seriamente con `MCTS_Tier_2`, especialmente gracias al rendimiento en `dark`.

## Trabajo futuro

Si se siguiera iterando, las mejoras mas prometedoras serian:

- reforzar especificamente el juego como Blanco
- mejorar la apertura horizontal de Blanco para no abrir varios frentes a la vez
- hacer la conversion de ventajas mas rapida sin perder robustez contra rivales fuertes
- introducir una estimacion mas rica de multiplicidad de rutas rivales
- afinar la seleccion de cortes puros en `classic`
- explorar una version mas fuerte del root search con mejores rollouts, pero sin volver a sobreexplorar como en las pruebas tipo UCT
- validar con mas muestra si el `two-stage scoring` podria extenderse a Negro sin romper su rendimiento

## Restricciones respetadas

- unico archivo evaluado: `estudiantes/MALIK_RUBEN/strategy.py`
- uso exclusivo de biblioteca estandar y utilidades del torneo
- compatibilidad con `classic` y `dark`
- uso de `on_move_result` para registrar colisiones
- sin ML ni RL
- nombre unico: `FogBridge_MALIK_RUBEN`
