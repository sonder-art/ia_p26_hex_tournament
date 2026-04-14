# Estrategia de nuestro equipo: julian_tania con estrategia HexMaster

**HexMaster** es una estrategia para el juego Hex que usa **Monte Carlo Tree Search (MCTS)**, un algoritmo que simula miles de partidas aleatorias para encontrar el mejor movimiento.

La estrategia está altamente optimizada para ser **rápida y eficiente**, permitiendo explorar más opciones en el tiempo disponible.

---

### **Fase 1: Verificación Inicial**

Cuando es tu turno, primero se comprueba:

1. ¿Hay celdas vacías disponibles?
   - Si no hay → Devolver movimiento por defecto (0,0)
   - Si hay solo 1 → Colocar ahí directamente

2. ¿Es el comienzo del juego?
   - Si el tablero tiene 0-1 movimientos → Jugar en el **centro o cerca del centro**
   - Esto ahorra tiempo de cálculo en la apertura

### **Fase 2: Preparación (Dark Mode)**

En modo "dark" (información incompleta):

- No ves todas las piedras del oponente
- La estrategia **adivinador dónde están** usando probabilidades
- Calcula cuántas piedras debería tener el oponente (aproximadamente las mismas que tú)
- Coloca piedras ficticias del oponente aleatoriamente en celdas vacías
- Esto simula un tablero completo para analizar mejor

### **Fase 3: Búsqueda MCTS (Lo Principal)**

Si hay tiempo suficiente (> 0.05 segundos), se inicia la búsqueda MCTS:

```
MIENTRAS hay tiempo disponible:
    1. SELECCIÓN: Navegar por el árbol de posibilidades (jugadas probadas)
    2. EXPANSIÓN: Agregar una nueva jugada posible
    3. ROLLOUT: Simular el resto del juego aleatoriamente
    4. BACKPROPAGATION: Actualizar estadísticas del árbol
```

#### **Componente 1: SELECCIÓN**
- Parte de la raíz del árbol (posición actual)
- Elige hijos del nodo actual basándose en:
  - **Tasa de victoria**: Porcentaje de simulaciones ganadas
  - **Exploración**: Favorecer movimientos poco estudiados
  - **Fórmula UCT**: `(victorias/intentos) + c × √(ln(total)/intentos)`
- Se repite hasta llegar a un nodo sin explorar completamente

#### **Componente 2: EXPANSIÓN**
- Agrega un nuevo movimiento al árbol
- **Si hay ≤8 movimientos candidatos**:
  - Usa análisis inteligente (Dijkstra)
  - Evalúa cada movimiento calculando: distancia del oponente - tu distancia
  - Elige el que maximice la diferencia
- **Si hay >8 movimientos**:
  - Elige uno al azar para no gastar tiempo

#### **Componente 3: ROLLOUT (Simulación Rápida)**
- Juega el resto de la partida automáticamente hasta terminar
- **Estrategia**:
  - Los próximos ~6 movimientos los alterna entre jugadores
  - Los movimientos restantes se colocan aleatoriamente
- **Detección rápida de victoria**:
  - Usa BFS (búsqueda en amplitud) para detectar si hay camino ganador
  - Optimizado con sellos para evitar recalcular

#### **Componente 4: BACKPROPAGATION**
- Sube por el árbol desde el nodo expandido hasta la raíz
- Actualiza estadísticas: `visitas += 1` y `victorias += resultado`
- El resultado se invierte según quién ganó (perspectiva del jugador)

### **Fase 4: Selección del Mejor Movimiento**

Después de hacer todas las simulaciones posibles:

1. Mira todos los movimientos analizados
2. Elige el que tiene **más simulaciones exitosas**
3. Si algo falla (en dark mode puede haber inconsistencias), elige al azar

### **Fase 5: Fallback (Sin Tiempo)**

Si el presupuesto de tiempo se agota (< 0.05 segundos):

- **No hay tiempo para MCTS** → Usar heurística rápida Dijkstra
- Muestrea hasta 8 movimientos aleatorios
- Evalúa cuál mejora más tu posición
- Devuelve el mejor encontrado

---

## Optimizaciones Clave

### 1. **Precálculo de Estructuras**
- Los vecinos de cada celda se calculan **una sola vez al cargar el módulo**
- Los bordes objetivo también se precalculan
- Evita recalcular estas estructuras miles de veces

### 2. **Arrays Planos en lugar de Objetos**
- Usa arrays de números en lugar de clases personalizadas
- Ejemplo: `parent[i]` en lugar de `node.parent`
- **Razón**: Mucho más rápido en memoria y acceso

### 3. **Reutilización de Memoria**
- El BFS reutiliza los mismos arrays para cada simulación
- Usa "sellos de generación" (números secuenciales) para limpiar sin reasignar
- **Efecto**: Reduce tiempo de asignación de memoria

### 4. **Sets para Celdas Vacías**
- Mantiene las celdas vacías en un `set` (O(1) para agregar/quitar)
- No reconstruye listas enteras con cada movimiento
- **Razón**: Acceso ultrarrápido

### 5. **Análisis Inteligente en Expansión**
- Si hay pocos candidatos (≤8), analiza cada uno con Dijkstra
- Si hay muchos, elige al azar
- **Balance**: Calidad vs velocidad

---

## Flujo de Decisión (Resumen Visual)

```
Inicio del turno
    ↓
¿Hay celdas vacías?
    ↓ No → Devolver (0,0)
    ↓ Sí
¿Solo 1 celda vacía?
    ↓ Sí → Colocar ahí
    ↓ No
¿Es apertura (0-1 movimientos en tablero)?
    ↓ Sí → Jugar en centro
    ↓ No
¿Es dark mode?
    ↓ Sí → Adivinar posición de piedras oponentes
    ↓ No → Usar tablero real
¿Hay tiempo suficiente (>0.05s)?
    ↓ Sí → Ejecutar MCTS
    ↓ No → Usar heurística rápida Dijkstra
¿MCTS encontró movimiento válido?
    ↓ Sí → Usar ese movimiento
    ↓ No → Elegir al azar entre celdas vacías
```

---

## Ejemplo de Simulación MCTS

Imaginemos que es tu turno en una posición con 10 celdas vacías:

### Ciclo 1:
1. **Selección**: Bajo por el árbol existente (si es el primer movimiento, es vacío)
2. **Expansión**: Agrego la celda A (con análisis Dijkstra si son pocas opciones)
3. **Rollout**: Simulo que yo juego en A, luego movimientos aleatorios hasta el final
4. **Resultado**: Victoria (1.0) o Derrota (0.0)
5. **Backpropagation**: Actualizo estadísticas de A

### Ciclo 2:
1. **Selección**: Vuelvo a la raíz, ahora A tiene 1 victoria de 1 intento
2. **Expansión**: Pruebo celda B
3. **Rollout**: Simulo el juego completo
4. **Resultado**: Derrota (0.0)
5. **Backpropagation**: Actualizo estadísticas de B

### Ciclos 3-1000:
- Se repite con diferentes celdas
- A acumula mejor puntuación: 850/1000
- B acumula peor: 300/1000
- C acumula intermedia: 600/1000

### Selección Final:
- Elige **celda A** porque tiene más victorias acumuladas

---

## Parámetros Importantes

| Parámetro | Valor | Significado |
|-----------|-------|-------------|
| Tamaño tablero | 11×11 | Hex estándar |
| Presupuesto temporal | 92% del tiempo límite | Evita timeout |
| Valor de exploración (c) | 1.8 | Balance entre explotación y exploración |
| Umbral Dijkstra | ≤8 movimientos | Cuándo usar análisis inteligente |
| Tiempo mínimo fallback | 0.05s | Cuándo cambiar a heurística rápida |

---

## Por Qué Funciona Este Enfoque

1. **MCTS es flexible**: Funciona en tableros parcialmente ocultos (dark mode)
2. **Simulaciones aleatorias**: Explora espacios de búsqueda enormes eficientemente
3. **Optimizaciones agresivas**: Permite hacer 1000+ simulaciones en tiempo real
4. **Heurística de respaldo**: Si falla algo, siempre hay un plan B rápido
5. **Análisis inteligente**: La heurística Dijkstra guía mejor las expansiones

---

## Cuando se inicia un juego:

- Se precalculan vecinos y bordes (una sola vez)
- Se inicializa el presupuesto temporal
- Se detecta modo de juego (classic/dark)
- En cada turno, se ejecuta el flujo de decisión
- Se elige el mejor movimiento disponible
- Se actualiza el contador de movimientos