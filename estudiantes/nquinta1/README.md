# NQuinta_Strategy_v1 — Hex Agent

## Descripción

`NQuinta_Strategy_v1` es un agente para el juego de **Hex (11x11, variante clásica)** diseñado para competir en el torneo del curso.

La estrategia implementa un enfoque heurístico enfocado en:
- Control del centro del tablero
- Conectividad progresiva entre lados
- Bloqueo básico del oponente
- Construcción de caminos eficientes

El agente fue evaluado contra un oponente aleatorio (`Random`), logrando un desempeño perfecto.

## Resultados

### vs Random (20 partidas)

- **Victorias:** 20 / 20 (100%)
- **Derrotas:** 0
- **Longitud promedio de partida:** 32 movimientos

Esto demuestra que el agente:
- Mantiene consistencia en decisiones
- Construye caminos ganadores de forma estable
- No depende del azar

## Estrategia

El agente sigue una lógica sencilla pero efectiva:

1. **Prioridad al centro**
   - Intenta ocupar posiciones centrales al inicio
   - Mejora la flexibilidad y conectividad

2. **Expansión de conexiones**
   - Busca extender sus propias piezas
   - Favorece caminos continuos hacia su objetivo

3. **Bloqueo del oponente**
   - Detecta posibles conexiones del rival
   - Interrumpe trayectorias clave

4. **Selección heurística**
   - Evalúa movimientos posibles
   - Elige aquellos que maximizan conexión y control

## Implementación

El agente está implementado en `strategy.py`.

Clase principal: `NQuintaStrategy`

La lógica principal está en el método `play(...)`, donde se evalúan todos los movimientos legales disponibles y se selecciona el que mejor balancea avance propio, contención del rival y cercanía al centro.

## Cómo ejecutar

Ejemplo de experimento:

```bash
python3 experiment.py --black "NQuinta_Strategy_v1" --white "Random" --num-games 20
