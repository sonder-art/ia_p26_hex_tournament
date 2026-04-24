"""
Precomputación del opening book con multiprocessing.

Mejoras sobre versión anterior:
  - Multiprocessing: usa todos los núcleos disponibles (N_CPU - 1)
  - Book deepening: después de cubrir el opening, profundiza en líneas buenas
  - Temperatura ajustable: controla diversidad de aperturas generadas
  - Tiempo asimétrico: jugador fuerte piensa más (mejor calidad de apertura)
  - ETA en tiempo real por partida
  - Guarda tras cada partida por defecto

Uso recomendado overnight:
    python precompute.py --games 5000 --time-per-move 8 --depth 16

Pausa con Ctrl+C — reanuda automáticamente la próxima vez.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from collections import defaultdict
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap de la interfaz Strategy
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass
import types as _types


@dataclass(frozen=True)
class GameConfig:
    board_size: int
    variant: str
    initial_board: tuple
    player: int
    opponent: int
    time_limit: float


class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    def begin_game(self, config: GameConfig) -> None:
        pass

    @abstractmethod
    def play(self, board, last_move): ...
    def on_move_result(self, move, success: bool) -> None:
        pass

    def end_game(self, board, winner: int, your_player: int) -> None:
        pass


_HERE = os.path.dirname(os.path.abspath(__file__))
_STRAT_PATH = os.path.join(_HERE, "strategy.py")
_BOOK_PATH = os.path.join(_HERE, "opening_book.json")


def _inject_fake_strategy_module() -> None:
    """Inyecta la interfaz Strategy en sys.modules antes de importar strategy.py."""
    fake = _types.ModuleType("strategy")
    fake.Strategy = Strategy
    fake.GameConfig = GameConfig
    sys.modules["strategy"] = fake


def _load_strategy_class():
    """Carga e instancia la clase Strategy desde strategy.py."""
    import importlib.util

    _inject_fake_strategy_module()
    spec = importlib.util.spec_from_file_location("_hex_strat", _STRAT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            return obj
    raise RuntimeError("No Strategy class found in strategy.py")


# Cargar en el proceso principal (para reportar nombre)
_StrategyClass = _load_strategy_class()
print(f"Estrategia cargada: {_StrategyClass().name}")

# ---------------------------------------------------------------------------
# Motor de juego mínimo
# ---------------------------------------------------------------------------

_NBS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


def _neighbors(r: int, c: int, size: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc) for dr, dc in _NBS if 0 <= r + dr < size and 0 <= c + dc < size
    ]


def _board_key(board: list[list[int]], size: int) -> str:
    return ";".join(
        sorted(
            f"{board[r][c]}:{r},{c}"
            for r in range(size)
            for c in range(size)
            if board[r][c] != 0
        )
    )


def _board_from_key(key: str, size: int) -> list[list[int]]:
    board = [[0] * size for _ in range(size)]
    if key:
        for part in key.split(";"):
            if not part:
                continue
            player_s, pos = part.split(":")
            r, c = map(int, pos.split(","))
            board[r][c] = int(player_s)
    return board


def _count_stones(board: list[list[int]], size: int) -> int:
    return sum(board[r][c] != 0 for r in range(size) for c in range(size))


# ---------------------------------------------------------------------------
# Worker para multiprocessing
# ---------------------------------------------------------------------------

# Variable global del worker: inicializada en _worker_init
_W_SC = None  # StrategyClass en cada worker


def _worker_init() -> None:
    """Inicializa la estrategia en cada proceso worker."""
    global _W_SC
    _W_SC = _load_strategy_class()


def _sample_with_temperature(children, temperature: float):
    """Muestrea movida proporcional a N^(1/T). T=1→proporcional, T→0→greedy."""
    if not children:
        return None
    visits = [max(ch.N, 1) for ch in children]
    if temperature < 0.01:
        return children[max(range(len(children)), key=lambda i: visits[i])].move
    weights = [v ** (1.0 / temperature) for v in visits]
    total = sum(weights)
    rv = random.random() * total
    cum = 0.0
    for ch, w in zip(children, weights):
        cum += w
        if cum >= rv:
            return ch.move
    return children[-1].move


def _worker_game(args) -> list[tuple[str, str, str]]:
    """
    Ejecuta una partida en el worker y retorna lista de (variant, key, move_str).
    Función a nivel de módulo: requerido para multiprocessing.
    """
    global _W_SC
    (size, variant, time_s, time_w, depth, temperature, seed, start_key) = args
    random.seed(seed)

    SC = _W_SC
    GC = GameConfig

    # Construir tablero inicial (puede ser una posición del libro)
    if start_key:
        board = _board_from_key(start_key, size)
    else:
        board = [[0] * size for _ in range(size)]

    empty_t = tuple(tuple(r) for r in board)
    stones_at_start = _count_stones(board, size)

    players = {}
    times = {1: time_s, 2: time_w}
    for p in (1, 2):
        s = SC()
        s.begin_game(GC(size, variant, empty_t, p, 3 - p, times[p]))
        players[p] = s

    # Determinar de quién es el turno
    ones = sum(board[r][c] == 1 for r in range(size) for c in range(size))
    twos = sum(board[r][c] == 2 for r in range(size) for c in range(size))
    current = 1 if ones <= twos else 2
    last_move: Optional[tuple[int, int]] = None
    results: list[tuple[str, str, str]] = []
    moves_played = 0

    while moves_played < depth:
        board_t = tuple(tuple(r) for r in board)
        strat = players[current]

        if variant == "dark":
            view = tuple(
                tuple(board[r][c] if board[r][c] == current else 0 for c in range(size))
                for r in range(size)
            )
            move = strat.play(view, None)
        else:
            move = strat.play(board_t, last_move)

        # Temperatura: diversifica el libro muestreando en lugar de greedy
        if (
            temperature > 0.01
            and hasattr(strat, "root")
            and strat.root is not None
            and strat.root.children
        ):
            sampled = _sample_with_temperature(strat.root.children, temperature)
            if sampled is not None:
                move = sampled

        key = _board_key(board, size)
        mv_str = f"{move[0]},{move[1]}"
        results.append((variant, key, mv_str))

        r, c = move
        if 0 <= r < size and 0 <= c < size and board[r][c] == 0:
            board[r][c] = current
            strat.on_move_result(move, True)
            last_move = move
        else:
            strat.on_move_result(move, False)

        current = 3 - current
        moves_played += 1

    return results


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def load_book(path: str) -> tuple[dict, dict]:
    book_counts: dict = {"classic": defaultdict(dict), "dark": defaultdict(dict)}
    book_best: dict = {"classic": {}, "dark": {}}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for variant in ("classic", "dark"):
                for key, mv_dict in data.get("_counts", {}).get(variant, {}).items():
                    book_counts[variant][key] = mv_dict
                for key, mv in data.get(variant, {}).items():
                    book_best[variant][key] = mv
            n = sum(len(book_best[v]) for v in ("classic", "dark"))
            print(f"Libro cargado: {n} entradas — {path}")
        except Exception as e:
            print(f"No se pudo cargar libro ({e}), empezando desde cero.")
    return book_counts, book_best


def compile_book(book_counts: dict) -> dict:
    book_best: dict = {}
    for variant in ("classic", "dark"):
        book_best[variant] = {}
        for key, mv_dict in book_counts[variant].items():
            if mv_dict:
                best_str = max(mv_dict, key=mv_dict.get)
                r, c = map(int, best_str.split(","))
                book_best[variant][key] = [r, c]
    return book_best


def save_book(path: str, book_counts: dict, book_best: dict) -> None:
    data = {
        "classic": book_best.get("classic", {}),
        "dark": book_best.get("dark", {}),
        "_counts": {v: dict(book_counts[v]) for v in ("classic", "dark")},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _get_deepening_positions(
    book_counts: dict,
    variant: str,
    min_visits: int = 3,
    max_positions: int = 50,
) -> list[str]:
    """
    Retorna posiciones del libro con suficientes visitas para profundizar.
    Book deepening: genera partidas empezando desde posiciones ya conocidas
    para extender la cobertura más allá del depth inicial.
    """
    candidates = []
    for key, mv_dict in book_counts[variant].items():
        total = sum(mv_dict.values())
        if total >= min_visits and key:  # excluye tablero vacío
            candidates.append((total, key))
    # Ordenar por visitas (más visitadas = más prometedoras)
    candidates.sort(reverse=True)
    return [key for _, key in candidates[:max_positions]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precomputación del opening book (multiprocessing, overnight)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  # Overnight recomendado (usa todos los núcleos)
  python precompute.py --games 5000 --time-per-move 8 --depth 16

  # Solo classic, más profundo
  python precompute.py --games 3000 --time-per-move 10 --depth 20 --variants classic

  # Test rápido
  python precompute.py --games 10 --time-per-move 3 --depth 6
""",
    )
    parser.add_argument(
        "--games", type=int, default=5000, help="Partidas por variante (default: 5000)"
    )
    parser.add_argument(
        "--time-per-move",
        type=float,
        default=8.0,
        help="Segundos/movida jugador fuerte (default: 8)",
    )
    parser.add_argument(
        "--time-weak",
        type=float,
        default=None,
        help="Segundos/movida jugador débil (default: time/3)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Movidas a cubrir por partida (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperatura 0=greedy 1=proporcional (default: 1.0)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["classic", "dark"],
        choices=["classic", "dark"],
    )
    parser.add_argument("--board-size", type=int, default=11)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Procesos paralelos (default: N_CPU-1)",
    )
    parser.add_argument(
        "--deepen-every",
        type=int,
        default=200,
        help="Cada N partidas, profundiza posiciones del libro (default: 200)",
    )
    parser.add_argument("--out", type=str, default=_BOOK_PATH)
    args = parser.parse_args()

    time_strong = args.time_per_move
    time_weak = args.time_weak if args.time_weak else max(1.0, time_strong / 3.0)
    n_workers = args.workers if args.workers else max(1, mp.cpu_count() - 1)

    print(
        f"Workers: {n_workers} | Fuerte: {time_strong}s | Débil: {time_weak:.1f}s | "
        f"depth: {args.depth} | T: {args.temperature}"
    )

    book_counts, book_best = load_book(args.out)

    interrupted = False

    def _handle(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrumpido — guardando...", flush=True)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    total_games = 0
    t_global = time.monotonic()

    # Crear pool con inicializador
    pool = mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
    )

    try:
        for variant in args.variants:
            if interrupted:
                break

            secs_game = args.depth * (time_strong + time_weak) / 2
            eta_min = secs_game * args.games / 60 / n_workers
            print(
                f"\n=== {variant.upper()} | {args.games} partidas | "
                f"ETA ≈ {eta_min:.0f} min con {n_workers} workers ===",
                flush=True,
            )

            t_variant = time.monotonic()
            g_done = 0
            rng = random.Random(42 + hash(variant))

            # Generar argumentos en lotes
            batch_size = n_workers * 4

            while g_done < args.games and not interrupted:
                remaining = args.games - g_done
                batch = min(batch_size, remaining)

                # Cada N partidas: incluir posiciones del libro para deepening
                use_deepen = (
                    args.deepen_every > 0
                    and g_done > 0
                    and g_done % args.deepen_every == 0
                )
                deep_positions = []
                if use_deepen:
                    deep_positions = _get_deepening_positions(
                        book_counts, variant, min_visits=3, max_positions=batch // 2
                    )

                job_args = []
                for i in range(batch):
                    seed = rng.randint(0, 2**31)
                    # Alternar entre tablero vacío y posiciones del libro
                    if deep_positions and i % 3 == 2:
                        start_key = rng.choice(deep_positions)
                    else:
                        start_key = ""
                    job_args.append(
                        (
                            args.board_size,
                            variant,
                            time_strong,
                            time_weak,
                            args.depth,
                            args.temperature,
                            seed,
                            start_key,
                        )
                    )

                # Ejecutar lote en paralelo
                t_batch = time.monotonic()
                try:
                    results_batch = pool.map(_worker_game, job_args)
                except Exception as e:
                    print(f"Error en worker: {e}", file=sys.stderr)
                    break

                # Agregar resultados al libro
                for game_results in results_batch:
                    for v, key, mv_str in game_results:
                        entry = book_counts[v][key]
                        entry[mv_str] = entry.get(mv_str, 0) + 1

                g_done += len(results_batch)
                total_games += len(results_batch)

                # Guardar y reportar
                book_best = compile_book(book_counts)
                save_book(args.out, book_counts, book_best)
                n_entries = sum(len(book_best[v2]) for v2 in ("classic", "dark"))

                elapsed_var = time.monotonic() - t_variant
                avg_per = elapsed_var / g_done
                remaining_m = avg_per * (args.games - g_done) / 60
                batch_t = time.monotonic() - t_batch

                print(
                    f"  [{variant}] {g_done:>5}/{args.games} | "
                    f"lote {len(results_batch)} en {batch_t:.1f}s | "
                    f"entradas: {n_entries} | "
                    f"ETA: {remaining_m:.1f} min",
                    flush=True,
                )

                if interrupted:
                    break

    finally:
        pool.terminate()
        pool.join()

    # Guardado final
    book_best = compile_book(book_counts)
    save_book(args.out, book_counts, book_best)
    n_entries = sum(len(book_best[v]) for v in ("classic", "dark"))
    total_elapsed = time.monotonic() - t_global

    print(f"\nListo. {total_games} partidas en {total_elapsed/60:.1f} min.")
    print(f"Entradas en libro: {n_entries}")
    print(f"Archivo: {args.out}")


if __name__ == "__main__":
    main()
