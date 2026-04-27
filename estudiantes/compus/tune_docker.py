from __future__ import annotations

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SPACE: dict[str, list[float | int]] = {
    "COMPUS_C_PUCT": [0.9, 1.1, 1.25, 1.4, 1.65],
    "COMPUS_ROLLOUT_DEPTH": [6, 8, 10, 12],
    "COMPUS_ROLLOUT_EPSILON": [0.08, 0.14, 0.18, 0.24, 0.30],
    "COMPUS_RISK_LOW": [0.20, 0.30, 0.35, 0.45],
    "COMPUS_RISK_MID": [0.35, 0.45, 0.55, 0.65],
    "COMPUS_RISK_HIGH": [0.50, 0.65, 0.80, 0.95],
    "COMPUS_BUDGET_OPEN": [0.30, 0.36, 0.40, 0.46],
    "COMPUS_BUDGET_MID": [0.44, 0.50, 0.56, 0.62],
    "COMPUS_BUDGET_LATE": [0.58, 0.64, 0.68, 0.74],
    "COMPUS_BUDGET_END": [0.72, 0.78, 0.82, 0.88],
    "COMPUS_ROOT_LIMIT_SCALE": [1.00, 1.15, 1.25, 1.35, 1.50],
    "COMPUS_TREE_LIMIT_SCALE": [1.00, 1.10, 1.20, 1.30, 1.45],
    "COMPUS_ROLLOUT_DEPTH_BONUS": [0, 1, 2, 3, 4],
}

FALLBACK_DEFAULTS: dict[str, float | int] = {
    "COMPUS_C_PUCT": 1.20,
    "COMPUS_ROLLOUT_DEPTH": 9,
    "COMPUS_ROLLOUT_EPSILON": 0.14,
    "COMPUS_RISK_LOW": 0.35,
    "COMPUS_RISK_MID": 0.50,
    "COMPUS_RISK_HIGH": 0.72,
    "COMPUS_BUDGET_OPEN": 0.40,
    "COMPUS_BUDGET_MID": 0.54,
    "COMPUS_BUDGET_LATE": 0.68,
    "COMPUS_BUDGET_END": 0.82,
    "COMPUS_ROOT_LIMIT_SCALE": 1.25,
    "COMPUS_TREE_LIMIT_SCALE": 1.20,
    "COMPUS_ROLLOUT_DEPTH_BONUS": 2,
}

INT_KEYS = {"COMPUS_ROLLOUT_DEPTH", "COMPUS_ROLLOUT_DEPTH_BONUS"}


@dataclass
class CandidateResult:
    index: int
    params: dict[str, float | int]
    wins: int
    losses: int
    forfeits: int
    games: int
    win_rate: float
    score: float
    classic_win_rate: float
    dark_win_rate: float
    elapsed_sec: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_strategy_defaults(strategy_path: Path) -> dict[str, float | int]:
    text = strategy_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(strategy_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DEFAULT_TUNING":
                parsed = ast.literal_eval(node.value)
                defaults = dict(FALLBACK_DEFAULTS)
                for key in FALLBACK_DEFAULTS:
                    if key in parsed:
                        defaults[key] = parsed[key]
                return defaults
    return dict(FALLBACK_DEFAULTS)


def _build_candidates(
    trials: int,
    seed: int,
    defaults: dict[str, float | int],
) -> list[dict[str, float | int]]:
    rng = random.Random(seed)
    candidates: list[dict[str, float | int]] = [dict(defaults)]
    seen = {tuple(sorted(defaults.items()))}

    while len(candidates) < max(1, trials):
        cand: dict[str, float | int] = {}
        for key, values in SPACE.items():
            cand[key] = rng.choice(values)
        if cand["COMPUS_RISK_LOW"] > cand["COMPUS_RISK_MID"]:
            cand["COMPUS_RISK_MID"] = cand["COMPUS_RISK_LOW"]
        if cand["COMPUS_RISK_MID"] > cand["COMPUS_RISK_HIGH"]:
            cand["COMPUS_RISK_HIGH"] = cand["COMPUS_RISK_MID"]
        if cand["COMPUS_BUDGET_OPEN"] > cand["COMPUS_BUDGET_MID"]:
            cand["COMPUS_BUDGET_MID"] = cand["COMPUS_BUDGET_OPEN"]
        if cand["COMPUS_BUDGET_MID"] > cand["COMPUS_BUDGET_LATE"]:
            cand["COMPUS_BUDGET_LATE"] = cand["COMPUS_BUDGET_MID"]
        if cand["COMPUS_BUDGET_LATE"] > cand["COMPUS_BUDGET_END"]:
            cand["COMPUS_BUDGET_END"] = cand["COMPUS_BUDGET_LATE"]

        key_tuple = tuple(sorted(cand.items()))
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        candidates.append(cand)

    return candidates[:trials]


def _parse_variants(raw: str) -> list[str]:
    variants = []
    for part in raw.split(","):
        item = part.strip().lower()
        if item not in {"classic", "dark"}:
            raise ValueError(f"Variante invalida: {item}")
        variants.append(item)
    if not variants:
        raise ValueError("Debes incluir al menos una variante")
    return variants


def _run_single(
    *,
    repo_root: Path,
    team: str,
    strategy: str,
    opponent: str,
    variant: str,
    num_games: int,
    move_timeout: float,
    seed: int,
    params: dict[str, float | int],
    as_black: bool,
    run_tag: str,
) -> tuple[int, int, int]:
    host_json = repo_root / "estudiantes" / team / "results" / f"{run_tag}.json"
    host_json.parent.mkdir(parents=True, exist_ok=True)
    if host_json.exists():
        host_json.unlink()

    container_json = f"/app/estudiantes/{team}/results/{run_tag}.json"
    black = strategy if as_black else opponent
    white = opponent if as_black else strategy

    cmd = ["docker", "compose", "run", "--rm", "-T"]
    for key, value in params.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(
        [
            "experiment",
            "python",
            "experiment.py",
            "--team",
            team,
            "--black",
            black,
            "--white",
            white,
            "--variant",
            variant,
            "--num-games",
            str(num_games),
            "--move-timeout",
            str(move_timeout),
            "--seed",
            str(seed),
            "--json",
            container_json,
        ]
    )

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        summary = stderr if stderr else stdout
        raise RuntimeError(
            f"Fallo docker en {run_tag}. "
            f"Comando: {' '.join(cmd)} "
            f"Salida: {summary[:800]}"
        )

    if not host_json.exists():
        raise RuntimeError(f"No se genero JSON de salida: {host_json}")

    data = json.loads(host_json.read_text(encoding="utf-8"))
    wins = 0
    losses = 0
    forfeits = 0
    for game in data.get("games", []):
        won = game.get("winner") == strategy
        if won:
            wins += 1
        else:
            losses += 1
        if game.get("forfeit") and not won:
            forfeits += 1

    return wins, losses, forfeits


def _evaluate_candidate(
    *,
    index: int,
    repo_root: Path,
    team: str,
    strategy: str,
    opponent: str,
    variants: list[str],
    num_games: int,
    move_timeout: float,
    base_seed: int,
    both_colors: bool,
    params: dict[str, float | int],
) -> CandidateResult:
    started = time.monotonic()
    wins = 0
    losses = 0
    forfeits = 0
    classic_wins = 0
    classic_games = 0
    dark_wins = 0
    dark_games = 0

    colors = [True, False] if both_colors else [True]

    for v_idx, variant in enumerate(variants):
        for c_idx, as_black in enumerate(colors):
            run_seed = base_seed + (index * 1000) + (v_idx * 100) + (c_idx * 11)
            tag = f"tune_{index:03d}_{variant}_{'B' if as_black else 'W'}"
            w, l, f = _run_single(
                repo_root=repo_root,
                team=team,
                strategy=strategy,
                opponent=opponent,
                variant=variant,
                num_games=num_games,
                move_timeout=move_timeout,
                seed=run_seed,
                params=params,
                as_black=as_black,
                run_tag=tag,
            )
            wins += w
            losses += l
            forfeits += f
            if variant == "classic":
                classic_wins += w
                classic_games += w + l
            else:
                dark_wins += w
                dark_games += w + l

    games = wins + losses
    win_rate = wins / games if games else 0.0
    forfeit_rate = forfeits / games if games else 0.0
    score = win_rate - 0.25 * forfeit_rate

    classic_wr = classic_wins / classic_games if classic_games else 0.0
    dark_wr = dark_wins / dark_games if dark_games else 0.0

    return CandidateResult(
        index=index,
        params=params,
        wins=wins,
        losses=losses,
        forfeits=forfeits,
        games=games,
        win_rate=win_rate,
        score=score,
        classic_win_rate=classic_wr,
        dark_win_rate=dark_wr,
        elapsed_sec=time.monotonic() - started,
    )


def _format_value(key: str, value: float | int) -> str:
    if key in INT_KEYS:
        return str(int(value))
    num = float(value)
    if num.is_integer():
        return f"{num:.1f}"
    return f"{num:.6g}"


def _apply_best_to_strategy(
    strategy_path: Path,
    best_params: dict[str, float | int],
) -> None:
    text = strategy_path.read_text(encoding="utf-8")

    for key, value in best_params.items():
        pattern = rf'("{re.escape(key)}"\s*:\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        replacement = rf'\g<1>{_format_value(key, value)}'
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise RuntimeError(
                f"No pude actualizar la clave {key} en {strategy_path}. "
                "Verifica el bloque DEFAULT_TUNING."
            )

    strategy_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tuning automatico usando Docker experiment")
    parser.add_argument("--team", type=str, default="compus")
    parser.add_argument("--strategy", type=str, default="MiEstrategia_mi_equipo")
    parser.add_argument("--opponent", type=str, default="MCTS_Tier_3")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--num-games", type=int, default=2)
    parser.add_argument("--variants", type=str, default="classic,dark")
    parser.add_argument("--both-colors", action="store_true")
    parser.add_argument("--move-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--strategy-file", type=str, default=None)
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials debe ser >= 1")
    if args.num_games < 1:
        raise ValueError("--num-games debe ser >= 1")

    repo_root = _repo_root()
    variants = _parse_variants(args.variants)
    if args.strategy_file:
        strategy_path = (repo_root / args.strategy_file).resolve()
    else:
        strategy_path = repo_root / "estudiantes" / args.team / "strategy.py"
    defaults = _load_strategy_defaults(strategy_path)
    candidates = _build_candidates(args.trials, args.seed, defaults)

    print(f"Repo: {repo_root}")
    print(f"Estrategia: {args.strategy}")
    print(f"Oponente: {args.opponent}")
    print(f"Trials: {len(candidates)} | Variants: {variants} | both_colors={args.both_colors}")
    print(f"num_games={args.num_games} | move_timeout={args.move_timeout}\n")

    results: list[CandidateResult] = []
    for idx, params in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] params={params}")
        res = _evaluate_candidate(
            index=idx,
            repo_root=repo_root,
            team=args.team,
            strategy=args.strategy,
            opponent=args.opponent,
            variants=variants,
            num_games=args.num_games,
            move_timeout=args.move_timeout,
            base_seed=args.seed,
            both_colors=args.both_colors,
            params=params,
        )
        results.append(res)
        print(
            f"  -> score={res.score:.3f} win_rate={res.win_rate:.3f} "
            f"classic={res.classic_win_rate:.3f} dark={res.dark_win_rate:.3f} "
            f"forfeits={res.forfeits}/{res.games} time={res.elapsed_sec:.1f}s\n"
        )

    results.sort(key=lambda r: (r.score, r.win_rate, -r.forfeits), reverse=True)
    top_n = min(args.top, len(results))

    print("=== TOP CANDIDATOS ===")
    for pos in range(top_n):
        r = results[pos]
        print(
            f"{pos + 1}. idx={r.index} score={r.score:.3f} "
            f"wr={r.win_rate:.3f} classic={r.classic_win_rate:.3f} "
            f"dark={r.dark_win_rate:.3f} params={r.params}"
        )

    out_dir = repo_root / "estudiantes" / args.team / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    out_file = out_dir / f"tuning_summary_{args.opponent}_{stamp}.json"
    payload = {
        "team": args.team,
        "strategy": args.strategy,
        "opponent": args.opponent,
        "variants": variants,
        "both_colors": args.both_colors,
        "num_games": args.num_games,
        "move_timeout": args.move_timeout,
        "trials": args.trials,
        "seed": args.seed,
        "space": SPACE,
        "defaults": defaults,
        "results": [asdict(r) for r in results],
        "best": asdict(results[0]) if results else None,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumen guardado en: {out_file}")
    if results:
        print(f"Mejor configuracion: {results[0].params}")
        if args.apply_best:
            _apply_best_to_strategy(strategy_path, results[0].params)
            print(f"Defaults aplicados en: {strategy_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
