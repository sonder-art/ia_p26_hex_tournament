from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tune_docker import SPACE, _apply_best_to_strategy, _load_strategy_defaults


OPPONENT_WEIGHTS: dict[str, float] = {
    "Random": 1.00,
    "MCTS_Tier_1": 1.25,
    "MCTS_Tier_2": 1.55,
    "MCTS_Tier_3": 2.00,
    "MCTS_Tier_4": 2.60,
    "MCTS_Tier_5": 3.30,
}

INT_KEYS = {"COMPUS_ROLLOUT_DEPTH", "COMPUS_ROLLOUT_DEPTH_BONUS"}


@dataclass
class MatchupStats:
    wins: int = 0
    losses: int = 0
    forfeits: int = 0
    games: int = 0
    my_move_count: int = 0
    my_move_time_sum: float = 0.0
    my_move_time_max: float = 0.0

    def merge(self, other: "MatchupStats") -> None:
        self.wins += other.wins
        self.losses += other.losses
        self.forfeits += other.forfeits
        self.games += other.games
        self.my_move_count += other.my_move_count
        self.my_move_time_sum += other.my_move_time_sum
        self.my_move_time_max = max(self.my_move_time_max, other.my_move_time_max)

    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def forfeit_rate(self) -> float:
        return self.forfeits / self.games if self.games else 0.0

    def avg_move_time(self) -> float:
        return self.my_move_time_sum / self.my_move_count if self.my_move_count else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_csv_list(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Lista vacia no permitida")
    return items


def _parse_variants(raw: str) -> list[str]:
    variants = [x.lower() for x in _parse_csv_list(raw)]
    for v in variants:
        if v not in {"classic", "dark"}:
            raise ValueError(f"Variante invalida: {v}")
    return variants


def _normalize_params(params: dict[str, float | int]) -> dict[str, float | int]:
    out = dict(params)
    for key in INT_KEYS:
        if key in out:
            out[key] = int(round(float(out[key])))

    # Monotonic constraints
    if out["COMPUS_RISK_LOW"] > out["COMPUS_RISK_MID"]:
        out["COMPUS_RISK_MID"] = out["COMPUS_RISK_LOW"]
    if out["COMPUS_RISK_MID"] > out["COMPUS_RISK_HIGH"]:
        out["COMPUS_RISK_HIGH"] = out["COMPUS_RISK_MID"]
    if out["COMPUS_BUDGET_OPEN"] > out["COMPUS_BUDGET_MID"]:
        out["COMPUS_BUDGET_MID"] = out["COMPUS_BUDGET_OPEN"]
    if out["COMPUS_BUDGET_MID"] > out["COMPUS_BUDGET_LATE"]:
        out["COMPUS_BUDGET_LATE"] = out["COMPUS_BUDGET_MID"]
    if out["COMPUS_BUDGET_LATE"] > out["COMPUS_BUDGET_END"]:
        out["COMPUS_BUDGET_END"] = out["COMPUS_BUDGET_LATE"]

    return out


def _params_fingerprint(params: dict[str, float | int]) -> str:
    p = _normalize_params(params)
    return json.dumps(p, sort_keys=True, separators=(",", ":"))


def _candidate_random(rng: random.Random, defaults: dict[str, float | int]) -> dict[str, float | int]:
    cand = dict(defaults)
    for key, values in SPACE.items():
        cand[key] = rng.choice(values)
    return _normalize_params(cand)


def _closest_index(values: list[float | int], current: float | int) -> int:
    cur = float(current)
    best_i = 0
    best_d = abs(float(values[0]) - cur)
    for i in range(1, len(values)):
        d = abs(float(values[i]) - cur)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _candidate_mutation(
    rng: random.Random,
    parent: dict[str, float | int],
    defaults: dict[str, float | int],
) -> dict[str, float | int]:
    cand = dict(parent)
    mutated = False

    for key, values in SPACE.items():
        ordered = sorted(values)
        if rng.random() < 0.62:
            mutated = True
            base = cand.get(key, defaults[key])
            idx = _closest_index(ordered, base)
            if len(ordered) == 1:
                step = 0
            else:
                step = rng.choice([-2, -1, 1, 2])
                if rng.random() < 0.15:
                    step = rng.randint(-(len(ordered) - 1), (len(ordered) - 1))
            new_idx = max(0, min(len(ordered) - 1, idx + step))
            cand[key] = ordered[new_idx]

    if not mutated:
        key = rng.choice(list(SPACE.keys()))
        cand[key] = rng.choice(SPACE[key])

    return _normalize_params(cand)


def _suggest_candidate(
    *,
    eval_index: int,
    seed: int,
    defaults: dict[str, float | int],
    top_pool: list[dict],
    seen: set[str],
    explore_prob: float,
) -> tuple[dict[str, float | int], str]:
    rng = random.Random(seed ^ (eval_index * 104729))

    for _ in range(300):
        if (not top_pool) or (rng.random() < explore_prob):
            cand = _candidate_random(rng, defaults)
        else:
            # Rank-based parent selection from best pool
            picks = top_pool[: min(14, len(top_pool))]
            rank = int((rng.random() ** 1.8) * len(picks))
            parent = picks[rank]["params"]
            cand = _candidate_mutation(rng, parent, defaults)

        fp = _params_fingerprint(cand)
        if fp not in seen:
            return cand, fp

    # Fallback: allow duplicate if search space exhausted
    cand = _candidate_random(rng, defaults)
    return cand, _params_fingerprint(cand)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _append_text(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip())
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _load_history(history_path: Path) -> tuple[set[str], list[dict], int, int]:
    seen: set[str] = set()
    successful: list[dict] = []
    total = 0
    failed = 0

    if not history_path.exists():
        return seen, successful, total, failed

    with history_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # tolerate partial final line due sudden poweroff
                continue
            total += 1
            params = item.get("params")
            if isinstance(params, dict):
                seen.add(_params_fingerprint(params))
            if item.get("status") == "ok":
                successful.append(item)
            else:
                failed += 1

    successful.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            float(x.get("global_win_rate", 0.0)),
            -int(x.get("forfeits", 0)),
        ),
        reverse=True,
    )
    return seen, successful, total, failed


def _run_matchup(
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
    run_dir: Path,
    runner: str,
    max_retries: int,
    retry_delay: float,
    keep_match_json: bool,
) -> MatchupStats:
    host_json = run_dir / "match_json" / f"{run_tag}.json"
    host_json.parent.mkdir(parents=True, exist_ok=True)
    if host_json.exists():
        host_json.unlink()

    container_json = f"/app/estudiantes/{team}/results/{run_dir.name}/match_json/{run_tag}.json"
    black = strategy if as_black else opponent
    white = opponent if as_black else strategy

    if runner == "docker":
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
    else:
        # local mode (useful for quick smoke tests with Random)
        cmd = [
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
            str(host_json),
        ]
        for key, value in params.items():
            os.environ[key] = str(value)

    attempt = 0
    last_err = ""
    while attempt <= max_retries:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and host_json.exists():
            break
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        last_err = stderr if stderr else stdout
        attempt += 1
        if attempt <= max_retries:
            sleep_s = retry_delay * (1.6 ** (attempt - 1))
            time.sleep(sleep_s)
    else:
        raise RuntimeError(
            f"Fallo matchup {run_tag} tras {max_retries + 1} intentos. "
            f"Comando: {' '.join(cmd)} | Salida: {last_err[:1000]}"
        )

    data = json.loads(host_json.read_text(encoding="utf-8"))
    out = MatchupStats()

    for game in data.get("games", []):
        won = game.get("winner") == strategy
        if won:
            out.wins += 1
        else:
            out.losses += 1
        if game.get("forfeit") and not won:
            out.forfeits += 1
        out.games += 1
        for move in game.get("move_log", []):
            if move.get("strategy") == strategy:
                t = float(move.get("time_s", 0.0))
                out.my_move_count += 1
                out.my_move_time_sum += t
                if t > out.my_move_time_max:
                    out.my_move_time_max = t

    if not keep_match_json:
        try:
            host_json.unlink(missing_ok=True)
        except OSError:
            pass
    return out


def _evaluate_candidate(
    *,
    eval_id: int,
    repo_root: Path,
    run_dir: Path,
    team: str,
    strategy: str,
    opponents: list[str],
    variants: list[str],
    num_games: int,
    move_timeout: float,
    base_seed: int,
    both_colors: bool,
    seed_batches: int,
    params: dict[str, float | int],
    forfeit_penalty: float,
    runner: str,
    max_retries: int,
    retry_delay: float,
    keep_match_json: bool,
) -> dict:
    started = time.monotonic()
    colors = [True, False] if both_colors else [True]

    per_opp: dict[str, MatchupStats] = {opp: MatchupStats() for opp in opponents}
    global_stats = MatchupStats()

    for b in range(seed_batches):
        for o_idx, opp in enumerate(opponents):
            for v_idx, variant in enumerate(variants):
                for c_idx, as_black in enumerate(colors):
                    run_seed = (
                        base_seed
                        + eval_id * 100_003
                        + b * 10_007
                        + o_idx * 1_009
                        + v_idx * 101
                        + c_idx * 11
                    )
                    tag = (
                        f"eval{eval_id:06d}_b{b}_o{o_idx}_{opp}"
                        f"_{variant}_{'B' if as_black else 'W'}"
                    )
                    stats = _run_matchup(
                        repo_root=repo_root,
                        team=team,
                        strategy=strategy,
                        opponent=opp,
                        variant=variant,
                        num_games=num_games,
                        move_timeout=move_timeout,
                        seed=run_seed,
                        params=params,
                        as_black=as_black,
                        run_tag=tag,
                        run_dir=run_dir,
                        runner=runner,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        keep_match_json=keep_match_json,
                    )
                    per_opp[opp].merge(stats)
                    global_stats.merge(stats)

    weighted_score_sum = 0.0
    weight_sum = 0.0
    per_opp_payload: dict[str, dict] = {}
    for opp in opponents:
        st = per_opp[opp]
        weight = OPPONENT_WEIGHTS.get(opp, 1.0)
        wr = st.win_rate()
        fr = st.forfeit_rate()
        opp_score = wr - forfeit_penalty * fr
        weighted_score_sum += weight * opp_score
        weight_sum += weight
        per_opp_payload[opp] = {
            "wins": st.wins,
            "losses": st.losses,
            "forfeits": st.forfeits,
            "games": st.games,
            "win_rate": wr,
            "forfeit_rate": fr,
            "score": opp_score,
            "weight": weight,
            "avg_move_time": st.avg_move_time(),
            "max_move_time": st.my_move_time_max,
        }

    score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0
    out = {
        "status": "ok",
        "eval_id": eval_id,
        "timestamp": _now_iso(),
        "elapsed_sec": time.monotonic() - started,
        "params": params,
        "score": score,
        "global_win_rate": global_stats.win_rate(),
        "wins": global_stats.wins,
        "losses": global_stats.losses,
        "forfeits": global_stats.forfeits,
        "games": global_stats.games,
        "avg_move_time": global_stats.avg_move_time(),
        "max_move_time": global_stats.my_move_time_max,
        "per_opponent": per_opp_payload,
    }
    return out


def _write_best_artifacts(
    *,
    run_dir: Path,
    best: dict,
    strategy_path: Path,
    apply_to_strategy: bool,
) -> None:
    if not best:
        return
    best_params = best["params"]

    _atomic_write_json(run_dir / "best_params.json", best_params)
    env_lines = [f"{k}={best_params[k]}" for k in sorted(best_params.keys())]
    env_path = run_dir / "best_params.env"
    with env_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(env_lines))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    if apply_to_strategy:
        _apply_best_to_strategy(strategy_path, best_params)


def _score_key(item: dict) -> tuple[float, float, int]:
    return (
        float(item.get("score", 0.0)),
        float(item.get("global_win_rate", 0.0)),
        -int(item.get("forfeits", 0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bucle robusto de entrenamiento nocturno con checkpoints")
    parser.add_argument("--team", type=str, default="compus")
    parser.add_argument("--strategy", type=str, default="MiEstrategia_mi_equipo")
    parser.add_argument("--strategy-file", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="overnight_main")
    parser.add_argument("--fresh", action="store_true")

    parser.add_argument("--runner", choices=["docker", "local"], default="docker")
    parser.add_argument("--opponents", type=str, default="MCTS_Tier_3")
    parser.add_argument("--variants", type=str, default="classic,dark")
    parser.add_argument("--both-colors", action="store_true")
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--move-timeout", type=float, default=8.0)
    parser.add_argument("--seed-batches", type=int, default=1)

    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--max-evals", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--explore-prob", type=float, default=0.38)
    parser.add_argument("--top-pool-size", type=int, default=30)
    parser.add_argument("--forfeit-penalty", type=float, default=0.35)

    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=6.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=8)
    parser.add_argument("--keep-match-json", action="store_true")

    parser.add_argument("--no-apply-best", action="store_true")
    parser.add_argument("--apply-best-every", type=int, default=5)
    parser.add_argument("--no-apply-best-on-improve", action="store_true")
    args = parser.parse_args()

    if args.hours <= 0:
        raise ValueError("--hours debe ser > 0")
    if args.num_games < 1:
        raise ValueError("--num-games debe ser >= 1")
    if args.seed_batches < 1:
        raise ValueError("--seed-batches debe ser >= 1")
    if not (0.0 <= args.explore_prob <= 1.0):
        raise ValueError("--explore-prob debe estar en [0, 1]")

    repo_root = _repo_root()
    opponents = _parse_csv_list(args.opponents)
    variants = _parse_variants(args.variants)

    if args.strategy_file:
        strategy_path = (repo_root / args.strategy_file).resolve()
    else:
        strategy_path = repo_root / "estudiantes" / args.team / "strategy.py"

    defaults = _load_strategy_defaults(strategy_path)
    defaults = _normalize_params(defaults)

    run_dir = repo_root / "estudiantes" / args.team / "results" / args.run_name
    checkpoint_path = run_dir / "checkpoint.json"
    history_path = run_dir / "history.jsonl"
    summary_path = run_dir / "summary.json"
    errors_path = run_dir / "errors.log"

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        archive_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if checkpoint_path.exists() or history_path.exists():
            archived = run_dir.parent / f"{run_dir.name}_archive_{archive_stamp}"
            run_dir.rename(archived)
            run_dir.mkdir(parents=True, exist_ok=True)

    seen, successful, total_prev, failed_prev = _load_history(history_path)
    top_pool = successful[: args.top_pool_size]
    best = top_pool[0] if top_pool else None

    started_at = _now_iso()
    if checkpoint_path.exists() and not args.fresh:
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            started_at = cp.get("started_at", started_at)
        except Exception:
            pass

    print(f"Run dir: {run_dir}")
    print(f"Estrategia: {args.strategy}")
    print(f"Oponentes: {opponents}")
    print(
        f"Variants={variants} both_colors={args.both_colors} "
        f"num_games={args.num_games} seed_batches={args.seed_batches}"
    )
    print(
        f"Resume: prev_evals={total_prev}, prev_success={len(successful)}, "
        f"prev_failed={failed_prev}"
    )

    started_monotonic = time.monotonic()
    eval_id = total_prev
    consecutive_failures = 0
    apply_on_improve = not args.no_apply_best_on_improve
    apply_best_enabled = not args.no_apply_best

    while True:
        elapsed_hours = (time.monotonic() - started_monotonic) / 3600.0
        if eval_id >= args.max_evals:
            print("Stop: max evals alcanzado")
            break
        if elapsed_hours >= args.hours:
            print("Stop: limite de horas alcanzado")
            break

        eval_id += 1
        params, fp = _suggest_candidate(
            eval_index=eval_id,
            seed=args.seed,
            defaults=defaults,
            top_pool=top_pool,
            seen=seen,
            explore_prob=args.explore_prob,
        )
        seen.add(fp)

        print(f"\n[{eval_id}] params={params}")
        try:
            result = _evaluate_candidate(
                eval_id=eval_id,
                repo_root=repo_root,
                run_dir=run_dir,
                team=args.team,
                strategy=args.strategy,
                opponents=opponents,
                variants=variants,
                num_games=args.num_games,
                move_timeout=args.move_timeout,
                base_seed=args.seed,
                both_colors=args.both_colors,
                seed_batches=args.seed_batches,
                params=params,
                forfeit_penalty=args.forfeit_penalty,
                runner=args.runner,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                keep_match_json=args.keep_match_json,
            )
            consecutive_failures = 0
            improved = best is None or (_score_key(result) > _score_key(best))
            print(
                f"  score={result['score']:.3f} wr={result['global_win_rate']:.3f} "
                f"forfeits={result['forfeits']}/{result['games']} "
                f"avg_move={result['avg_move_time']:.3f}s "
                f"elapsed={result['elapsed_sec']:.1f}s"
            )
            _append_jsonl(history_path, result)
            successful.append(result)
            successful.sort(key=_score_key, reverse=True)
            top_pool = successful[: args.top_pool_size]
            best = top_pool[0]

            if improved:
                print(f"  NEW BEST score={best['score']:.3f} params={best['params']}")
                _write_best_artifacts(
                    run_dir=run_dir,
                    best=best,
                    strategy_path=strategy_path,
                    apply_to_strategy=apply_best_enabled and apply_on_improve,
                )
            elif (
                apply_best_enabled
                and args.apply_best_every > 0
                and (eval_id % args.apply_best_every == 0)
            ):
                _write_best_artifacts(
                    run_dir=run_dir,
                    best=best,
                    strategy_path=strategy_path,
                    apply_to_strategy=True,
                )

        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            err = {
                "status": "failed",
                "eval_id": eval_id,
                "timestamp": _now_iso(),
                "params": params,
                "error": str(exc),
            }
            _append_jsonl(history_path, err)
            _append_text(errors_path, f"[{err['timestamp']}] eval={eval_id} error={exc}")
            print(f"  FAILED: {exc}")
            if consecutive_failures >= args.max_consecutive_failures:
                print("Stop: demasiados fallos consecutivos")
                break

        checkpoint_payload = {
            "version": 1,
            "run_name": args.run_name,
            "started_at": started_at,
            "updated_at": _now_iso(),
            "strategy": args.strategy,
            "strategy_file": str(strategy_path),
            "team": args.team,
            "runner": args.runner,
            "opponents": opponents,
            "variants": variants,
            "both_colors": args.both_colors,
            "num_games": args.num_games,
            "move_timeout": args.move_timeout,
            "seed_batches": args.seed_batches,
            "hours": args.hours,
            "max_evals": args.max_evals,
            "seed": args.seed,
            "forfeit_penalty": args.forfeit_penalty,
            "explore_prob": args.explore_prob,
            "top_pool_size": args.top_pool_size,
            "apply_best_every": args.apply_best_every,
            "defaults_at_start": defaults,
            "history_path": str(history_path),
            "summary_path": str(summary_path),
            "checkpoint_path": str(checkpoint_path),
            "evals_done": eval_id,
            "successful_evals": len(successful),
            "consecutive_failures": consecutive_failures,
            "best": best,
            "top_pool": top_pool[: min(10, len(top_pool))],
        }
        _atomic_write_json(checkpoint_path, checkpoint_payload)

        summary_payload = {
            "updated_at": _now_iso(),
            "run_name": args.run_name,
            "evals_done": eval_id,
            "successful_evals": len(successful),
            "best": best,
            "top10": top_pool[: min(10, len(top_pool))],
        }
        _atomic_write_json(summary_path, summary_payload)

    # Final persist
    if best is not None:
        _write_best_artifacts(
            run_dir=run_dir,
            best=best,
            strategy_path=strategy_path,
            apply_to_strategy=apply_best_enabled,
        )
    print(f"\nFinalizado. Checkpoint: {checkpoint_path}")
    if best is not None:
        print(f"Mejor score={best['score']:.3f} params={best['params']}")
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
