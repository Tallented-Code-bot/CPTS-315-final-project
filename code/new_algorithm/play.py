from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

from .agent import Move, RPSAgent, move_to_str, parse_move


def _fmt_dist(p: np.ndarray) -> str:
    p = np.asarray(p, dtype=np.float64)
    return f"R:{p[0]:.2f} P:{p[1]:.2f} S:{p[2]:.2f}"


def _default_log_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "logs" / f"session_{ts}.csv"

def main() -> None:
    ap = argparse.ArgumentParser(description="Online opponent-modeling RPS")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help="CSV log path (optional)",
    )
    args = ap.parse_args()

    # Tuned defaults for "beat humans quickly".
    agent = RPSAgent(
        gamma=0.95,
        alpha=1.0,
        eta=1.0,
        weight_decay=0.98,
        tau=0.35,
        uniform_mix=0.05,
        seed=args.seed,
    )
    opp = None

    log_path = Path(args.log) if args.log else _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "opp_move",
                "agent_move",
                "outcome_agent",
                "agent_score",
                "pR",
                "pP",
                "pS",
                "weights",
            ]
        )

        last_outcome_my: int | None = None
        my_last: Move | None = None

        def play_one(opp_move: Move) -> None:
            nonlocal last_outcome_my, my_last
            p_hat = agent.predict_distribution()
            agent_move = agent.choose_move(p_hat)
            out = agent.observe(agent_move, opp_move)
            my_last = agent_move
            last_outcome_my = out

            weights = {
                agent.expert_names[i]: float(agent.weights[i])
                for i in range(len(agent.expert_names))
            }
            w.writerow(
                [
                    agent.rounds,
                    move_to_str(opp_move),
                    move_to_str(agent_move),
                    out,
                    agent.score,
                    float(p_hat[0]),
                    float(p_hat[1]),
                    float(p_hat[2]),
                    weights,
                ]
            )

            # Print concise diagnostics.
            top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:2]
            top_str = ", ".join([f"{k}:{v:.2f}" for k, v in top])
            print(
                f"opp={move_to_str(opp_move)} agent={move_to_str(agent_move)} out={out:+d} score={agent.score:+d} | "
                f"p_hat({_fmt_dist(p_hat)}) | top[{top_str}]"
            )

        print("Enter your move: r/p/s (or q to quit)")
        while True:
            try:
                raw = input("> ").strip()
            except EOFError:
                print(f"\nLog written to {log_path}")
                return
            if raw.lower() in {"q", "quit", "exit"}:
                print(f"Log written to {log_path}")
                return
            try:
                opp_move = parse_move(raw)
            except ValueError as e:
                print(str(e))
                continue
            play_one(opp_move)


if __name__ == "__main__":
    main()
