from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _latest_session_csv(logs_dir: Path) -> Path:
    # Pick the newest session_*.csv by mtime.
    candidates = sorted(logs_dir.glob("session_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No session logs found in {logs_dir}. Run code/new_algorithm/play.py first."
        )
    return candidates[-1]


def _default_out_path(log_path: Path) -> Path:
    # session_YYYYMMDD_HHMMSS.csv -> session_..._winrate.png
    return log_path.with_name(f"{log_path.stem}_winrate.png")


def plot_winrate(*, log_path: Path, out_path: Path, window: int) -> None:
    df = pd.read_csv(log_path)
    missing = {"round", "outcome_agent"} - set(df.columns)
    if missing:
        raise ValueError(f"Log file missing columns: {sorted(missing)}")

    rounds = df["round"].astype(int)
    # outcome_agent: 1 win, 0 tie, -1 loss
    win = (df["outcome_agent"].astype(int) == 1).astype(float)
    winrate = win.rolling(window=max(int(window), 1), min_periods=1).mean()

    # Summary stats for title.
    overall = float(win.mean()) if len(win) else 0.0
    final_rolling = float(winrate.iloc[-1]) if len(winrate) else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        rounds, winrate, label=f"Rolling mean win rate (window={max(int(window), 1)})"
    )
    ax.axhline(
        1.0 / 3.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Random baseline (1/3)",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(
        f"Win Rate vs Round ({log_path.name})\n"
        f"overall={overall:.3f}  final_rolling={final_rolling:.3f}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot rolling mean win rate vs round from a new_algorithm session CSV"
    )
    ap.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to log",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="PNG path",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size",
    )
    args = ap.parse_args()

    logs_dir = Path(__file__).resolve().parent / "logs"
    log_path = Path(args.log) if args.log else _latest_session_csv(logs_dir)
    out_path = Path(args.out) if args.out else _default_out_path(log_path)

    plot_winrate(log_path=log_path, out_path=out_path, window=int(args.window))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
