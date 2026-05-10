from __future__ import annotations

import math
from enum import IntEnum

import numpy as np


class Move(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


def parse_move(s: str) -> Move:
    s = s.strip().lower()
    match s:
        case "r" | "rock":
            return Move.ROCK
        case "p" | "paper":
            return Move.PAPER
        case "s" | "scissors" | "scissor":
            return Move.SCISSORS
        case _:
            raise ValueError("Expected r/p/s (or rock/paper/scissors)")


def move_to_str(m: Move) -> str:
    match Move(m):
        case Move.ROCK:
            return "R"
        case Move.PAPER:
            return "P"
        case Move.SCISSORS:
            return "S"
    raise ValueError("Invalid move")


def counter_to(m: Move) -> Move:
    # Move that beats m.
    match Move(m):
        case Move.ROCK:
            return Move.PAPER
        case Move.PAPER:
            return Move.SCISSORS
        case Move.SCISSORS:
            return Move.ROCK
    raise ValueError("Invalid move")


# return 1 if you won, 0 for a tie, -1 for a loss
def payoff(my_move: Move, opp_move: Move) -> int:
    if my_move == opp_move:
        return 0
    if (
        (my_move == Move.ROCK and opp_move == Move.SCISSORS)
        or (my_move == Move.PAPER and opp_move == Move.ROCK)
        or (my_move == Move.SCISSORS and opp_move == Move.PAPER)
    ):
        return 1
    return -1


# norm a np array
def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    return p / s


#
def expected_values(p_opp: np.ndarray) -> np.ndarray:
    p = _norm(p_opp)
    p_r, p_p, p_s = p.tolist()
    return np.array(
        [
            (1.0 * p_s) + (-1.0 * p_p),  # play R
            (1.0 * p_r) + (-1.0 * p_s),  # play P
            (1.0 * p_p) + (-1.0 * p_r),  # play S
        ],
        dtype=np.float64,
    )


class RPSAgent:
    def __init__(
        self,
        *,
        gamma: float = 0.98,
        alpha: float = 1.0,
        eta: float = 0.6,
        weight_decay: float = 0.995,
        tau: float = 0.7,
        uniform_mix: float = 0.15,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.weight_decay = float(weight_decay)
        self.tau = float(tau)
        self.uniform_mix = float(uniform_mix)
        self.eps = 1e-6

        self.rounds = 0
        self.score = 0

        self.my_last: Move | None = None
        self.opp_last: Move | None = None
        self.opp_last2: Move | None = None
        self.last_outcome_opp: int | None = None

        # Expert state.
        self.freq = np.zeros(3, dtype=np.float64)
        self.m1 = np.zeros((3, 3), dtype=np.float64)
        self.m2 = np.zeros((3, 3, 3), dtype=np.float64)
        self.outcome = np.zeros((3, 3, 3), dtype=np.float64)  # last_opp, outcome, next

        self.expert_names = [
            "freq",
            "markov1",
            "markov2",
            "outcome",
            "copy",
            "anticopy",
        ]
        self.weights = np.ones(len(self.expert_names), dtype=np.float64) / float(
            len(self.expert_names)
        )

    def _smooth(self, counts: np.ndarray) -> np.ndarray:
        return _norm(counts + self.alpha)

    def _expert_probs(self) -> list[np.ndarray]:
        # Each returns P(opp_next=R/P/S).
        out: list[np.ndarray] = []

        out.append(self._smooth(self.freq))

        if self.opp_last is None:
            out.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
        else:
            out.append(self._smooth(self.m1[int(self.opp_last)]))

        if self.opp_last2 is None or self.opp_last is None:
            out.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
        else:
            out.append(self._smooth(self.m2[int(self.opp_last2), int(self.opp_last)]))

        if self.opp_last is None or self.last_outcome_opp is None:
            out.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
        else:
            oi = {-1: 0, 0: 1, 1: 2}[int(self.last_outcome_opp)]
            out.append(self._smooth(self.outcome[int(self.opp_last), oi]))

        if self.my_last is None:
            out.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
            out.append(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
        else:
            p = np.zeros(3, dtype=np.float64)
            p[int(self.my_last)] = 1.0
            out.append(p)

            q = np.zeros(3, dtype=np.float64)
            q[int(counter_to(self.my_last))] = 1.0
            out.append(q)

        return out

    def predict_distribution(self) -> np.ndarray:
        probs = self._expert_probs()
        P = np.stack(probs, axis=0)
        return _norm((self.weights[:, None] * P).sum(axis=0))

    def choose_move(self, p_opp: np.ndarray) -> Move:
        ev = expected_values(p_opp)

        if self.tau <= 0:
            m = float(np.max(ev))
            pi = (ev == m).astype(np.float64)
            pi = pi / float(pi.sum())
        else:
            z = ev / self.tau
            z = z - float(np.max(z))
            pi = np.exp(z)
            pi = pi / float(pi.sum())

        a = min(max(self.uniform_mix, 0.0), 1.0)
        pi = (1.0 - a) * pi + a * (np.ones(3, dtype=np.float64) / 3.0)
        pi = _norm(pi)

        idx = int(self.rng.choice([0, 1, 2], p=pi))
        return Move(idx)

    def _update_weights(self, probs: list[np.ndarray], opp_move: Move) -> None:
        self.weights = np.power(self.weights, self.weight_decay)
        self.weights = self.weights / float(self.weights.sum())

        j = int(opp_move)
        for k, p in enumerate(probs):
            pr = float(p[j])
            pr = min(max(pr, self.eps), 1.0)
            self.weights[k] *= math.exp(-self.eta * (-math.log(pr)))

        s = float(self.weights.sum())
        if not np.isfinite(s) or s <= 0:
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights /= s

    def observe(self, my_move: Move, opp_move: Move) -> int:
        out_my = payoff(my_move, opp_move)
        out_opp = -out_my

        probs = self._expert_probs()
        self._update_weights(probs, opp_move)

        # Update model tables.
        g = self.gamma
        self.freq *= g
        self.m1 *= g
        self.m2 *= g
        self.outcome *= g

        self.freq[int(opp_move)] += 1.0
        if self.opp_last is not None:
            self.m1[int(self.opp_last), int(opp_move)] += 1.0
        if self.opp_last2 is not None and self.opp_last is not None:
            self.m2[int(self.opp_last2), int(self.opp_last), int(opp_move)] += 1.0
        if self.opp_last is not None and self.last_outcome_opp is not None:
            oi = {-1: 0, 0: 1, 1: 2}[int(self.last_outcome_opp)]
            self.outcome[int(self.opp_last), oi, int(opp_move)] += 1.0

        # Roll state.
        self.opp_last2 = self.opp_last
        self.opp_last = opp_move
        self.my_last = my_move
        self.last_outcome_opp = int(out_opp)

        self.rounds += 1
        self.score += int(out_my)
        return int(out_my)
