# test
from random import random
from typing import Annotated

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PredictionsType = Annotated[NDArray[np.float64], 3]
GameType = Annotated[NDArray[np.int32], 3, 2]

ROCK = np.array([1, 0, 0])
PAPER = np.array([0, 1, 0])
SCISSORS = np.array([0, 0, 1])


def computerPredict() -> PredictionsType:
    """Function that returns a np.array of size 3 indicating the odds of each
    option to be correct"""
    return np.ndarray([3])


def computerPlay(predictions: PredictionsType) -> PredictionsType:
    """Function to return the played option (rock, paper, or scissors)"""

    return ROCK


def computerRandom() -> PredictionsType:
    return np.array([random(), random(), random()])


def compare(arg: GameType):
    pass


def playSingleGame():
    """Interactively plays a single game"""

    play = input("Enter your play")
    actualPlay = np.array([0.0, 0.0, 0.0])
    match play:
        case "rock" | "r":
            actualPlay[0] = 1
        case "paper" | "p":
            actualPlay[1] = 1
        case "scissors" | "s":
            actualPlay[2] = 1
        case _:
            print("Invalid option")
            return

    computerValue = computerPlay(ROCK)
    # computerValue = computerPlay(computerPredict(np.array([0.33, 0.33, 0.33])))

    game = np.array([actualPlay, computerValue])
    return game


def collectData(timesPlayed: int):
    """Plays a series of games and collects data for each one"""

    for _ in range(timesPlayed):
        game = playSingleGame()


# one series:
# [
#  [item 1, item 2]
#
#
# ]


# prediction array
# [ 0.5, 0,7, 0.2] (between 0 and 1, probability for rock, paper, scissors)
#
def main():
    # TODO: replace with gameplay function
    print("Play rock paper scisors")

    pass


if __name__ == "__main__":
    main()
