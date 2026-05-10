# test
import pickle
from random import random
from typing import Annotated


from algorithms import *
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import matplotlib.pyplot as plt

PredictionsType = Annotated[NDArray[np.float64], 3]
GameType = Annotated[NDArray[np.float64], 3, 2]

ROCK = np.array([1, 0, 0])
PAPER = np.array([0, 1, 0])
SCISSORS = np.array([0, 0, 1])


PLAYER = 0
TIE = 1
COMPUTER = 2


# def computerPredict() -> PredictionsType:
#     """Function that returns a np.array of size 3 indicating the odds of each
#     option to be correct"""
#     return np.ndarray([3])


def computerPlay(predictions: PredictionsType) -> PredictionsType:
    """Function to return the played option (rock, paper, or scissors) based on probabilities."""

    # Randomly choose based on the probabilities in predictions
    # options = [ROCK, PAPER, SCISSORS]
    options = [1, 2, 3]
    picked = np.random.choice(options, p=predictions)
    match picked:
        case 1:
            return ROCK
        case 2:
            return PAPER
        case 3:
            return SCISSORS
        case _:
            raise ValueError("Invalid Option")


def computerRandom() -> PredictionsType:
    return np.array([random(), random(), random()])


def playSingleGame() -> GameType:
    """Interactively plays a single game"""

    play = np.random.choice(["rock", "paper", "scissors"]) #random pick fct, for debug
    # print(f"(r)ock, (p)aper, or (s)cissors")
    # play = input()
    playerPlay = np.array([0.0, 0.0, 0.0],)
    match play:
        case "rock" | "r":
            playerPlay[0] = 1
        case "paper" | "p":
            playerPlay[1] = 1
        case "scissors" | "s":
            playerPlay[2] = 1
        case _:
            raise ValueError("Invalid Option")

    computerValue = computerPlay(np.array([0.34, 0.33, 0.33]))

    game = np.array([playerPlay, computerValue])

    print(f"Player play: {playerPlay}, Computer play: {computerValue}")  # Debugging

    winner = determineWinner(game)

    if winner == PLAYER:
        print("You win!")
    elif winner == COMPUTER:
        print("You lose!")
    else:
        print("It's a tie!")

    return game


def determineWinner(game: GameType) -> int:
    # 0 -> player
    # 1 -> tie
    # 2 -> computer

    playerPlay, computerValue = map(np.array, game)

    match playerPlay.tolist(), computerValue.tolist():
        case [1, 0, 0], [1, 0, 0]:
            return TIE
        case [1, 0, 0], [0, 1, 0]:
            return COMPUTER
        case [1, 0, 0], [0, 0, 1]:
            return PLAYER
        case [0, 1, 0], [1, 0, 0]:
            return PLAYER
        case [0, 1, 0], [0, 1, 0]:
            return TIE
        case [0, 1, 0], [0, 0, 1]:
            return COMPUTER
        case [0, 0, 1], [1, 0, 0]:
            return COMPUTER
        case [0, 0, 1], [0, 1, 0]:
            return PLAYER
        case [0, 0, 1], [0, 0, 1]:
            return TIE
        case _:
            return TIE


def collectData(timesPlayed: int) -> list:
    """Plays a series of games and collects data for each one"""

    games = []  # List to store games played

    for _ in range(timesPlayed):
        game = playSingleGame()  # Play a single game and collect the result
        games.append(game)  # Append each game to the list
        print(f"Game {_ + 1}: {game}")  # Debugging

    return games  # Return the collected games


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
    print("Simulating Rock-Paper-Scissors games...")

    allGames = []
    wlr = []
    # allGames = pickle.load(open("allGames.pkl", "rb"))

    for i in range(5):
        gameSeries = collectData(5)  # Simulate a series of 5 games
        print(f"Game series {i + 1}: {gameSeries}")  # Debugging
        allGames.append(gameSeries)  # Append the returned series of games
        for game, index in enumerate(gameSeries):
            print(game)
            if determineWinner(game) == 0:
                wlr[i] += 1
            elif determineWinner(game) == 2: 
                wlr[i] -= 1

    pickle.dump(allGames, open("allGames.pkl", "wb"))

    # calling apriori:
    com_games = []
    player_games = []
    full_games = []
    
    for series in allGames:
        for game in series:
            full_games.append(game[0].tolist() + game[1].tolist())
    #print(full_games)
    freq = {}
    sup_ct = {}
    freq, sup_ct = new_apriori(2, full_games)


    plt.plot(wlr)
    plt.show()

if __name__ == "__main__":
    main()
