import pickle
from itertools import accumulate
from random import random
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

GameType = Annotated[NDArray[np.float64], 3, 2]

ROCK = np.array([1, 0, 0])
PAPER = np.array([0, 1, 0])
SCISSORS = np.array([0, 0, 1])


PLAYER = -1
TIE = 0
COMPUTER = 1


def determineWinner(game: GameType) -> int:
    # -1 -> player
    # 0 -> tie
    # 1 -> computer

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


def main():
    try:
        try:
            with open("allGames.pkl", "rb") as f:
                games = pickle.load(f)  # list of lists of games
                print(
                    "Loaded games:", games
                )  # Debugging statement to inspect loaded data
                # Proceed to plot the data after debugging
                games = [
                    game for series in games if series is not None for game in series
                ]
                # Flatten the nested list structure
                games = [
                    game for series in games if series is not None for game in series
                ]
        except FileNotFoundError:
            print(
                "Error: The file 'allGames.pkl' was not found. Please ensure it exists."
            )
            return
        except (pickle.UnpicklingError, ValueError) as e:
            print(f"Error loading data: {e}")
            return
    except FileNotFoundError:
        print("Error: The file 'allGames.pkl' was not found. Please ensure it exists.")
        return
    except (pickle.UnpicklingError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

        # [ # list of series
        #   [ # series
        #      [[1,0,0], [0,1,0]] # game
        #      [[1,0,0], [0,1,0]]
        #      [[1,0,0], [0,1,0]]
        #      [[1,0,0], [0,1,0]]
        #    ],
        #    [
        #      [[1,0,0], [0,1,0]] # game
        #      [[1,0,0], [0,1,0]]
        #      [[1,0,0], [0,1,0]]
        #      [[1,0,0], [0,1,0]]
        #    ]
        # ]

        # gameArray = np.array(games)
        # gameDF = pd.DataFrame(
        #     {
        #         "Human": gameArray[:, 0],
        #         "Computer": gameArray[:, 1],
        #     }
        # )
        # gameDF["Winner"] = list(
        #     accumulate(gameDF.apply(lambda row: determineWinner(row), axis=1))
        # )
        # plt.plot(gameDF.index, gameDF["Winner"])
        # plt.show()
        #
        # Create a DataFrame for all games
        gameDF = pd.DataFrame(
            {
                "Human": [game[0] for game in games],
                "Computer": [game[1] for game in games],
            }
        )
        # Calculate cumulative scores
        gameDF["Winner"] = list(
            accumulate(gameDF.apply(lambda row: determineWinner(row), axis=1))
        )

        # Plot cumulative scores
        plt.plot(gameDF.index, gameDF["Winner"])
        plt.xlabel("Game")
        plt.ylabel("Cumulative Score")
        try:
            save_path = "cumulative_scores.png"
            plt.savefig(
                save_path
            )  # Save the plot as an image file in the current directory
            print(f"Plot successfully saved as '{save_path}' in the current directory.")
        except Exception as e:
            print(f"Failed to save the plot: {e}")


if __name__ == "__main__":
    main()
