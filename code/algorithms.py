from typing import List

from data_collector import GameType


def generate_candidates(freq_item_set: List[GameType]):
    cand_item_set = []
    for item_set in freq_item_set:
        for item in item_set:
            cand_item_set.append(item_set + [item])
    return cand_item_set


def apriori(minsup: int, games: List[GameType]):
    freq_item_set = []
    cand_item_set = []

    # get L_1, initial frequent items
    while freq_item_set:
        """
        While L_k is not empty
        """
        cand_item_set = generate_candidates(freq_item_set)
        for game in games:
            for item_set in cand_item_set:
                if set(item_set).issubset(set(game)):
                    freq_item_set.append(item_set)
        if not cand_item_set:
            break
        freq_item_set = []
        for item_set in cand_item_set:
            if cand_item_set.count(item_set) >= minsup:
                freq_item_set.append(item_set)
        cand_item_set = []
        if not freq_item_set:
            break
    return freq_item_set


def mostFreq(games: List[GameType]):
    freq_item_set = apriori(2, games)
    if not freq_item_set:
        return None
    return max(freq_item_set, key=lambda x: games.count(x))
