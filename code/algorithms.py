from collections import defaultdict
from typing import List

import numpy as np
from data_collector import GameType


def generate_candidates(freq_item_set: set, k: int):
    candidates = set()
    list_freq = list(freq_item_set)

    for i in range(len(list_freq)):
        for j in range(i + 1, len(list_freq)):
            new_cand = list_freq[i] | list_freq[j]  # Set Union
            if len(new_cand) == k:
                subsets = [new_cand - {item} for item in new_cand]
                if all(s in freq_item_set for s in subsets):
                    candidates.add(new_cand)
    return candidates


def apriori(minsup: int, games: List[GameType]):
    counts = defaultdict(int)
    for game in games:
        for item in game:
            counts[frozenset([item])] += 1

    current_freq_sets = {
        itemset for itemset, count in counts.items() if count >= minsup
    }
    all_frequent_sets = list(current_freq_sets)

    k = 2
    while current_freq_sets:
        candidates = generate_candidates(current_freq_sets, k)

        cand_counts = defaultdict(int)
        game_sets = [set(g) for g in games]

        for cand in candidates:
            for g_set in game_sets:
                if cand.issubset(g_set):
                    cand_counts[cand] += 1

        current_freq_sets = {
            cand for cand, count in cand_counts.items() if count >= minsup
        }
        all_frequent_sets.extend(list(current_freq_sets))
        k += 1

    return all_frequent_sets


def mostFreq(games: List[GameType], minsup: int = 2) -> GameType | None:
    freq_item_sets = apriori(minsup, games)
    if not freq_item_sets:
        return None

    game_sets = [set(map(tuple, g)) for g in games]

    def count_support(itemset: frozenset) -> int:
        return sum(1 for g in game_sets if itemset.issubset(g))

    most_frequent = max(freq_item_sets, key=count_support)

    # Convert the winning frozenset back to a numpy array (3x2 shape per GameType)
    return np.array(list(most_frequent), dtype=np.float64)
