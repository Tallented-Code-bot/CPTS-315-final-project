from collections import defaultdict
from typing import List

import numpy as np
from data_collector import GameType


def manual_hash(game: List[float]) -> int:
    match game:
        case [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]:
            return 0
        case [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
            return 1
        case [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]:
            return 2
        case [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]:
            return 3
        case [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]:
            return 4
        case [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]:
            return 5
        case [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]:
            return 6
        case [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]:
            return 7
        case [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]:
            return 8
    return -1


def most_picked(games: List[List[float]]):
    order = [0 for _ in range(9)]
    for game in games:
        order[manual_hash(game)] += 1
    max_value = max(order)
    return order.index(max_value)


def get_game(index: int) -> List[float]:
    match index:
        case 0:
            return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        case 1:
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        case 2:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        case 3:
            return [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        case 4:
            return [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        case 5:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        case 6:
            return [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        case 7:
            return [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        case 8:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


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


def print_table(T, supp_count):
    print("Itemset | Frequency")
    for k in range(len(T)):
        print("{} : {}", format(T[k], supp_count[k]))
    print("\n\n")


def count_occurances(itemset, Transactions):
    count = 0
    for i in range(len(Transactions)):
        if set(itemset).issubset(set(Transactions[i])):
            count += 1
    return count


def join_two_itemsets(it1, it2, order):
    it1.sort(key=lambda x: order.index(x))
    it2.sort(key=lambda x: order.index(x))

    for i in range(len(it1) - 1):
        if it1[i] != it2[i]:
            return []

    if order.index(it1[-1]) < order.index(it2[-1]):
        return it1 + [it2[-1]]

    return []


def join_set_itemsets(set_of_its, order):
    C = []
    for i in range(len(set_of_its)):
        for j in range(i + 1, len(set_of_its)):
            it_out = join_two_itemsets(set_of_its[i], set_of_its[j], order)
            if len(it_out) > 0:
                C.append(it_out)
    return C


def get_frequent(itemsets, Transactions, minsup, prev_discarded):
    L = []  # list to store freq_itemsets
    supp_count = []  # counts of itemsets
    new_discarded = []  # items we discard in this itteration
    num_transactions = len(Transactions)
    k = len(prev_discarded.keys())

    for s in range(len(itemsets)):
        # check if item contains previously discarded itemset
        discarded_before = False
        if k > 0:
            for it in prev_discarded[k]:
                if set(it).issubset(set(itemsets[s])):
                    discarded_before = True
                    break
        if not discarded_before:
            count = count_occurances(itemsets[s], Transactions)
            if count / num_transactions >= minsup:
                L.append(itemsets[s])
                supp_count.append(count)
            else:
                new_discarded.append(itemsets[s])
    return L, supp_count, new_discarded


def new_apriori(minsup: int, games: List[float]):
    order = [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ]
    order = [tuple(x) for x in order]
    C = {}
    L = {}
    itemset_size = 1
    Discarded = {itemset_size: []}
    C.update({itemset_size: [tuple(f) for f in order]})
    # this should be C_1

    supp_count_L = {}  # idk why dictionary
    f, sup, new_discarded = get_frequent(C[itemset_size], games, minsup, Discarded)
    Discarded.update({itemset_size: new_discarded})
    L.update({itemset_size: f})
    supp_count_L.update({itemset_size: sup})

    k = itemset_size + 1
    convergence = 0
    while not convergence:
        C.update({k: join_set_itemsets(L[k - 1], order)})
        print("Table C{}: \n".format(k))
        print_table(C[k], [count_occurances(it, games) for it in C[k]])
        f, sup, new_discarded = get_frequent(C[k], games, minsup, Discarded)
        Discarded.update({k: new_discarded})
        L.update({k: f})
        supp_count_L.update({k: sup})
        if len(L[k]) == 0:
            convergence = True
        k += 1


# def mostFreq(games: List[GameType], minsup: int = 2) -> GameType | None:
#     freq_item_sets = apriori(minsup, games)
#     if not freq_item_sets:
#         return None

#     game_sets = [set(map(tuple, g)) for g in games]

#     def count_support(itemset: frozenset) -> int:
#         return sum(1 for g in game_sets if itemset.issubset(g))

#     most_frequent = max(freq_item_sets, key=count_support)

#     # Convert the winning frozenset back to a numpy array (3x2 shape per GameType)
#     return np.array(list(most_frequent), dtype=np.float64)
