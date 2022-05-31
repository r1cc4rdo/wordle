from pathlib import Path
from itertools import product
from collections import Counter, defaultdict, OrderedDict

import tqdm
import numpy as np


def average_number_of_guesses(vocabulary, target_set, show_progress=True):

    assert len(target_set) > 0
    if len(target_set) == 1:
        return {'': 0}

    results = {}
    progress_bar = tqdm.tqdm(vocabulary) if show_progress else vocabulary
    for guess in progress_bar:

        reduced_vocabulary = sorted(set(vocabulary) - set([guess]))

        hint_to_solutions = defaultdict(list)
        for solution in target_set:
            hint_to_solutions[guess_to_hint(guess, solution)].append(solution)

        if len(hint_to_solutions) == 1:  # no information gained
            results[guess] = 1e30
            continue

        num_guesses = {}
        for hint in hint_to_solutions:
            reduced_target_set = filter_wordlist(target_set, guess, hint)

            if len(reduced_target_set) == 1:
                num_guesses[hint] = 1.0
                continue

            if len(reduced_target_set) == len(target_set):  # no information gained
                num_guesses[hint] = 1e30
                continue

            guess_avg = average_number_of_guesses(reduced_vocabulary, reduced_target_set, show_progress=False)
            num_guesses[hint] = 1 + min(guess_avg.values())

        results[guess] = sum(num_guesses.values()) / len(num_guesses)
        if results[guess] == 1:  # this is the best possible scenario, and we only need one
            break

    return results


def naive_best_starting_word_faster(vocabulary, target_set, cached_hints, hint_num):
    """
    We call this naive because it only considers the average number of remaining solutions after one guess.
    A non-naive solution would minimize instead the average number of required guesses.

    guess >> 'aback'
    Counter({'xxxxx': 925, 'yxxxx': 287, 'xxgxx': 159, 'xxxyx': 147, 'xyxxx': 134, 'gxxxx': 81, 'yxxyx': 63, 'xxxgx': 62,
    'yyxxx': 47, 'xxxxy': 44, 'xxgyx': 39, 'xxxgg': 30, 'xxxxg': 24, 'yxyxx': 21, 'xygxx': 20, 'yxxgx': 16, 'xxggx': 15,
    'xxgxg': 15, 'gxgxx': 14, 'gxyxx': 13, 'xxggg': 13, 'xxgxy': 13, 'ggxxx': 11, 'yxxxy': 9, 'gxxyx': 7, 'yyxyx': 7,
    'xyxgx': 7, 'xyxyx': 7, 'gyxxx': 6, 'xxxyg': 6, 'xxxyy': 6, 'yxxxg': 6, 'yxgxx': 5, 'gxxxy': 4, 'xyxxg': 4, 'yxyyx': 4,
    'yxxyg': 4, 'xyxxy': 3, 'xxgyg': 3, 'yxyxy': 3, 'gggxx': 2, 'yyxxy': 2, 'yyyxx': 2, 'xyggx': 2, 'yyxxg': 2, 'xyxgg': 2,
    'xgxxx': 2, 'yxxyy': 2, 'ggggg': 1, 'gxgxy': 1, 'yyxgx': 1, 'xyggg': 1, 'xygxg': 1, 'xygxy': 1, 'yyyyx': 1, 'yxyxg': 1,
    'yxgxy': 1})
    len(hints) >> 57

    """
    results = {}
    progress_bar = tqdm.tqdm(vocabulary)
    for guess in progress_bar:

        guess_idx = vocabulary.index(guess)
        hints = Counter(cached_hints[guess_idx])
        remaining = [len(filter_wordlist(target_set, guess, hint_num[hint])) for hint in hints]
        remaining_total = sum(val * count for val, count in zip(remaining, hints.values()))
        average_case, worst_case = remaining_total / hints.total(), max(remaining)
        num_groups, average_group_size = len(hints), sum(remaining) / len(hints)
        results[guess] = average_case, worst_case, num_groups, average_group_size

        progress_bar.set_description(f'{guess} {average_case:.2f} (max {worst_case}, groups {num_groups})')

    return results


if __name__ == '__main__':
    valid_words, solution_indexes, hint_to_index, hints = preload_words_and_hints()
    print(f'Wordle has {len(solution_indexes)} possible solutions')
    print(f'Wordle will accept {len(valid_words)} words')

    # average_number_of_guesses(solutions, solutions)

    # print_results(naive_best_starting_word(solutions, solutions))
    # print_results(naive_best_starting_word_faster(solutions, solutions, hints, hint_num))

    solutions = np.array(valid_words)[solution_indexes]
    print_results(naive_best_starting_word(solutions, solutions))
    """
        1 raise: 86.99 (max 182, groups 132 x ~28.83)
        2 irate: 88.33 (max 193, groups 124 x ~30.10)
        3 arise: 89.33 (max 182, groups 123 x ~30.87)
          ...
       33 crane: 108.11 (max 263, groups 142 x ~28.49)
          ...
     2307 mummy: 836.24 (max 1321, groups 37 x ~83.38)
     2308 vivid: 844.35 (max 1324, groups 45 x ~64.00)
     2309 fuzzy: 873.77 (max 1349, groups 34 x ~89.21)
        
    An interesting observation is that internet's darling "crane" only appears at the 33rd place, with an
    average of about 108 solutions remaining. It is supposed to be the best starting word, with an average
    expected reduction to about ~72 solutions. I'm puzzled by the discrepancy.
    """

    print_results(naive_best_starting_word(valid_words, solutions))
    """
        1 roate: 85.78 (max 194, groups 126 x ~29.93)
        2 raise: 86.99 (max 182, groups 132 x ~28.83)
        3 raile: 87.70 (max 195, groups 128 x ~29.62)
          ...
    12970 jujus: 917.02 (max 1349, groups 23 x ~114.04)
    12971 immix: 991.85 (max 1424, groups 34 x ~76.65)
    12972 qajaq: 996.73 (max 1366, groups 18 x ~155.72)
    """
