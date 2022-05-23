from pathlib import Path
from itertools import product
from collections import Counter, defaultdict

import tqdm
import numpy as np


def filter_wordlist(wordlist, guess, hint):
    """
    This function will filter a wordlist based on the hint constraints.

    For example, assume the word of the day is 'porch'.
    You guess 'power' and the hint is two green squares, two gray squares and a yellow one.
    We encode the hint as 'ggxxy', where 'g' is green, 'y' is yellow and 'x' is gray/not present.

    >>> print(filter_wordlist(solutions, 'power', 'ggxxy'))
    ['polar', 'porch']

    >>> print(filter_wordlist(solutions, 'array', 'ygxxy'))
    ['artsy', 'crazy', 'gravy']
    """
    guess, hint = guess.lower(), hint.lower()
    filtered_wordlist = wordlist[:]

    # Green hints are easy: that letter at that location
    for index, (letter, status) in enumerate(zip(guess, hint)):
        if status == 'g':  # 'list' forces evaluation because lambdas only captures parameters
            filtered_wordlist = list(filter(lambda word: word[index] == letter, filtered_wordlist))

    # Yellow and gray hints together give an lower/upper count for each letter.
    # For example, if the same letter appears in one green and one yellow instances, it means that the word contains
    # two or more of the letter. A same letter in one yellow and one gray means that the word contains one and only
    # one count of the letter.
    for letter in set(guess):
        letter_hints = ''.join(h for (l, h) in zip(guess, hint) if l == letter)
        gs, ys, xs = (letter_hints.count(c) for c in 'gyx')
        min_bound, max_bound = gs + ys, gs + ys if xs else 6
        filtered_wordlist = list(filter(lambda word: min_bound <= word.count(letter) <= max_bound, filtered_wordlist))

    return list(filtered_wordlist)


def guess_to_hint(guess, solution):
    """
    Return a hint for the given guess and solution.

    >>> print(guess_to_hint('array', 'trash'))
    ygxxx

    >>> print(guess_to_hint('cabal', 'coral'))
    gxxgg
    """
    guess, solution = guess.lower(), solution.lower()
    letter_counts = Counter(solution)

    hint = ['_'] * 5  # green hints need to be done first, since they take priority over yellow and gray
    for index, (guess_letter, solution_letter) in enumerate(zip(guess, solution)):
        if guess_letter == solution_letter:
            letter_counts[solution_letter] -= 1
            hint[index] = 'g'

    for index, (guess_letter, hint_letter) in enumerate(zip(guess, hint)):
        if hint_letter == 'g':
            continue

        hint[index] = 'y' if letter_counts[guess_letter] > 0 else 'x'
        letter_counts[guess_letter] -= 1

    return ''.join(hint)


def naive_best_starting_word(vocabulary, target_set):
    """
    We call this naive because it only considers the average number of remaining solutions after one guess.
    A non-naive solution would minimize instead the average number of required guesses.
    """
    results = {}
    progress_bar = tqdm.tqdm(vocabulary)
    for guess in progress_bar:

        hints = Counter(guess_to_hint(guess, solution) for solution in target_set)
        remaining = [len(filter_wordlist(target_set, guess, hint)) for hint in hints]
        remaining_total = sum(val * count for val, count in zip(remaining, hints.values()))
        average_case, worst_case = remaining_total / hints.total(), max(remaining)
        num_groups, average_group_size = len(hints), sum(remaining) / len(hints)
        results[guess] = average_case, worst_case, num_groups, average_group_size

        progress_bar.set_description(f'{guess} {average_case:.2f} (max {worst_case}, groups {num_groups})')

    return results


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


def preload_words_and_hints(cache_path=Path('cache.npz')):
    """
    Returns valid guesses, indexes of the solutions' subset within guesses' list, hint <> index dictionary,
    and precomputed hints for any possible valid word pair (as a flattened list of above diagonal elements).

    Both valid words and solutions are loaded from disk, respectively from 'valid.txt' and 'solutions.txt'.
    They are returned alphabetically sorted, in the order used to precomputed hints, described later.
    They are originally from: static.nytimes.com/newsgraphics/2022/01/25/wordle-solver/assets/solutions.txt
    and gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93

    Hints are precomputed in the form of a full matrix indexed by the sorted indexes of valid words and solutions
    respectively. Each entry is itself an index to the actual hint string, which can be decoded with the returned
    hint and indexes dictionary. The dictionary works both ways, from indexes to hints and viceversa.

    Hints are provided as 5 letter strings composed by the 'x', 'y' and 'g' characters. They stand respectively
    for missing letter, yellow and green.
    """
    with Path('valid.txt').open() as txt_in:
        valid_words = sorted(s.strip() for s in txt_in)  # 12972 words, INCLUDING solutions

    hint_num = dict(enumerate(map(lambda t: ''.join(t), product('xyg', repeat=5))))
    hint_num = hint_num | {h: i for i, h in hint_num.items()}  # hint to index, index to hint

    def inflate(above_diag, dim):
        """
        Rehydrates a flattened list of above diagonal elements into a symmetric matrix.
        Diagonal elements, not saved, are set to the index of the 'ggggg' (aka 'correct') hint.
        """
        inflated = np.zeros((dim, dim), dtype=above_diag.dtype)
        inflated[np.triu_indices(dim, 1)] = above_diag
        inflated += np.transpose(inflated)
        np.fill_diagonal(inflated, hint_num['ggggg'])
        return inflated

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            sol_indexes, hints_triu = data['sol_indexes'], data['hints_triu']
            return valid_words, list(sol_indexes), hint_num, inflate(hints_triu, len(valid_words))

    with Path('solutions.txt').open() as txt_in:
        solutions = sorted(s.strip() for s in txt_in)  # 2309 words
    sol_indexes = [valid_words.index(solution) for solution in solutions]

    hints_triu = np.empty((len(valid_words) ** 2 - len(valid_words)) // 2, dtype=np.uint8)  # above diagonal, flat
    progress_bar = tqdm.tqdm(zip(*np.triu_indices(len(valid_words), 1)), total=len(hints_triu))
    for triu_index, (guess_index, sol_index) in enumerate(progress_bar):
        guess, solution = valid_words[guess_index], valid_words[sol_index]
        hints_triu[triu_index] = hint_num[guess_to_hint(guess, solution)]
        progress_bar.set_description(f'{guess} / {solution}')

    np.savez_compressed(cache_path, {'sol_indexes': sol_indexes, 'hints_triu': hints_triu})
    return valid_words, sol_indexes, hint_num, inflate(hints_triu, len(valid_words))


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
    valid_words, sol_indexes, hint_num, hints = preload_words_and_hints()
    print(f'Wordle has {len(sol_indexes)} possible solutions')
    print(f'Wordle will accept {len(valid_words)} words')

    def print_results(guesses, order_by=0, trim=3):
        print('\n')
        sorted_guesses = sorted(guesses.items(), key=lambda r: (r[1][order_by], r[1]))
        for index in list(range(trim)) + list(range(len(guesses)))[-trim:]:
            guess, (average, worst, groups, avg_group_size) = sorted_guesses[index]
            print(f'{index + 1:5} {guess}: {average:.2f} (max {worst}, groups {groups} x ~{avg_group_size:.2f})')
            if index == trim - 1:
                print('      ...')

    # average_number_of_guesses(solutions, solutions)
    # average_number_of_guesses(solutions, solutions)

    # TODO: create reduced version of cached
    # TODO: restore faster version with enumerate
    # TODO: verify if remaining counts can be obtained from cached
    # TODO: precompute vocabulary compatible with hints
    # TODO: compute triangular and mirror. ggggg on diagonal

    # print_results(naive_best_starting_word(solutions, solutions))
    print_results(naive_best_starting_word_faster(solutions, solutions, hints, hint_num))

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
