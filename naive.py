from pathlib import Path
from collections import Counter

import tqdm


def filter_wordlist(wordlist, guess, hint):
    """
    This function returns the list of wordlist's indexes that corresponds to words compatible with the guess/hint
    combination.

    Green hints are straighforward: that letter at that location

    Yellow and gray hints together give a lower/upper count for each letter.
    For example, if the same letter appears in one green and one yellow instances, it means that the word contains
    two or more of the letter. A same letter in one yellow and one gray means that the word contains one and only
    one count of the letter.

    For example, assume the word of the day is 'porch'.
    You guess 'power' and the hint is two green squares, two gray squares and a yellow one.
    We encode the hint as 'ggxxy', where 'g' is green, 'y' is yellow and 'x' is gray/not present.

    >>> from itertools import compress
    >>> with Path('solutions.txt').open() as txt_in:
    ...     solutions = sorted(s.strip().lower() for s in txt_in)  # 2309 words
    >>> print([solutions[index] for index in filter_wordlist(solutions, 'power', 'ggxxy')])
    ['polar', 'porch']

    >>> print([solutions[index] for index in filter_wordlist(solutions, 'array', 'ygxxy')])
    ['artsy', 'crazy', 'gravy']
    """
    filtered_wordlist = range(len(wordlist))
    for letter_index, (letter, status) in enumerate(zip(guess, hint)):
        if status != 'g':  # green hints: that letter at that location
            continue

        filtered_wordlist = [index for index in filtered_wordlist if wordlist[index][letter_index] == letter]

    for letter in set(guess):
        letter_hints = ''.join(h for (l, h) in zip(guess, hint) if l == letter)
        gs, ys, xs = (letter_hints.count(c) for c in 'gyx')
        min_bound, max_bound = gs + ys, gs + ys if xs else 6

        filtered_wordlist = [index for index in filtered_wordlist if min_bound <= wordlist[index].count(letter) <= max_bound]

    return filtered_wordlist


def guess_to_hint(guess, solution):
    """
    Return a hint for the given guess and solution.

    >>> print(guess_to_hint('array', 'trash'))
    ygxxx

    >>> print(guess_to_hint('cabal', 'coral'))
    gxxgg
    """
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
    for guess in tqdm.tqdm(vocabulary):
        hints = Counter(guess_to_hint(guess, solution) for solution in target_set)
        remaining = [len(filter_wordlist(target_set, guess, hint)) for hint in hints]
        remaining_total = sum(val * count for val, count in zip(remaining, hints.values()))
        average_case, worst_case = remaining_total / hints.total(), max(remaining)
        num_groups, average_group_size = len(hints), sum(remaining) / len(hints)
        results[guess] = average_case, worst_case, num_groups, average_group_size

    return results


def print_results(guesses, order_by=0, trim=3):
    """
    Guesses is a dictionary of the form {guess: (average_case, worst_case, num_groups, average_group_size)}
    This function prints results like the following, obtained for naive_best_starting_word(solutions, solutions):

        1 raise: 86.99 (max 182, groups 132 x ~28.83)
        2 irate: 88.33 (max 193, groups 124 x ~30.10)
        3 arise: 89.33 (max 182, groups 123 x ~30.87)
          ...
       33 crane: 108.11 (max 263, groups 142 x ~28.49)
          ...
     2307 mummy: 836.24 (max 1321, groups 37 x ~83.38)
     2308 vivid: 844.35 (max 1324, groups 45 x ~64.00)
     2309 fuzzy: 873.77 (max 1349, groups 34 x ~89.21)

    Internet's darling and agreed upon best starting guess "crane" appears at the 33rd place when naively
    counting the average remaining solutions. The following results are instead obtained for
    naive_best_starting_word(valid_words, solutions):

        1 roate: 85.78 (max 194, groups 126 x ~29.93)
        2 raise: 86.99 (max 182, groups 132 x ~28.83)
        3 raile: 87.70 (max 195, groups 128 x ~29.62)
          ...
    12970 jujus: 917.02 (max 1349, groups 23 x ~114.04)
    12971 immix: 991.85 (max 1424, groups 34 x ~76.65)
    12972 qajaq: 996.73 (max 1366, groups 18 x ~155.72)

    Solutions are the 2309 words that are and will be Wordle solutions.
    Valid words are the set of 12972 that are accepted as valid guesses.
    """
    print('\n')
    sorted_guesses = sorted(guesses.items(), key=lambda r: (r[1][order_by], r[1]))
    for index in list(range(trim)) + list(range(len(guesses)))[-trim:]:
        guess, (average, worst, groups, avg_group_size) = sorted_guesses[index]
        print(f'{index + 1:5} {guess}: {average:.2f} (max {worst}, groups {groups} x ~{avg_group_size:.2f})')
        if index == trim - 1:
            print('      ...')


def load_wordlists():
    """
    List of Wordle solutions downloaded from:
    static.nytimes.com/newsgraphics/2022/01/25/wordle-solver/assets/solutions.txt

    List of valid 5-letter guesses from:
    gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93
    """
    with Path('valid.txt').open() as txt_in:
        valid_words = sorted(s.strip().lower() for s in txt_in)  # 12972 words, INCLUDING solutions

    with Path('solutions.txt').open() as txt_in:
        solutions = sorted(s.strip().lower() for s in txt_in)  # 2309 words

    return valid_words, solutions


if __name__ == '__main__':
    valid_words, solutions = load_wordlists()
    print(f'Wordle has {len(solutions)} possible solutions')
    print(f'Wordle will accept {len(valid_words)} words')

    print_results(naive_best_starting_word(solutions, solutions))
    print_results(naive_best_starting_word(valid_words, solutions))
