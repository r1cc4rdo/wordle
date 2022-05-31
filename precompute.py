from pathlib import Path
from itertools import product, count

import tqdm
import numpy as np

from naive import filter_wordlist, load_wordlists


def hint_strings_builder():
    """
    Return a list of all possible 243 hints, from 'xxxxx' to 'ggggg'.
    243 == 3**5, three possible letters for of the five slots.
    """
    return list(map(lambda t: ''.join(t), product('xyg', repeat=5)))


def precompute_filtered_wordlists(source, target, filtered_path=None):
    """
    This takes ~1h20 on a Macbook 2015, for the full 12972 x 243 list.
    Each entry is a list of indexes into target, corresponding to the
    compatible words for the given guess/hint combination.
    """
    filtered_path = filtered_path or Path(f'filtered_{len(source)}_{len(target)}.npz')

    if filtered_path.exists():
        return np.load(filtered_path, allow_pickle=True)['arr_0']

    hint_strings = hint_strings_builder()
    filtered_wordlists = [[filter_wordlist(target, guess, hint)
                           for hint in hint_strings]
                          for guess in tqdm.tqdm(source)]

    np.savez_compressed(filtered_path, np.array(filtered_wordlists, dtype=object))
    return filtered_wordlists


def precompute_guesses_to_hints(source, target, hints_path=None):
    """
    Precompute the resulting hint for every pair of guess and solution.
    Returns a matrix of shape NxN where N is the length of vocabulary,
    containing the uint8 index into the list from hint_strings_builder().
    The implementation here is equivalent to the following, but ~12x faster:

    from naive import guess_to_hint
    guesses_to_hints = np.empty((len(valid_words), len(valid_words)), dtype=np.uint8)
    for guess_index, guess in enumerate(tqdm.tqdm(valid_words)):
        for sol_index, solution in enumerate(valid_words):
            guesses_to_hints[guess_index, sol_index] = hint_to_index[guess_to_hint(guess, solution)]

    On a Macbook 2015, this takes ~6 minutes for the full 12972 x 12972 matrix.
    """
    hints_path = hints_path or Path(f'hints_{len(source)}_{len(target)}.npz')

    if hints_path.exists():
        return np.load(hints_path, allow_pickle=True)['arr_0']

    chars = np.array([list(s) for s in target])
    hint_to_index = dict(zip(hint_strings_builder(), count()))
    guesses_to_hints = np.empty((len(source), len(target)), dtype=np.uint8)
    for guess_index, guess in enumerate(tqdm.tqdm(source)):

        guess_to_hints = np.full_like(chars, 'x')
        for index, letter in enumerate(guess):
            guess_to_hints[chars[:, index] == letter, index] = 'g'

        for index, letter in enumerate(guess):
            is_green = guess_to_hints[:, index] == 'g'
            letter_in_solution = np.count_nonzero(chars == letter, axis=1)
            letter_in_guess = np.array(list(guess)) == letter
            already_assigned = np.count_nonzero(guess_to_hints[:, letter_in_guess] != 'x', axis=1)
            has_capacity = letter_in_solution - already_assigned > 0
            guess_to_hints[has_capacity & ~is_green, index] = 'y'

        guesses_to_hints[guess_index] = [hint_to_index[''.join(hint)] for hint in guess_to_hints]

    np.savez_compressed(hints_path, guesses_to_hints)
    return guesses_to_hints


if __name__ == '__main__':
    valid_words, solutions = load_wordlists()
    precompute_guesses_to_hints(valid_words, valid_words)
    precompute_filtered_wordlists(valid_words, valid_words)
    precompute_guesses_to_hints(valid_words, solutions)
    precompute_filtered_wordlists(valid_words, solutions)
    precompute_guesses_to_hints(solutions, solutions)
    precompute_filtered_wordlists(solutions, solutions)

    # TODO: test make smaller from the large versions
