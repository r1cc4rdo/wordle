# Wordle stats
This code measures the expected number of surviving solutions for a set vocabulary.

These are the results when using guessing one of the 2309 [Wordle](https://www.nytimes.com/games/wordle) solutions, and using only guesses in the same list.
```
    1 raise: 86.99 (max 182, groups 132 x ~28.83)
    2 irate: 88.33 (max 193, groups 124 x ~30.10)
    3 arise: 89.33 (max 182, groups 123 x ~30.87)
      ...
   33 crane: 108.11 (max 263, groups 142 x ~28.49)
      ...
 2307 mummy: 836.24 (max 1321, groups 37 x ~83.38)
 2308 vivid: 844.35 (max 1324, groups 45 x ~64.00)
 2309 fuzzy: 873.77 (max 1349, groups 34 x ~89.21)
```
These results are näive in the sense that the ranking is taken after a single guess. This correlates both with information gain and expected number of required guesses, but does not produce the same ordering.

An interesting observation is that internet's darling "crane" only appears at the 33rd place, with an average of about 108 solutions remaining. It is supposed to be the best starting word, with an average expected reduction to about ~72 solutions.

## Wordlists
* [solutions.txt](https://github.com/r1cc4rdo/wordle/blob/main/solutions.txt): Wordle solution set, 2309 words. Dowloaded from the [NY Times website](https://static.nytimes.com/newsgraphics/2022/01/25/wordle-solver/assets/solutions.txt)
* [valid.txt](https://github.com/r1cc4rdo/wordle/blob/main/solutions.txt): all 12972 words accepted as guesses by Wordle, INCLUDING solutions. Downloaded from [this Draco's gist](https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93)

## TODO
* use the [```naive_best_starting_word```](https://github.com/r1cc4rdo/wordle/blob/a12ff6c61abb93ff3dbc9e6e82f02375a99429eb/wordle.py#L71) recursively to build a ```best_starting_word``` function that minimizes the expected number of guesses.
