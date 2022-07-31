import cProfile
import json
import pstats
import re
from itertools import product

import numpy as np
import pandas as pd

LETTERS = [chr(x + ord('a')) for x in range(26)]
BEFORE = 'b'
AFTER = 'a'
CORRECT = 'c'
RESULT_GROUPS = {''.join(option): i for option, i in zip(product('abc', repeat=5), range(3 ** 5))}

PEAKS_URL = 'https://vegeta897.github.io/wordle-peaks'


class PeaksSolverBase:
    def __init__(self, limit=None):

        with open('targets_dictionary.json', 'r', encoding='utf-8') as infile:
            self.target_words_list = json.load(infile)[:limit]

        with open('full_dictionary.json', 'r', encoding='utf-8') as infile:
            self.all_words_list = json.load(infile)

        self.target_words = pd.Series(self.target_words_list)
        self.all_words = pd.Series(self.all_words_list)

        self.letter_ranges = None

        self.word_matrix = self.create_word_matrix()

        self.update_ranges()
        self.start_word, _ = self.get_best_word()
        print("Finished init")

    def create_word_matrix(self):
        return None

    def play(self):
        print("Hints:",
              "a = after",
              "b = before",
              "c = correct",
              "e.g. abcba",
              "",
              "Input 'quit' to end.\n", sep='\n')

        best_word, info = self.get_best_word()
        while (hints := self.get_clean_input(best_word, info)) != 'quit':
            self.update_ranges(best_word, hints)
            best_word, info = self.get_best_word()
            if info['only_word']:
                print(f"The answer is {best_word}!\n")
                self.update_ranges()
                best_word, info = self.get_best_word()

    def test(self):
        results = pd.Series(None, index=self.target_words_list, dtype='Int64')
        for i, answer in enumerate(self.target_words_list):
            guesses = 1
            best_word = self.start_word
            self.update_ranges()
            while best_word != answer:
                hints = self.simulate_hints(best_word, answer)
                self.update_ranges(best_word, hints)
                best_word, _ = self.get_best_word()
                guesses += 1
            results[answer] = guesses

            if re.fullmatch(r'0b10*', bin(i)):
                print(f'Tested {i:>5}/{len(self.target_words_list)}')

        average_score = results.mean()
        distribution = results.value_counts().sort_index()
        print(f"The average score for {type(self).__name__} is {average_score:.4f}")
        print(distribution)

    @staticmethod
    def get_clean_input(best_word, info):
        def is_clean(input_string):
            return len(input_string) == 5 and all(c in list('abc') for c in input_string)

        user_input = input(
            f"{info['target_words_left']:>4}|{info['words_left']:>5}: Use '{best_word}' and enter hints: ").lower()
        while not is_clean(user_input) and user_input != 'quit':
            user_input = input("Enter a 5 letter combination of a, b and c: ").lower()

        return user_input

    @staticmethod
    def range_to_regex(ranges):
        pattern = ''
        for start, end in ranges:
            letters = ''.join(LETTERS[slice(start, end + 1)])
            pattern += f'[{letters}]'
        return re.compile(pattern)

    @staticmethod
    def simulate_hints(guess, answer):
        split_guess = np.array([guess]).view('U1').reshape((1, -1))
        split_answer = np.array([answer]).view('U1').reshape((1, -1))

        hints_temp = np.empty_like(split_guess)
        hints_temp[split_guess < split_answer] = BEFORE
        hints_temp[split_guess == split_answer] = CORRECT
        hints_temp[split_guess > split_answer] = AFTER

        return hints_temp.view('U5').reshape((-1))

    def update_ranges(self, word=None, hints=None):
        if word is None:
            self.letter_ranges = [(0, 25)] * 5
        else:
            # ranges = np.array(['a', 'z'] * 5).view('U1').reshape((5, 2))
            # ranges[0, 0] = (ranges[0, 0].view('int') + 1).view('U1')

            hints = list(hints)
            letters = [ord(c) - ord('a') for c in word]
            for i in range(5):
                hint, letter = hints[i], letters[i]
                start, end = self.letter_ranges[i]
                if hint == BEFORE:
                    start = letter + 1
                elif hint == AFTER:
                    end = letter - 1
                elif hint == CORRECT:
                    end = start = letter
                else:
                    raise ValueError(f"Hint = '{hint}'????")
                self.letter_ranges[i] = start, end

    def target_words_left(self, pattern):
        return self.target_words[self.target_words.str.fullmatch(pattern)].reset_index(drop=True)

    def words_left(self, pattern):
        return self.all_words[self.all_words.str.fullmatch(pattern)].reset_index(drop=True)

    def get_best_word(self):
        word_pattern = self.range_to_regex(self.letter_ranges)
        available_targets = self.target_words_left(word_pattern)
        info_dict = {
            'only_word': True,
            'words_left': -1,
            'target_words_left': len(available_targets)
        }
        if len(available_targets) == 1:
            return available_targets[0], info_dict

        available_words = self.words_left(word_pattern)
        info_dict['only_word'] = False
        info_dict['words_left'] = len(available_words)
        best_word = self._choose_best_word(word_pattern, available_words, available_targets)
        return best_word, info_dict

    def _choose_best_word(self, word_pattern, available_words, available_targets):
        return available_targets[0]


class PeaksSolverMinTargets(PeaksSolverBase):
    """
    Select the word which minimises the largest number of remaining target words
    """

    def create_word_matrix(self):
        split_guesses = np.array(self.all_words_list).view('U1').reshape((len(self.all_words_list), -1))

        hints_dict = {}
        for i, answer in enumerate(self.target_words_list):
            split_answer = np.array([answer]).view('U1')

            hints_temp = np.empty_like(split_guesses)
            hints_temp[split_guesses < split_answer] = BEFORE
            hints_temp[split_guesses == split_answer] = CORRECT
            hints_temp[split_guesses > split_answer] = AFTER

            hints_dict[answer] = hints_temp.view('U5').reshape((-1))

            if re.fullmatch(r'0b10*', bin(i)):
                print(f'Matrix column {i:>5}/{len(self.target_words_list)}')

        return pd.DataFrame(hints_dict, index=self.all_words_list)

    def simulate_hints(self, guess, answer):
        return self.word_matrix.loc[guess, answer]

    def _choose_best_word(self, word_pattern, available_words, available_targets):
        reduced_word_matrix = self.word_matrix.loc[available_words, available_targets]
        worst_case_words = (pd.melt(reduced_word_matrix, var_name='answer', value_name='hint', ignore_index=False)
                            .drop('answer', axis=1)
                            .reset_index()
                            .value_counts(sort=False)
                            .groupby(level='index', sort=False)
                            .max())

        return worst_case_words.idxmin()


if __name__ == '__main__':
    profiler = cProfile.Profile(subcalls=False, builtins=False)
    profiler.enable()

    solver = PeaksSolverMinTargets()
    solver.test()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
