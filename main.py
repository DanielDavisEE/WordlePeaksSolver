import cProfile
import os
import pstats
import re
import abc

import numpy as np
import pandas as pd


class SolverBase(abc.ABC):
    source_folder = None

    def __init__(self, /, limit=None, verbose=False):

        self.verbose = verbose

        with open(os.path.join(self.source_folder, 'targets_dictionary.txt'), 'r', encoding='utf-8') as infile:
            self.target_words = np.array(infile.read().splitlines()[:limit], dtype='U5')

        with open(os.path.join(self.source_folder, 'full_dictionary.txt'), 'r', encoding='utf-8') as infile:
            self.all_words = np.array(infile.read().splitlines(), dtype='U5')

        self.start_word, self.start_info = None, None

        self.init()

    def init(self):
        self.update_ranges()
        self.start_word, self.start_info = self.get_best_word()
        if self.verbose:
            print("Finished init")

    @abc.abstractmethod
    def play(self):
        best_word, info = self.start_word, self.start_info
        while (hints := self.get_clean_input(best_word, info)) != 'quit':
            self.update_ranges(best_word, hints)
            best_word, info = self.get_best_word()
            if info['only_word']:
                print(f"The answer is {best_word}!\n")
                self.update_ranges()
                best_word, info = self.start_word, self.start_info

    def test_word(self, answer=None):
        verbose = False
        if answer is None:
            verbose = True
            answer = input('The final answer was: ')

        guesses = 1
        best_word = self.start_word
        self.update_ranges()
        while best_word != answer:
            if verbose:
                print(f'Guess {guesses}: {best_word}')
            hints = self.simulate_hints(best_word, answer)
            self.update_ranges(best_word, hints)
            best_word, _ = self.get_best_word()
            guesses += 1
        if verbose:
            print(f'The answer is: {best_word}')
        return guesses

    def test(self):
        results = pd.Series(None, index=self.target_words, dtype='Int64')
        for i, answer in enumerate(self.target_words):
            results[answer] = self.test_word(answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.target_words)):
                print(f'Tested {i:>5}/{len(self.target_words)}')

        return results

    def get_clean_input(self, best_word, info):
        def is_clean(input_string):
            return len(input_string) == 5 and all(c in self.HINT_TYPES for c in input_string)

        user_input = input(
            f"{info['target_words_left']:>4}|{info['words_left']:>5}: Use '{best_word}' and enter hints: ").lower()
        while not is_clean(user_input) and user_input != 'quit':
            user_input = input(f"Enter a 5 letter combination of {self.HINT_TYPES[0]},"
                               f" {self.HINT_TYPES[1]} and {self.HINT_TYPES[2]}: ").lower()

        return user_input

    @abc.abstractmethod
    def simulate_hints(self, guess: str, answer: str):
        raise NotImplementedError

    @abc.abstractmethod
    def target_words_left(self):
        raise NotImplementedError

    @abc.abstractmethod
    def words_left(self):
        raise NotImplementedError

    def get_best_word(self):
        available_targets = self.target_words_left()
        info_dict = {
            'only_word': True,
            'words_left': -1,
            'target_words_left': len(available_targets)
        }
        if len(available_targets) == 1:
            return available_targets[0], info_dict

        available_words = self.words_left()
        info_dict['only_word'] = False
        info_dict['words_left'] = len(available_words)
        best_word = self._choose_best_word(available_words, available_targets)
        return best_word, info_dict

    @abc.abstractmethod
    def _choose_best_word(self, available_words, available_targets):
        raise NotImplementedError


class SolverMinTargets(abc.ABC):

    def __init__(self, **kwargs):
        self.word_matrix = None
        super().__init__(**kwargs)

    def init(self):
        self.word_matrix = self.create_word_matrix()
        super().init()

    @abc.abstractmethod
    def create_word_matrix(self):
        raise NotImplementedError

    def simulate_hints(self, guess, answer):
        return self.word_matrix.loc[guess, answer]

    def _choose_best_word(self, available_words, available_targets):
        reduced_word_matrix = self.word_matrix.loc[available_words, available_targets]
        worst_case_words = (pd.melt(reduced_word_matrix, var_name='answer', value_name='hint', ignore_index=False)
                            .drop('answer', axis=1)
                            .reset_index()
                            .value_counts(sort=False)
                            .groupby(level='index', sort=False)
                            .max())

        return worst_case_words.idxmin()


class PeaksSolverBase(SolverBase):
    source_folder = 'peaks'

    PEAKS_URL = 'https://vegeta897.github.io/wordle-peaks'

    BEFORE = 'y'
    AFTER = 'b'
    CORRECT = 'g'

    HINT_TYPES = (BEFORE, AFTER, CORRECT)

    def __init__(self, **kwargs):
        self.letter_ranges = None
        super().__init__(**kwargs)

    def init(self):
        self.update_ranges()
        super().init()

    def play(self):
        print("Hints:",
              f"{self.AFTER} = after (blue)",
              f"{self.BEFORE} = before (yellow)",
              f"{self.CORRECT} = correct (green)",
              f"e.g. {self.AFTER}{self.BEFORE}{self.CORRECT}{self.BEFORE}{self.CORRECT}",
              "",
              "Input 'quit' to end.\n", sep='\n')

        super().play()

    def simulate_hints(self, guess: str, answer: str):
        split_guess = np.array([guess]).view('U1').reshape((1, -1))
        split_answer = np.array([answer]).view('U1').reshape((1, -1))

        hints_temp = np.empty_like(split_guess)
        hints_temp[split_guess < split_answer] = self.BEFORE
        hints_temp[split_guess == split_answer] = self.CORRECT
        hints_temp[split_guess > split_answer] = self.AFTER

        return hints_temp.view('U5').reshape((-1))[0]

    def update_ranges(self, word=None, hints=None):
        if word is None:
            self.letter_ranges = np.array([
                ['a'] * 5,
                ['z'] * 5
            ]).view('U1').reshape((2, 5))
        else:
            split_word = np.array([word]).view('U1')
            split_hints = np.array([hints]).view('U1')

            self.letter_ranges[0, split_hints == self.BEFORE] = (split_word[split_hints == self.BEFORE]
                                                                 .view('int') + 1).view('U1')
            self.letter_ranges[:, split_hints == self.CORRECT] = split_word[split_hints == self.CORRECT]
            self.letter_ranges[1, split_hints == self.AFTER] = (split_word[split_hints == self.AFTER]
                                                                .view('int') - 1).view('U1')

    def target_words_left(self):
        split_guesses = self.target_words.view('U1').reshape((len(self.target_words), -1))
        mask = np.logical_and((self.letter_ranges[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.letter_ranges[1]).all(axis=1))
        return self.target_words[mask]

    def words_left(self):
        split_guesses = self.all_words.view('U1').reshape((len(self.all_words), -1))
        mask = np.logical_and((self.letter_ranges[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.letter_ranges[1]).all(axis=1))
        return self.all_words[mask]

    def get_best_word(self):
        available_targets = self.target_words_left()
        info_dict = {
            'only_word': True,
            'words_left': -1,
            'target_words_left': len(available_targets)
        }
        if len(available_targets) == 1:
            return available_targets[0], info_dict

        available_words = self.words_left()
        info_dict['only_word'] = False
        info_dict['words_left'] = len(available_words)
        best_word = self._choose_best_word(available_words, available_targets)
        return best_word, info_dict

    def _choose_best_word(self, available_words, available_targets):
        return available_targets[0]


class PeaksSolverMinTargets(SolverMinTargets, PeaksSolverBase):
    """
    Select the word which minimises the largest number of remaining target words
    """

    def create_word_matrix(self):
        split_guesses = self.all_words.view('U1').reshape((len(self.all_words), -1))

        hints_dict = {}
        for i, answer in enumerate(self.target_words):
            split_answer = np.array([answer]).view('U1')

            hints_temp = np.empty_like(split_guesses)
            hints_temp[split_guesses < split_answer] = self.BEFORE
            hints_temp[split_guesses == split_answer] = self.CORRECT
            hints_temp[split_guesses > split_answer] = self.AFTER

            hints_dict[answer] = hints_temp.view('U5').reshape((-1))

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.target_words)):
                print(f'Matrix column {i:>5}/{len(self.target_words)}')

        return pd.DataFrame(hints_dict, index=self.all_words)


def main():
    question = ("Type 'play' to have suggestions given to you or 'test' to see how the solver would solve a word.\n"
                "Type 'quit' to end the program: ")
    while (choice := input(question).lower()) not in ['q', 'quit']:
        try:
            {'play': solver.play,
             'test': solver.test_word}[choice]()
        except KeyError:
            continue


if __name__ == '__main__':
    TEST = True
    if TEST:
        profiler = cProfile.Profile(subcalls=False, builtins=False)
        profiler.enable()

    solver = PeaksSolverBase(verbose=True)
    if TEST:
        results = solver.test()
        average_score = results.mean()
        distribution = results.value_counts().sort_index()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats('WordlePeaksSolver.main.py', 20)

        print(f"The average score for PeaksSolverMinTargets is {average_score:.4f}")
        print(distribution)
        print(results.loc[results == distribution.index.max()].index.to_list())
    else:
        main()
