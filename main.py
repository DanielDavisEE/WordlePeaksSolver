import os
import re
import abc
import string
import pstats
import cProfile
import time

import numpy as np
import pandas as pd


class SolverBase(abc.ABC):
    source_folder = None
    start_word, start_info = None, None
    game_state = None

    def __init__(self, /, limit=None, verbose=False):

        self.verbose = verbose

        with open(os.path.join(self.source_folder, 'targets_dictionary.txt'), 'r', encoding='utf-8') as infile:
            self.target_words = np.array(infile.read().splitlines()[:limit], dtype='U5')

        with open(os.path.join(self.source_folder, 'full_dictionary.txt'), 'r', encoding='utf-8') as infile:
            self.all_words = np.array(infile.read().splitlines(), dtype='U5')

        self.init()

    def init(self):
        self.reset_state()
        self.start_word, self.start_info = self.get_best_word()
        if self.verbose:
            print("Finished init")

    @abc.abstractmethod
    def play(self):
        best_word, info = self.start_word, self.start_info
        while (hints := self.get_clean_input(best_word, info)) != 'quit':
            self.update_state(best_word, hints)
            best_word, info = self.get_best_word()
            if info['only_word']:
                print(f"The answer is {best_word}!\n")
                self.reset_state()
                best_word, info = self.start_word, self.start_info

    def test_word(self, answer=None):
        verbose = False
        if answer is None:
            verbose = True
            answer = input('The final answer was: ')

        guesses = 1
        best_word = self.start_word
        self.reset_state()
        while best_word != answer:
            if verbose:
                print(f'Guess {guesses}: {best_word}')
            hints = self.simulate_hints(best_word, answer)
            self.update_state(best_word, hints)
            best_word, _ = self.get_best_word()
            guesses += 1
        if verbose:
            print(f'The answer is: {best_word}')
        return guesses

    def test(self):
        results = pd.Series(None, index=self.target_words, dtype='Int64')
        for i, answer in enumerate(self.target_words, start=1):
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
    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        raise NotImplementedError

    @abc.abstractmethod
    def reset_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_state(self, word, hints):
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

    def _choose_best_word(self, available_words, available_targets):
        return available_targets[0]


class SolverMinTargets(SolverBase):

    def __init__(self, **kwargs):
        self.word_matrix = None
        super().__init__(**kwargs)

    def init(self):
        self.word_matrix = self.create_word_matrix()
        super().init()

    def create_word_matrix(self):
        hints_dict = {}
        for i, answer in enumerate(self.target_words, start=1):
            hints_dict[answer] = self.simulate_hints(self.all_words, answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.target_words)):
                print(f'Matrix column {i:>5}/{len(self.target_words)}')

        return pd.DataFrame(hints_dict, index=self.all_words)

    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        if self.word_matrix is None:
            return super().simulate_hints(guesses, answer)
        else:
            return self.word_matrix.loc[guesses, answer]

    def _choose_best_word(self, available_words, available_targets):
        reduced_word_matrix = self.word_matrix.loc[available_words, available_targets]
        word_groups = (pd.melt(reduced_word_matrix, var_name='answer', value_name='hint', ignore_index=False)
                       .drop('answer', axis=1)
                       .reset_index()
                       .value_counts(sort=False))
        max_group = (word_groups
                     .groupby(level='index', sort=False)
                     .max())
        best_words = max_group.index[max_group == max_group.min()]

        best_words_max_groups = pd.DataFrame(
            word_groups[best_words]
                .groupby(level='index', sort=False)
                .nlargest(n=5)
                .droplevel([1, 2], axis=0))
        best_words_max_groups.rename({best_words_max_groups.columns[0]: 'counts'}, axis=1, inplace=True)

        best_words_max_groups['rank'] = (best_words_max_groups
                                         .groupby(by='index')
                                         .rank('first', ascending=False))

        best_words_max_groups = (best_words_max_groups
            .pivot(columns='rank', values='counts')
            .sort_values(
            by=[float(x) for x in range(1, int(best_words_max_groups['rank'].max()) + 1)],
            ascending=True))

        return best_words_max_groups.index[0]


class PeaksSolverBase(SolverBase):
    source_folder = 'peaks'

    PEAKS_URL = 'https://vegeta897.github.io/wordle-peaks'

    BEFORE = 'y'
    AFTER = 'b'
    CORRECT = 'g'

    HINT_TYPES = (BEFORE, AFTER, CORRECT)

    def play(self):
        print("Hints:",
              f"{self.AFTER} = after (blue)",
              f"{self.BEFORE} = before (yellow)",
              f"{self.CORRECT} = correct (green)",
              f"e.g. {self.AFTER}{self.BEFORE}{self.CORRECT}{self.BEFORE}{self.CORRECT}",
              "",
              "Input 'quit' to end.\n", sep='\n')

        super().play()

    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        split_guess = np.array([guesses]).view('U1').reshape((-1, 5))
        split_answer = np.array([answer]).view('U1').reshape((-1, 5))

        hints = np.empty_like(split_guess)
        hints[split_guess < split_answer] = self.BEFORE
        hints[split_guess == split_answer] = self.CORRECT
        hints[split_guess > split_answer] = self.AFTER

        hints = hints.view('U5').reshape((-1))
        if isinstance(guesses, str):
            hints = hints[0]
        return hints

    def reset_state(self):
        self.game_state = np.array([
            ['a'] * 5,
            ['z'] * 5
        ])

    def update_state(self, word, hints):
        split_word = np.array([word]).view('U1')
        split_hints = np.array([hints]).view('U1')

        tmp = np.array([
            ['a'] * 5,
            ['z'] * 5
        ])

        tmp[0, split_hints == self.BEFORE] = (split_word[split_hints == self.BEFORE]
                                              .view('int') + 1).view('U1')
        tmp[:, split_hints == self.CORRECT] = split_word[split_hints == self.CORRECT]
        tmp[1, split_hints == self.AFTER] = (split_word[split_hints == self.AFTER]
                                             .view('int') - 1).view('U1')

        self.game_state[0, :] = np.maximum(self.game_state.view('int')[0, :], tmp.view('int')[0, :]).view('U1')
        self.game_state[1, :] = np.minimum(self.game_state.view('int')[1, :], tmp.view('int')[1, :]).view('U1')

    def target_words_left(self):
        split_guesses = self.target_words.view('U1').reshape((len(self.target_words), -1))
        mask = np.logical_and((self.game_state[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.game_state[1]).all(axis=1))
        return self.target_words[mask]

    def words_left(self):
        split_guesses = self.all_words.view('U1').reshape((len(self.all_words), -1))
        mask = np.logical_and((self.game_state[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.game_state[1]).all(axis=1))
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


class PeaksSolverMinTargets(SolverMinTargets, PeaksSolverBase):
    """
    Select the word which minimises the largest number of remaining target words
    """


class WordleSolverBase(SolverBase):
    source_folder = 'wordle'

    INCORRECT = 'b'
    PARTIAL = 'y'
    CORRECT = 'g'

    HINT_TYPES = (INCORRECT, PARTIAL, CORRECT)

    letter_counts = None

    def init(self):
        split_words = self.all_words.view('U1').reshape((len(self.all_words), -1))
        self.letter_counts = (pd.DataFrame(split_words, index=self.all_words)
                              .melt(value_name='letter', ignore_index=False)
                              .drop('variable', axis=1)
                              .groupby(level=0, sort=False)
                              .value_counts(sort=False)
                              .reset_index(level='letter')
                              .rename(columns={0: 'counts'}, index={0: 'word'}))
        super().init()

    def play(self):
        print("Hints:",
              f"{self.INCORRECT} = incorrect (black)",
              f"{self.PARTIAL} = in word (yellow)",
              f"{self.CORRECT} = correct (green)",
              f"e.g. {self.INCORRECT}{self.PARTIAL}{self.CORRECT}{self.PARTIAL}{self.CORRECT}",
              "",
              "Input 'quit' to end.\n", sep='\n')

        super().play()

    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        GUESS_PLACEHOLDER = ','
        ANSWER_PLACEHOLDER = '.'

        split_guess = np.array([guesses]).view('U1').reshape(-1)
        split_answer = np.array([answer]).view('U1').reshape(-1)

        hints_temp = np.full_like(split_guess, self.INCORRECT)

        # Mark correct letters as such and then remove them from the options
        correct_mask = split_guess == split_answer
        hints_temp[correct_mask] = self.CORRECT
        split_guess[correct_mask] = GUESS_PLACEHOLDER
        split_answer[correct_mask] = ANSWER_PLACEHOLDER

        answer_letter_counts = {letter: count for letter, count in zip(*np.unique(split_answer, return_counts=True))}

        for i in range(5):
            letter = split_guess[i]
            if answer_letter_counts.get(letter, 0):
                answer_letter_counts[letter] -= 1
            else:
                split_guess[i] = GUESS_PLACEHOLDER
        hints_temp[split_guess != GUESS_PLACEHOLDER] = self.PARTIAL

        # answer_matrix = np.broadcast_to(split_answer, (5, 5)).transpose()
        # hints_temp[(answer_matrix == split_guess).any(axis=0)] = self.PARTIAL

        return hints_temp.view('U5')[0]

    def reset_state(self):
        self.game_state = pd.DataFrame(True, index=list(string.ascii_lowercase), columns=range(5))
        self.game_state['min'] = 0
        self.game_state['max'] = 5

    def update_state(self, word, hints):
        split_combo = pd.DataFrame(zip(word, hints), columns=['letter', 'hint'])
        split_info = (split_combo
                      .value_counts(sort=False)
                      .reset_index(level='hint')
                      .pivot(columns='hint')
                      .fillna(0)
                      .astype('int64')
                      .droplevel(level=0, axis=1))

        for hint in self.HINT_TYPES:
            if hint not in split_info:
                split_info[hint] = 0

        for letter, info in split_info.iterrows():
            occurrences = info[self.CORRECT] + info[self.PARTIAL]
            self.game_state.loc[letter, 'min'] = max(occurrences, self.game_state.loc[letter, 'min'])
            if info[self.INCORRECT]:
                self.game_state.loc[letter, 'max'] = occurrences

        for position, info in split_combo.iterrows():
            if info.hint == self.INCORRECT:
                if self.game_state.loc[info.letter, 'max'] == 0:
                    self.game_state.loc[info.letter,
                                        np.arange(5)] = False
            elif info.hint == self.PARTIAL:
                self.game_state.loc[info.letter, position] = False
            elif info.hint == self.CORRECT:
                self.game_state.loc[:, position] = False
                self.game_state.loc[info.letter,
                                    position] = True
            else:
                raise ValueError(f"How did we get this hint?: '{position.hint}'.")

    def target_words_left(self):
        target_letter_counts = self.letter_counts.loc[self.target_words, :]
        min_count, max_count = (self.game_state.loc[target_letter_counts.letter, 'min'],
                                self.game_state.loc[target_letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= target_letter_counts.counts.to_numpy(),
                                  target_letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=target_letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = self.target_words[mask1].view('U1').reshape((len(self.target_words[mask1]), -1))
        mask2 = self.game_state[np.arange(5)].to_numpy()[split_words.view(int) - ord('a'), np.arange(5)].all(axis=1)

        return self.target_words[mask1][mask2]

    def words_left(self):
        min_count, max_count = (self.game_state.loc[self.letter_counts.letter, 'min'],
                                self.game_state.loc[self.letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= self.letter_counts.counts.to_numpy(),
                                  self.letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=self.letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = self.all_words[mask1].view('U1').reshape((len(self.all_words[mask1]), -1))
        mask2 = self.game_state[np.arange(5)].to_numpy()[split_words.view(int) - ord('a'), np.arange(5)].all(axis=1)

        return self.all_words[mask1][mask2]


class WordleSolverMinTargets(SolverMinTargets, WordleSolverBase):

    def create_word_matrix(self):
        hints_dict = {}
        for i, answer in enumerate(self.target_words, start=1):
            hints = np.empty_like(self.all_words)
            for j, guess in enumerate(self.all_words):
                hints[j] = self.simulate_hints(guess, answer)
            hints_dict[answer] = hints

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

    solver = WordleSolverMinTargets(verbose=True)
    if TEST:
        results = solver.test()
        average_score = results.mean()
        distribution = results.value_counts().sort_index()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats('WordlePeaksSolver.main.py', 20)

        print(f"The average score for PeaksSolverMinTargets is {average_score:.4f}")
        print(distribution)
        print('Worst words:', results.loc[results == distribution.index.max()].index.to_list())
    else:
        main()
