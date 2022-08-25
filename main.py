"""
Wordle Peaks and Wordle Solvers
Author: Daniel Davis
"""

import os
import re
import abc
import cmd
import string
import pstats
import cProfile

import numpy as np
import pandas as pd


class SolverBase(abc.ABC):
    """
    An abstract base class for implementing solvers for Wordle-like games.
    """
    source_folder = None
    start_word, start_info = None, None
    game_state = None

    HINT_TYPES = ()

    def __init__(self, /, limit=None, verbose=False):

        self.verbose = verbose

        answers_path = os.path.join(self.source_folder, 'targets_dictionary.txt')
        with open(answers_path, 'r', encoding='utf-8') as infile:
            self.all_targets = np.array(infile.read().splitlines()[:limit], dtype='U5')

        guesses_path = os.path.join(self.source_folder, 'full_dictionary.txt')
        with open(guesses_path, 'r', encoding='utf-8') as infile:
            self.all_words = np.array(infile.read().splitlines(), dtype='U5')

        self.init()

    @property
    @abc.abstractmethod
    def available_targets(self) -> np.ndarray:
        """A property of solvers which is a one dimensional numpy array of words which match
        the known hints and are in the answer list.

        Returns: A one dimensional numpy array of words
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def available_words(self) -> np.ndarray:
        """A property of solvers which is a one dimensional numpy array of words which match
        the known hints and are in the valid guess list.

        Returns: A one dimensional numpy array of words
        """
        raise NotImplementedError

    def init(self) -> None:
        """A secondary init function, called at the end of __init__ to give more control over
        when class properties are initialised in subclasses
        """
        self.reset_state()
        self.start_word, self.start_info = self.get_best_word()
        if self.verbose:
            print("Finished init")

    @abc.abstractmethod
    def play(self) -> None:
        """A method to let a player user the solver for playing a game by being suggested
        words and inputting the corresponding hints.
        """
        best_word, info = self.start_word, self.start_info
        while (hints := self.get_clean_input(best_word, info)) != 'quit':
            self.update_state(best_word, hints)
            best_word, info = self.get_best_word()
            if info['only_word']:
                print(f"The answer is {best_word}!\n")
                self.reset_state()
                best_word, info = self.start_word, self.start_info

    def test_word(self, answer: str = None, *, verbose: bool = False) -> int:
        """A method to simulate playing a particular answer.

        Args:
            answer: The answer to try and find. If None is supplied, user input will be
                requested instead
            verbose: Whether to print information on the steps taken for a user

        Returns: An int representing the steps taken to find the answer
        """
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

    def test(self) -> pd.Series:
        """Tests the solver's performance for every answer.

        Returns: A pandas Series of the number of guesses for each answer word
        """
        results = pd.Series(None, index=self.available_targets, dtype='Int64')
        for i, answer in enumerate(self.available_targets, start=1):
            results[answer] = self.test_word(answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.available_targets)):
                print(f'Tested {i:>5}/{len(self.available_targets)}')

        return results

    def get_clean_input(self, best_word: str, info: dict) -> str:
        """Gets input from user and ensures it is a valid format
        
        Args:
            best_word: The word for the solver to suggest to the user
            info: A dictionary of information about the game state

        Returns: A string representing hints or a command

        """

        def is_clean(input_string):
            return len(input_string) == 5 and all(c in self.HINT_TYPES for c in input_string)

        user_input = input(
            f"{info['available_targets']:>4}|{info['available_words']:>5}: "
            f"Use '{best_word}' and enter hints: ").lower()
        while not is_clean(user_input) and user_input != 'quit':
            user_input = input(f"Enter a 5 letter combination of {self.HINT_TYPES[0]},"
                               f" {self.HINT_TYPES[1]} and {self.HINT_TYPES[2]}: ").lower()

        return user_input

    @abc.abstractmethod
    def simulate_hints(self, guesses: str | np.ndarray, answer: str) -> str | np.ndarray:
        """Simulate the hints for a guess or guesses against a particular answer.
        
        Args:
            guesses: The guess(es) to find hint(s) for
            answer: The answer to check against

        Returns: A string or array (matching the type of guesses) of hints
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_state(self) -> None:
        """Reset the solver's game state to its initial state
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_state(self, guess: str, hints: str) -> None:
        """Iterate the solver's game state based on a guess and its corresponding hints
        
        Args:
            guess: The most recent guess made in a game
            hints: The hints obtained from the guess
        """
        raise NotImplementedError

    def get_best_word(self) -> tuple[str, dict]:
        """Based on the current game state, find the best next guess
        
        Returns: The solver's recommended next guess and a dictionary of information
            about the guess (the number of remaining guesses and answers)
        """
        info_dict = {
            'only_word': True,
            'available_words': -1,
            'available_targets': len(self.available_targets)
        }
        if len(self.available_targets) == 1:
            return self.available_targets[0], info_dict

        info_dict['only_word'] = False
        info_dict['available_words'] = len(self.available_words)
        best_word = self._choose_best_word()
        return best_word, info_dict

    def _choose_best_word(self):
        """The overwrite-able method for determining the next best guess
        
        Returns: The solver's recommended next guess
        """
        return self.available_targets[0]


class SolverMinTargets(SolverBase, abc.ABC):

    def __init__(self, **kwargs):
        self.word_matrix = None
        super().__init__(**kwargs)

    def init(self):
        self.word_matrix = self.create_word_matrix()
        super().init()

    def create_word_matrix(self):
        hints_dict = {}
        for i, answer in enumerate(self.available_targets, start=1):
            hints_dict[answer] = self.simulate_hints(self.all_words, answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.available_targets)):
                print(f'Matrix column {i:>5}/{len(self.available_targets)}')

        return pd.DataFrame(hints_dict, index=self.all_words)

    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        if self.word_matrix is None:
            return super().simulate_hints(guesses, answer)
        return self.word_matrix.loc[guesses, answer]

    def _choose_best_word(self):
        reduced_word_matrix = self.word_matrix.loc[:, self.available_targets]
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

    @property
    def available_targets(self):
        split_guesses = self.available_targets.view('U1').reshape((len(self.available_targets), -1))
        mask = np.logical_and((self.game_state[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.game_state[1]).all(axis=1))
        return self.available_targets[mask]

    @property
    def available_words(self):
        split_guesses = self.all_words.view('U1').reshape((len(self.all_words), -1))
        mask = np.logical_and((self.game_state[0] <= split_guesses).all(axis=1),
                              (split_guesses <= self.game_state[1]).all(axis=1))
        return self.all_words[mask]

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

    def update_state(self, guess, hints):
        split_word = np.array([guess]).view('U1')
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

        self.game_state[0, :] = (np.maximum(self.game_state.view('int')[0, :],
                                            tmp.view('int')[0, :])
                                 .view('U1'))
        self.game_state[1, :] = (np.minimum(self.game_state.view('int')[1, :],
                                            tmp.view('int')[1, :])
                                 .view('U1'))

    def get_best_word(self):
        info_dict = {
            'only_word': True,
            'available_words': -1,
            'available_targets': len(self.available_targets)
        }
        if len(self.available_targets) == 1:
            return self.available_targets[0], info_dict

        available_words = self.available_words
        info_dict['only_word'] = False
        info_dict['available_words'] = len(available_words)
        best_word = self._choose_best_word()
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

    @property
    def available_targets(self):
        target_letter_counts = self.letter_counts.loc[self.available_targets, :]
        min_count, max_count = (self.game_state.loc[target_letter_counts.letter, 'min'],
                                self.game_state.loc[target_letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= target_letter_counts.counts.to_numpy(),
                                  target_letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=target_letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = (self.available_targets[mask1]
                       .view('U1')
                       .reshape((len(self.available_targets[mask1]), -1)))
        mask2 = (self.game_state[np.arange(5)]
                 .to_numpy()[split_words.view(int) - ord('a'), np.arange(5)]
                 .all(axis=1))

        return self.available_targets[mask1][mask2]

    @property
    def available_words(self):
        min_count, max_count = (self.game_state.loc[self.letter_counts.letter, 'min'],
                                self.game_state.loc[self.letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= self.letter_counts.counts.to_numpy(),
                                  self.letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=self.letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = (self.all_words[mask1]
                       .view('U1')
                       .reshape((len(self.all_words[mask1]), -1)))
        mask2 = (self.game_state[np.arange(5)]
                 .to_numpy()[split_words.view(int) - ord('a'), np.arange(5)]
                 .all(axis=1))

        return self.all_words[mask1][mask2]

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

        split_guesses = np.array([guesses]).view('U1').reshape((-1, 5))
        split_answer = np.array([answer]).view('U1')
        split_answer_pd = pd.DataFrame(np.broadcast_to(split_answer, split_guesses.shape),
                                       columns=split_answer,
                                       index=guesses)

        hints = np.full_like(split_guesses, self.INCORRECT)

        # Mark correct letters as such and then remove them from the options
        correct_mask = split_guesses == split_answer
        hints[correct_mask] = self.CORRECT
        split_guesses[correct_mask] = GUESS_PLACEHOLDER
        split_answer_pd[correct_mask] = ANSWER_PLACEHOLDER

        answer_letter_counts = (split_answer_pd
                                .melt(ignore_index=False)
                                .drop(columns=['variable'])
                                .groupby(level=0, sort=False)
                                .value_counts(sort=False)
                                .unstack()
                                .drop(columns=ANSWER_PLACEHOLDER)
                                .fillna(0)
                                .astype('int'))

        for letter in answer_letter_counts:
            letters_in_answer = split_guesses == letter
            letter_ranks = letters_in_answer.cumsum(axis=1) * letters_in_answer
            letter_counts = np.broadcast_to(answer_letter_counts[letter].to_numpy().reshape((-1, 1)),
                                            letter_ranks.shape)
            partial_letters = np.logical_and(letter_ranks, letter_ranks <= letter_counts)
            hints[partial_letters] = self.PARTIAL

        hints = hints.view('U5').reshape((-1))
        if isinstance(guesses, str):
            hints = hints[0]
        return hints

    def reset_state(self):
        self.game_state = pd.DataFrame(True, index=list(string.ascii_lowercase), columns=range(5))
        self.game_state['min'] = 0
        self.game_state['max'] = 5

    def update_state(self, guess, hints):
        split_combo = pd.DataFrame(zip(guess, hints), columns=['letter', 'hint'])
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


class WordleSolverMinTargets(SolverMinTargets, WordleSolverBase):
    """
    Select the word which minimises the largest number of remaining target words
    """


# noinspection SpellCheckingInspection
class SolverCmd(cmd.Cmd):
    """
    A command line interface to use the solvers in this module
    """
    intro = (
        "Choose a solver type with 'choose_solver' and then use it with either "
        "'play', 'test_word' or 'benchmark'.\n"
        "Type help or ? to list commands.")

    def __init__(self):
        super().__init__()
        self.solver_types = [
            PeaksSolverBase,
            PeaksSolverMinTargets,
            WordleSolverBase,
            WordleSolverMinTargets
        ]
        self.solvers = {}
        self.solver = None
        self.do_choose_solver(4)

    def do_choose_solver(self, arg):
        """Choose the solver to use from the following list by providing its number as an argument:
    1. PeaksSolverBase - A basic solver for wordle peaks
    2. PeaksSolverMinTargets - A more advanced solver for wordle peaks
    3. WordleSolverBase - A basic solver for wordle
    4. WordleSolverMinTargets - A more advanced solver for wordle

Usage:
> choose_solver 2
        """
        solver_type = self.solver_types[int(arg) - 1]
        self.solver = self.solvers.get(solver_type, None)
        if self.solver is None:
            self.solver = solver_type(verbose=True)
            self.solvers[solver_type] = self.solver

    def do_print_solver(self, _):
        """Print the type of solver currently in use

Usage:
> do_print_solver
        """
        print(type(self.solver).__name__)

    def do_play(self, _):
        """Use this mode to have the solver suggest words base on the hints you receive.

Usage:
> play
        """
        self.solver.play()

    def do_test_word(self, word):
        """Use this mode to see how the solver would have play a round.

Usage:
> test charm
        """
        self.solver.test_word(word, verbose=True)

    def do_benchmark(self, _):
        """Use this mode to find the performance of the solver over the whole target word list.

Usage:
> benchmark
        """
        profiler = cProfile.Profile(subcalls=False, builtins=False)
        profiler.enable()

        results = self.solver.test()
        average_score = results.mean()
        distribution = results.value_counts().sort_index()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats('WordlePeaksSolver.main.py', 20)

        print(f"The average score for {type(self.solver).__name__} is {average_score:.4f}")
        print(distribution)
        print('Worst words:', results.loc[results == distribution.index.max()].index.to_list())

    def emptyline(self):
        pass


def test():
    profiler = cProfile.Profile(subcalls=False, builtins=False)
    profiler.enable()
    solver = WordleSolverMinTargets(verbose=True)
    results = solver.test_word('poker', verbose=True)
    average_score = results.mean()
    distribution = results.value_counts().sort_index()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats('WordlePeaksSolver.main.py', 20)

    print(f"The average score for {type(solver).__name__} is {average_score:.4f}")
    print(distribution)
    print('Worst words:', results.loc[results == distribution.index.max()].index.to_list())


if __name__ == '__main__':
    # shell = SolverCmd()
    # shell.cmdloop()
    test()
