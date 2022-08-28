"""
Wordle Peaks and Wordle Solvers
Author: Daniel Davis
"""
import functools
import os
import re
import abc
import cmd
import sys
import time
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

    HINT_TYPES = ()

    def __init__(self, /, limit=None, verbose=False):

        self.verbose = verbose

        answers_path = os.path.join(self.source_folder, 'targets_dictionary.txt')
        with open(answers_path, 'r', encoding='utf-8') as infile:
            self.all_targets = np.array(infile.read().splitlines()[:limit], dtype='U5')

        guesses_path = os.path.join(self.source_folder, 'full_dictionary.txt')
        with open(guesses_path, 'r', encoding='utf-8') as infile:
            self.all_words = np.array(infile.read().splitlines(), dtype='U5')

        self.available_words = self.all_words
        self.available_targets = self.all_targets

        self.init()

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

    def benchmark(self) -> pd.Series:
        """Tests the solver's performance for every answer.

        Returns: A pandas Series of the number of guesses for each answer word
        """
        results = pd.Series(None, index=self.all_targets, dtype='Int64')
        start_time = time.time()
        for i, answer in enumerate(self.all_targets, start=1):
            results[answer] = self.test_word(answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.all_targets)):
                print(f'Tested {i:>5}/{len(self.all_targets)} ({time.time() - start_time:>6.2f} s)')

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

    def reset_state(self) -> None:
        """Reset the solver's game state to its initial state
        """
        self.available_words = self.all_words
        self.available_targets = self.all_targets

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
    word_matrix = None

    def init(self):
        self.word_matrix = self.create_word_matrix()
        super().init()

    def create_word_matrix(self):
        hints_dict = {}
        for i, answer in enumerate(self.all_words, start=1):
            hints_dict[answer] = self.simulate_hints(self.all_words, answer)

            if self.verbose and (re.fullmatch(r'0b10*', bin(i)) or i == len(self.all_words)):
                print(f'Matrix column {i:>5}/{len(self.all_words)}')

        return pd.DataFrame(hints_dict, index=self.all_words)

    def simulate_hints(self, guesses: str | np.ndarray, answer: str):
        if self.word_matrix is None:
            return super().simulate_hints(guesses, answer)
        return self.word_matrix.loc[guesses, answer]

    def update_state(self, guess: str, hints: str) -> None:
        previous_targets = self.word_matrix.loc[guess, self.available_targets]
        self.available_targets = previous_targets[previous_targets == hints].index.to_numpy()

        previous_targets = self.word_matrix.loc[guess, self.available_words]
        self.available_words = previous_targets[previous_targets == hints].index.to_numpy()

    def _choose_best_word(self):
        reduced_word_matrix = self.word_matrix.loc[:, self.available_targets]

        # For each possible guess, count how many answers would give each set of hints
        word_groups = (pd.melt(reduced_word_matrix, var_name='answer', value_name='hint', ignore_index=False)
                       .drop('answer', axis=1)
                       .query(f'hint != "{self.CORRECT * 5}"')
                       .reset_index()
                       .value_counts(sort=False))

        # Get the size of the largest hint group for each guess, and find the words which minimise this
        max_group = (word_groups
                     .groupby(level='index', sort=False)
                     .max())
        best_words = max_group.index[max_group == max_group.min()]

        # Using only the best words from the previous step, find the sizes of their 5 largest word groups
        best_words_max_groups = pd.DataFrame(
            word_groups[best_words]
                .sort_values()
                .droplevel([1], axis=0))
        best_words_max_groups.rename({best_words_max_groups.columns[0]: 'counts'}, axis=1, inplace=True)

        best_words_max_groups['rank'] = (best_words_max_groups
                                         .groupby(by='index')
                                         .rank('first'))

        # Sort the best words by the sizes of their 5 largest word groups
        best_words_max_groups = (best_words_max_groups
            .pivot(columns='rank', values='counts')
            .fillna(0)
            .sort_values(
            by=[float(x) for x in range(1, int(best_words_max_groups['rank'].max()) + 1)],
            ascending=True))

        # From the words which give the best performance, try to select those which fit the hints first
        filtered_best_words = best_words_max_groups.reindex(self.available_words).dropna().index.to_list()
        if filtered_best_words:
            return filtered_best_words[0]
        return best_words_max_groups.index[0]


def benchmark(solver_type):
    with cProfile.Profile(subcalls=False, builtins=False) as profiler:
        solver = solver_type(verbose=True)
        results = solver.benchmark()

    average_score = results.mean()
    distribution = results.value_counts().sort_index()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats('WordlePeaksSolver.main.py', 20)

    print(f"The average score for {type(solver).__name__} is {average_score:.4f}")
    print(distribution)
    print('Worst words:', results.loc[results == distribution.index.max()].index.to_list())


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

        self.solver_types = []
        try:
            self.solver_types += list(get_solvers())
        except NameError:
            pass

        for module_name, module in sys.modules.items():
            if 'solvers' in module_name and 'get_solvers' in dir(module):
                self.solver_types += list(getattr(module, 'get_solvers')())

        self.solvers = {}
        self.solver = None
        self.do_choose_solver(1)

    def _make_safe(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except Exception:
                print('Oops, an error occurred. Returning to the main console.\n')
                raise
            finally:
                self.solver.reset_state()

        return wrapper

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

    @_make_safe
    def do_play(self, _):
        """Use this mode to have the solver suggest words base on the hints you receive.

Usage:
> play
        """
        self.solver.play()

    @_make_safe
    def do_test_word(self, word):
        """Use this mode to see how the solver would have play a round.

Usage:
> test charm
        """
        self.solver.test_word(word, verbose=True)

    @_make_safe
    def do_benchmark(self, _):
        """Use this mode to find the performance of the solver over the whole target word list.

Usage:
> benchmark
        """

        with cProfile.Profile(subcalls=False, builtins=False) as profiler:
            results = self.solver.benchmark()

        average_score = results.mean()
        distribution = results.value_counts().sort_index()

        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats('WordlePeaksSolver.main.py', 20)

        print(f"The average score for {type(self.solver).__name__} is {average_score:.4f}")
        print(distribution)
        print('Worst words:', results.loc[results == distribution.index.max()].index.to_list())

    def emptyline(self):
        pass


if __name__ == '__main__':
    # shell = SolverCmd()
    # shell.cmdloop()
    benchmark()
