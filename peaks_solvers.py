"""
Wordle Peaks and Wordle Solvers
Author: Daniel Davis
"""
import functools
import os
import re
import abc
import cmd
import string
import pstats
import cProfile
import time

import numpy as np
import pandas as pd

from solver_utils import SolverBase, SolverMinTargets, SolverCmd


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

    def update_state(self, guess, hints):
        split_word = np.array([guess]).view('U1')
        split_hints = np.array([hints]).view('U1')

        constraints = np.array([
            ['a'] * 5,
            ['z'] * 5
        ])

        constraints[0, split_hints == self.BEFORE] = (split_word[split_hints == self.BEFORE]
                                                      .view('int') + 1).view('U1')
        constraints[:, split_hints == self.CORRECT] = split_word[split_hints == self.CORRECT]
        constraints[1, split_hints == self.AFTER] = (split_word[split_hints == self.AFTER]
                                                     .view('int') - 1).view('U1')

        split_guesses = self.available_words.view('U1').reshape((len(self.available_words), -1))
        mask = np.logical_and((constraints[0] <= split_guesses).all(axis=1),
                              (split_guesses <= constraints[1]).all(axis=1))
        self.available_words = self.available_words[mask]

        split_guesses = self.available_targets.view('U1').reshape((len(self.available_targets), -1))
        mask = np.logical_and((constraints[0] <= split_guesses).all(axis=1),
                              (split_guesses <= constraints[1]).all(axis=1))
        self.available_targets = self.available_targets[mask]

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


def get_solvers():
    return PeaksSolverBase, PeaksSolverMinTargets


if __name__ == '__main__':
    shell = SolverCmd()
    shell.cmdloop()
