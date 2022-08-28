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

    def update_state(self, guess, hints):

        game_state = pd.DataFrame(True, index=list(string.ascii_lowercase), columns=range(5))
        game_state['min'] = 0
        game_state['max'] = 5

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
            game_state.loc[letter, 'min'] = max(occurrences, game_state.loc[letter, 'min'])
            if info[self.INCORRECT]:
                game_state.loc[letter, 'max'] = occurrences

        for position, info in split_combo.iterrows():
            if info.hint == self.INCORRECT:
                if game_state.loc[info.letter, 'max'] == 0:
                    game_state.loc[info.letter,
                                   np.arange(5)] = False
            elif info.hint == self.PARTIAL:
                game_state.loc[info.letter, position] = False
            elif info.hint == self.CORRECT:
                game_state.loc[:, position] = False
                game_state.loc[info.letter,
                               position] = True
            else:
                raise ValueError(f"How did we get this hint?: '{position.hint}'.")

        # def available_targets(self):
        target_letter_counts = self.letter_counts.loc[self.available_targets, :]
        min_count, max_count = (game_state.loc[target_letter_counts.letter, 'min'],
                                game_state.loc[target_letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= target_letter_counts.counts.to_numpy(),
                                  target_letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=target_letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = (self.available_targets[mask1]
                       .view('U1')
                       .reshape((len(self.available_targets[mask1]), -1)))
        mask2 = (game_state[np.arange(5)]
                 .to_numpy()[split_words.view(int) - ord('a'), np.arange(5)]
                 .all(axis=1))

        self.available_targets = self.available_targets[mask1][mask2]

        # def available_words(self):
        min_count, max_count = (game_state.loc[self.letter_counts.letter, 'min'],
                                game_state.loc[self.letter_counts.letter, 'max'])
        in_range = np.logical_and(min_count.to_numpy() <= self.letter_counts.counts.to_numpy(),
                                  self.letter_counts.counts.to_numpy() <= max_count.to_numpy())
        mask1 = (pd.DataFrame({'inRange': in_range}, index=self.letter_counts.index)
                 .groupby(level=0, sort=False)
                 .all()).inRange

        split_words = (self.available_words[mask1]
                       .view('U1')
                       .reshape((len(self.available_words[mask1]), -1)))
        mask2 = (game_state[np.arange(5)]
                 .to_numpy()[split_words.view(int) - ord('a'), np.arange(5)]
                 .all(axis=1))

        self.available_words = self.available_words[mask1][mask2]


class WordleSolverMinTargets(SolverMinTargets, WordleSolverBase):
    """
    Select the word which minimises the largest number of remaining target words
    """


def get_solvers():
    return WordleSolverBase, WordleSolverMinTargets


if __name__ == '__main__':
    shell = SolverCmd()
    shell.cmdloop()
