import unittest
import numpy as np
import pandas as pd

from main import PeaksSolverBase, PeaksSolverMinTargets, WordleSolverBase


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaks_solver_base = PeaksSolverBase(limit=20)
        self.peaks_solver_targets = PeaksSolverMinTargets(limit=20)
        self.wordle_solver_base = WordleSolverBase(limit=20)

    def test_simulate_hints(self):
        hint = self.wordle_solver_base.simulate_hints('climb', 'frill')
        self.assertEqual(hint, 'bygbb')
        hint = self.wordle_solver_base.simulate_hints('rfete', 'greet')
        self.assertEqual(hint, 'ybgyy')
        hint = self.wordle_solver_base.simulate_hints('rfete', 'gremt')
        self.assertEqual(hint, 'ybgyb')
        hint = self.wordle_solver_base.simulate_hints('paper', 'empty')
        self.assertEqual(hint, 'bbgyb')
        hint = self.wordle_solver_base.simulate_hints('ppaer', 'empty')
        self.assertEqual(hint, 'ybbyb')

    def test_reset_state(self):
        self.peaks_solver_base.reset_state()
        true_letter_ranges = np.array([
            ['a'] * 5,
            ['z'] * 5
        ])
        np.testing.assert_array_equal(self.peaks_solver_base.game_state, true_letter_ranges)

    def test_update_state(self):
        self.peaks_solver_base.update_state('guess', 'ygbyb')
        true_letter_ranges = np.array([
            ['h', 'u', 'a', 't', 'a'],
            ['z', 'u', 'd', 'z', 'r']
        ])
        np.testing.assert_array_equal(self.peaks_solver_base.game_state, true_letter_ranges)

    def test_words_left(self):
        self.peaks_solver_base.game_state = np.array([
            ['b'] * 5,
            ['g'] * 5
        ], dtype='U1')
        words_left = self.peaks_solver_base.words_left()
        true_words_left = pd.Series([
            'ceded',
            'ebbed',
            'edged',
            'effed',
            'egged',
        ])
        np.testing.assert_array_equal(words_left, true_words_left)

    def test_target_words_left(self):
        self.peaks_solver_base.game_state = np.array([
            ['a'] * 5,
            ['s'] * 5
        ], dtype='U1')
        words_left = self.peaks_solver_base.target_words_left()
        true_words_left = pd.Series([
            'charm',
            'skill',
            'peace',
            'choir',
            'erase',
            'chord',
            'snack',
            'flair',
            'radio',
        ])
        np.testing.assert_array_equal(words_left, true_words_left)


if __name__ == '__main__':
    unittest.main()
