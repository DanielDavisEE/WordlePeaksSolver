import unittest
import numpy as np
import pandas as pd

from main import PeaksSolverBase, PeaksSolverMinTargets


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver_base = PeaksSolverBase(limit=20)
        self.solver_targets = PeaksSolverMinTargets(limit=20)

    def test_update_ranges_init(self):
        self.solver_base.update_ranges()
        true_letter_ranges = np.array([
            ['a'] * 5,
            ['z'] * 5
        ])
        np.testing.assert_array_equal(self.solver_base.letter_ranges, true_letter_ranges)

    def test_update_ranges(self):
        self.solver_base.update_ranges('geuss', 'bcaba')
        true_letter_ranges = np.array([
            ['h', 'e', 'a', 't', 'a'],
            ['z', 'e', 't', 'z', 'r']
        ])
        np.testing.assert_array_equal(self.solver_base.letter_ranges, true_letter_ranges)

    def test_words_left(self):
        self.solver_base.letter_ranges = np.array([
            ['b'] * 5,
            ['g'] * 5
        ], dtype='U1')
        words_left = self.solver_base.words_left()
        true_words_left = pd.Series([
            'ceded',
            'ebbed',
            'edged',
            'effed',
            'egged',
        ])
        np.testing.assert_array_equal(words_left, true_words_left)

    def test_target_words_left(self):
        self.solver_base.letter_ranges = np.array([
            ['a'] * 5,
            ['s'] * 5
        ], dtype='U1')
        words_left = self.solver_base.target_words_left()
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
