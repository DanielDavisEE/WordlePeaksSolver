"""
Wordle Peaks and Wordle Solvers
Author: Daniel Davis
"""

from solver_utils import SolverCmd
from peaks_solvers import get_solvers as get_peaks_solvers
from wordle_solvers import get_solvers as get_wordle_solvers


def get_solvers():
    return get_peaks_solvers() + get_wordle_solvers()


if __name__ == '__main__':
    shell = SolverCmd()
    shell.cmdloop()
