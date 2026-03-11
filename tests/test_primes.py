import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from weil_functional import WeilFunctional
from weil_primes import PrimesContribution


def test_prime_count_semantics():
    f, _ = WeilFunctional.gaussian_testfunc(1.0)
    _, contributions = PrimesContribution.compute(f, num_primes=10)
    primes = list(contributions.keys())
    assert len(primes) == 10
    assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
