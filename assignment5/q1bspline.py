
from __future__ import annotations
from dataclasses import dataclass, replace, field
from operator import itemgetter
from scipy.interpolate import splrep, BSpline
from typing import Callable, Iterable, Sequence, Tuple, TypeVar, Optional
from rl.function_approx import FunctionApprox
import matplotlib.pyplot as plt
import numpy as np


# reimplemented same BSplineApprox class from function_approx.py
@dataclass(frozen=True)
class BSplineApprox(FunctionApprox[X]):
    feature_function: Callable[[X], float]
    deg: int
    knots: np.ndarray
    coefs: np.ndarray

    def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
        return self.feature_function(x) for x in x_values_seq

    def representational_gradient(self, x_value: X) -> BSplineApprox[X]:
        # don't understand this yet
        pass

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        # create the spline
        spline: Callable[[Sequence[float]], np.ndarray] = \
            BSpline(self.knots, self.coefs, self.degree)
        # return the spline evaluated at the x_values
        return spline(self.get_feature_values(x_values_seq))

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> BSplineApprox[X]:
        #unpack and zip together
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: Sequence[float] = self.get_feature_values(x_vals)
        
        # sort the pairs after zipping them into tuples by the feature value
        sorted_pairs: Sequence[Tuple[float, float]] = \
            sorted(zip(feature_vals, y_vals), key=itemgetter(0))
        
        # find the b-spline representation of 1-D curve
        newknots, newcoefs, _ = splrep(
            [k for k, _ in sorted_pairs],
            [c for _, c in sorted_pairs],
            k = self.degree
        )
        return replace(
            self,
            knots=newknots,
            coeffs = newcoefs
        )

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ) -> FunctionApprox[X]:
        # same as update
        return self.update(xy_vals_seq)

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        # check if a valid instance of BSpline
        if isinstance(other, BSplineApprox):
            return \
                # check that all the knots and coefs are within
                # a certain tolerance individually
                np.all(np.abs(self.knots - other.knots) <= tolerance).item() \
                and \
                np.all(np.abs(self.coeffs - other.coeffs) <= tolerance).item()

        return False

if __name__ == '__main__':
    pass