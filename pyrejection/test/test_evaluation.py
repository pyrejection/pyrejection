from numpy.testing import assert_almost_equal
from pyrejection.evaluation import (
    CapacityPoint,
    find_point_on_capacity_curve
)


def test_find_point_on_capacity_curve():
    capacity_points = [
        CapacityPoint(1, 0.5),
        CapacityPoint(0.9, 0.4),
        CapacityPoint(0.3, 0.1),
        # Points closer to interpolated points, but should not be on
        # the capacity curve because of their high errors.
        CapacityPoint(0.8, 0.35),
        CapacityPoint(0.4, 0.15),
    ]

    def check_point(coverage, unconditional_error):
        assert_almost_equal(
            find_point_on_capacity_curve(capacity_points, 'coverage', coverage),
            unconditional_error,
            decimal=10,
        )
        assert_almost_equal(
            find_point_on_capacity_curve(capacity_points, 'unconditional_error', unconditional_error),
            coverage,
            decimal=10,
        )

    check_point(1, 0.5)
    check_point(0.9, 0.4)
    check_point(0.7, 0.3)
    check_point(0.6, 0.25)
    check_point(0.5, 0.2)
    check_point(0.3, 0.1)
    check_point(0.15, 0.05)
    check_point(0, 0)
