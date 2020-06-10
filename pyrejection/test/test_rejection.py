import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from pandas.testing import assert_series_equal
import pytest

from pyrejection.rejection import (
    REJECT,
    NULL_CLASS,
    is_reject,
    is_reject_or_null,
    is_reject_mask,
    is_reject_or_null_mask,
    probs_to_preds,
    QuantileRejector,
    IncorrectRejector,
    null_label_w_repetitions,
    rejects_to_nulls,
)


def test_is_reject():
    assert is_reject('foo') is False
    assert is_reject([0.1, 0.2, 0.7]) is False
    assert is_reject([REJECT]) is False
    assert is_reject(NULL_CLASS) is False
    assert is_reject(REJECT) is True


def test_is_reject_or_null():
    assert is_reject_or_null('foo') is False
    assert is_reject_or_null([0.1, 0.2, 0.7]) is False
    assert is_reject_or_null([REJECT]) is False
    assert is_reject_or_null(NULL_CLASS) is True
    assert is_reject_or_null(REJECT) is True


def test_is_reject_mask_classes():
    pairs = [
        ('foo', False),
        (NULL_CLASS, False),
        (REJECT, True),
    ]
    values = [pair[0] for pair in pairs]
    mask = [pair[1] for pair in pairs]
    # list input
    assert_equal(is_reject_mask(values), np.array(mask))
    # Series input
    index = list(reversed(range(len(pairs))))
    assert_series_equal(is_reject_mask(pd.Series(values, index=index)),
                        pd.Series(mask, index=index))


def test_is_reject_mask_probs():
    pairs = [
        ([0.1, 0.2, 0.7], False),
        ([REJECT], False),
        (REJECT, True),
    ]
    values = [pair[0] for pair in pairs]
    mask = [pair[1] for pair in pairs]
    # list input
    assert_equal(is_reject_mask(values), np.array(mask))
    # ndarray input
    assert_equal(is_reject_mask(np.array(values)), np.array(mask))
    # Series input
    index = list(reversed(range(len(pairs))))
    assert_series_equal(is_reject_mask(pd.Series(values, index=index)),
                        pd.Series(mask, index=index))
    # No rejects
    assert_equal(is_reject_mask(np.array([[0.1, 0.2, 0.3],
                                          [0.4, 0.5, 0.6],
                                          [0.7, 0.8, 0.9]])),
                 np.array([False, False, False]))


def test_is_reject_or_null_mask_classes():
    pairs = [
        ('foo', False),
        (NULL_CLASS, True),
        (REJECT, True),
    ]
    values = [pair[0] for pair in pairs]
    mask = [pair[1] for pair in pairs]
    # list input
    assert_equal(is_reject_or_null_mask(values),
                 np.array(mask))
    # Series input
    index = list(reversed(range(len(pairs))))
    assert_series_equal(is_reject_or_null_mask(pd.Series(values, index=index)),
                        pd.Series(mask, index=index))


def test_is_reject_or_null_mask_probs():
    pairs = [
        ([0.1, 0.2, 0.7], False),
        ([REJECT], False),
        (REJECT, True),
    ]
    values = [pair[0] for pair in pairs]
    mask = [pair[1] for pair in pairs]
    # list input
    assert_equal(is_reject_or_null_mask(values),
                 np.array(mask))
    # ndarray input
    assert_equal(is_reject_or_null_mask(np.array(values)),
                 np.array(mask))
    # Series input
    index = list(reversed(range(len(pairs))))
    assert_series_equal(is_reject_or_null_mask(pd.Series(values, index=index)),
                        pd.Series(mask, index=index))


def test_probs_to_preds():
    classes = ['A', 'B']
    probs = [
        [0.5, 0.5],
        [0.6, 0.4],
        [0.4, 0.6],
        REJECT,
    ]
    assert_equal(probs_to_preds(classes, probs), ['A', 'A', 'B', REJECT])
    assert_equal(probs_to_preds(classes, np.array(probs)),
                 ['A', 'A', 'B', REJECT])
    assert_series_equal(probs_to_preds(classes, pd.Series(probs)),
                        pd.Series(['A', 'A', 'B', REJECT]))
    index = list(reversed(range(len(probs))))
    assert_series_equal(probs_to_preds(classes, pd.Series(probs, index=index)),
                        pd.Series(['A', 'A', 'B', REJECT], index=index))


def test_QuantileRejector():
    y_probs = [
        [0.9, 0.1, 0.0],
        [0.5, 0.3, 0.2],
        [0.8, 0.2, 0.0],
        [0.4, 0.3, 0.3],
    ]
    y_probs_w_reject = [
        [0.9, 0.1, 0.0],
        REJECT,
        [0.8, 0.2, 0.0],
        REJECT,
    ]
    rejector = QuantileRejector(q=0.5)
    rejector.fit(y_probs)
    # List input
    assert rejector.transform(y_probs) == y_probs_w_reject
    # ndarray input
    assert_equal(
        rejector.transform(np.array(y_probs)),
        [REJECT if is_reject(probs) else np.array(probs)
         for probs in y_probs_w_reject]
    )
    # Series input
    assert_series_equal(
        rejector.transform(pd.Series(y_probs)),
        pd.Series(y_probs_w_reject)
    )
    index = list(reversed(range(len(y_probs))))
    assert_series_equal(
        rejector.transform(pd.Series(y_probs, index=index)),
        pd.Series(y_probs_w_reject, index=index)
    )
    # Uniform probs
    uniform_y_probs = [
        [0.9, 0.1, 0.0],
        [0.5, 0.9, 0.2],
        [0.8, 0.2, 0.9],
        [0.9, 0.9, 0.3],
    ]
    uniform_rejector = QuantileRejector(q=0.5)
    uniform_rejector.fit(uniform_y_probs)
    assert uniform_rejector.prob_threshold == 0
    assert uniform_rejector.transform(uniform_y_probs) == uniform_y_probs
    assert uniform_rejector.transform(y_probs) == y_probs


def test_rejects_to_nulls():

    def series(items):
        return pd.Series(items, index=list(reversed(range(len(items)))))

    # Lists
    assert_equal(rejects_to_nulls([REJECT, REJECT, REJECT]),
                 [NULL_CLASS, NULL_CLASS, NULL_CLASS])
    assert_equal(rejects_to_nulls([REJECT, REJECT, 'B']),
                 [NULL_CLASS, NULL_CLASS, 'B'])
    # Series
    assert_series_equal(rejects_to_nulls(series([REJECT, REJECT, REJECT])),
                        series([NULL_CLASS, NULL_CLASS, NULL_CLASS]))
    assert_series_equal(rejects_to_nulls(series([REJECT, REJECT, 'B'])),
                        series([NULL_CLASS, NULL_CLASS, 'B']))


def test_IncorrectRejector():
    classes = ['A', 'B', 'C']
    y = pd.Series([
        'A',
        'A',
        'B',
        'B',
        'C',
        'C',
    ])
    y_probs = np.array([
        [0.8, 0.1, 0.1],  # Correct, conf=0.8
        [0.1, 0.1, 0.8],  # Incorrect, conf=0.8
        [0.2, 0.6, 0.2],  # Correct, conf=0.6
        [0.6, 0.2, 0.2],  # Incorrect, conf=0.6
        [0.3, 0.3, 0.4],  # Correct, conf=0.4
        [0.3, 0.4, 0.3],  # Incorrect, conf=0.4
    ])

    # Test exceptions
    rejector = IncorrectRejector(classes)
    with pytest.raises(ValueError) as ex:
        rejector.fit(y_probs)
    assert 'y cannot be None' in str(ex.value)
    rejector.fit(y_probs, y)
    with pytest.raises(ValueError) as ex:
        different_probs = y_probs.copy()
        different_probs[0, 0] = 0.1
        rejector.transform(different_probs)
    assert 'Cannot transform y_probs we did not fit on.' in str(ex.value)
    rejector.transform(y_probs)

    def test_rejector(iq, cq):
        rejector = IncorrectRejector(
            classes=classes,
            incorrect_quantile=iq,
            correct_quantile=cq,
        )
        rejector.fit(y_probs, y)
        return rejector.transform(y_probs)

    print(test_rejector(0.6, 0.3))

    # List input
    assert_equal(test_rejector(1, 0),
                 [
                     y_probs[0],
                     REJECT,
                     y_probs[2],
                     REJECT,
                     y_probs[4],
                     REJECT,
                 ])
    assert_equal(test_rejector(0.6, 0.3),
                 [
                     y_probs[0],
                     y_probs[1],
                     y_probs[2],
                     REJECT,
                     REJECT,
                     REJECT,
                 ])
    assert_equal(test_rejector(0, 0), y_probs)
    assert_equal(test_rejector(1, 1), np.full([y_probs.shape[0]], REJECT))


def test_null_label_w_repetitions():
    y = [
        'A',
        'A',
        'B',
        'B',
        'C',
        'C',
    ]
    repetition_rejections = [
        [REJECT, 'foo', REJECT, 'foo', 'foo', 'foo'],
        [REJECT, REJECT, REJECT, 'foo', 'foo', 'foo'],
        [REJECT, REJECT, REJECT, 'foo', REJECT, 'foo'],
    ]
    # List inputs.
    assert (null_label_w_repetitions(y, repetition_rejections, 1) ==
            [NULL_CLASS, NULL_CLASS, NULL_CLASS, 'B', NULL_CLASS, 'C'])
    assert (null_label_w_repetitions(y, repetition_rejections, 2) ==
            [NULL_CLASS, NULL_CLASS, NULL_CLASS, 'B', 'C', 'C'])
    assert (null_label_w_repetitions(y, repetition_rejections, 3) ==
            [NULL_CLASS, 'A', NULL_CLASS, 'B', 'C', 'C'])
    # Series inputs.
    y_series = pd.Series(y)
    rep_rej_series = [pd.Series(rep_rej) for rep_rej in repetition_rejections]
    assert_series_equal(null_label_w_repetitions(y_series, rep_rej_series, 1),
                        pd.Series([NULL_CLASS, NULL_CLASS, NULL_CLASS, 'B', NULL_CLASS, 'C']))
    assert_series_equal(null_label_w_repetitions(y_series, rep_rej_series, 2),
                        pd.Series([NULL_CLASS, NULL_CLASS, NULL_CLASS, 'B', 'C', 'C']))
    assert_series_equal(null_label_w_repetitions(y_series, rep_rej_series, 3),
                        pd.Series([NULL_CLASS, 'A', NULL_CLASS, 'B', 'C', 'C']))
