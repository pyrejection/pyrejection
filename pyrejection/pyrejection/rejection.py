from abc import ABC, abstractmethod
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import numpy as np
import pandas as pd
from typing import Any, Union, List
import warnings

from .utils import (mask_iterable, map_seq)

REJECT = np.nan
NULL_CLASS = 'NULL'

# TYPES

Classes = Union[List[str],
                List[np.str_],
                np.ndarray,
                pd.Series]

ClassesWReject = Union[Classes,
                       List[Union[str, float]],
                       List[Union[np.str_, float]],
                       List[float]]

Probs = Union[List[List[float]],
              List[np.ndarray],
              np.ndarray,
              pd.Series]

ProbsWReject = Union[Probs,
                     List[Union[List[float], float]],
                     List[Union[np.ndarray, float]],
                     List[float]]

Mask = Union[np.ndarray, pd.Series]


# REJECT/NULL UTILITY FUNCTIONS

def is_reject(value: Any) -> bool:
    """Given a class value that may be a REJECT, return True if it is."""
    return np.isscalar(value) and pd.isnull(value)


def is_reject_or_null(value: Any) -> bool:
    """Given a class value that may be a REJECT or NULL_CLASS, return True if it is."""
    return (np.isscalar(value) and pd.isnull(value)) or (value == NULL_CLASS)


def is_reject_mask(values: Union[ClassesWReject, ProbsWReject]) -> Mask:
    """Given class values or probs lists that may contain REJECTs, return
    a mask that is True for values that are REJECTs.
    """
    if isinstance(values, pd.Series):
        return pd.isnull(values)
    else:
        return np.array([is_reject(val) for val in values])


def is_reject_or_null_mask(values: Union[ClassesWReject, ProbsWReject]) -> Mask:
    """Given class values or probs lists that may contain REJECTs and
    NULL_CLASS labels, return a mask that is True for values that
    are REJECTs or NULL_CLASS.

    """
    if isinstance(values, pd.Series):
        null_mask = (values == NULL_CLASS)
    else:
        null_mask = np.array([val == NULL_CLASS for val in values])
    return (is_reject_mask(values) | null_mask)


def probs_to_preds(classes: Classes, y_probs: ProbsWReject) -> ClassesWReject:
    """Return predictions (from the set of given classes, taken from
    estimator.classes_), and the probs array (one row per record,
    which is a list of probabilities in the same order as classes).

    Handles REJECT values.
    """
    transform_row = (lambda probs: REJECT if is_reject(probs)
                     else classes[np.argmax(probs)])
    return map_seq(transform_row, y_probs)


def rejects_to_nulls(y: ClassesWReject) -> ClassesWReject:
    """Given a series y of true class values and preds_or_probs_w_reject
    with preds or probs that may contain REJECTs, return a copy of y
    where entries that were REJECTs in preds_or_probs_w_reject are
    relabeled with the NULL_CLASS.
    """
    return mask_iterable(y, is_reject_mask(y), NULL_CLASS)


# REJECTOR CLASSES

class BaseRejector(TransformerMixin, BaseEstimator, ABC):
    """A type of transformer that expects it's `X` to be a sklearn proba
    series. Transformation is expected to replace some of the proba
    entries with REJECT, and return the output as a pd.Series.
    """

    def fit(self, y_probs: Probs, y=None):
        """Takes a list, np.array, or pd.Series of y_probs and y."""
        return self

    @abstractmethod
    def transform(self, y_probs: Probs) -> ProbsWReject:
        """Take a list, np.array, or pd.Series of y_probs and return it as a
        Series with added REJECTs."""
        pass


class ThresholdRejector(BaseRejector):
    """Reject all records predicted with a maximum probability less than
    the given prob_threshold."""

    def __init__(self, prob_threshold):
        self.prob_threshold = prob_threshold

    def reject(self, probs: Union[List[float], np.ndarray]) -> bool:
        """Reject when the maximum probability for any class assigned to the
        record is less than the probability threshold."""
        if max(probs) < self.prob_threshold:
            return True
        return False

    def transform(self, y_probs: Probs) -> ProbsWReject:
        return map_seq(lambda probs: REJECT if self.reject(probs) else probs,
                       y_probs)


class QuantileRejector(ThresholdRejector):
    """Use a confidence threshold to reject the given quantile/proportion
    of records."""

    def __init__(self, q):
        self.q = q

    def fit(self, y_probs: Probs, y=None):
        """Learn the probability threshold for the configured quantile."""
        max_probs = np.max(y_probs, axis=1)
        if len(np.unique(max_probs)) == 1:
            self.prob_threshold = 0
        else:
            self.prob_threshold = np.quantile(max_probs, q=self.q,
                                              interpolation='linear')
        return self


def select_lowest_n_mask(xs: Union[List[float], np.ndarray], n):
    """Return a mask of xs that is True for the n lowest values
    in xs, and False at other indexes. NaN values are ignored."""
    # Returns a list of indexes into xs sorted by the value at that
    # index in xs. NaNs will go to the end in Numpy 1.4+. argpartition
    # may be faster, but documentation is unclear as to whether the
    # NaN property holds.
    sorted_xs = np.argsort(xs)
    # Create a full False mask, and set the first n indexes from the
    # partition to True.
    mask = np.full(len(xs), False)
    mask[sorted_xs[:n]] = True
    return mask


class IncorrectRejector(BaseRejector):
    """Rejection based on whether records were classified correctly
    (evaluation is according to y provided to fit, so the y_probs passed
    to fit and transform must be the same).

    classes must be provided in the same order as the classes are
    ordered in y_probs.

    incorrect_quantile and correct_quantile represent the proportions
    of incorrect and correct records to reject (respectively, and
    rejecting records with low confidences first).
    """

    def __init__(self, classes, incorrect_quantile=1, correct_quantile=0):
        self.classes = classes
        self.incorrect_quantile = incorrect_quantile
        self.correct_quantile = correct_quantile

    def fit(self, y_probs: np.ndarray, y: pd.Series = None):
        if y is None:
            raise ValueError('y cannot be None')
        self.y_probs = y_probs

        y_confs = np.max(y_probs, axis=1)
        y_preds = probs_to_preds(self.classes, y_probs)

        incorrect_mask = (y != y_preds)
        correct_mask = ~incorrect_mask
        incorrect_count = np.count_nonzero(incorrect_mask)
        correct_count = np.count_nonzero(correct_mask)
        incorrect_reject_mask = select_lowest_n_mask(
            # Set correct records to NaN so they are ignored.
            mask_iterable(y_confs, correct_mask, np.nan),
            int(np.ceil(incorrect_count * self.incorrect_quantile))
        )
        correct_reject_mask = select_lowest_n_mask(
            # Set correct records to NaN so they are ignored.
            mask_iterable(y_confs, incorrect_mask, np.nan),
            int(np.ceil(correct_count * self.correct_quantile))
        )
        self.reject_mask = incorrect_reject_mask | correct_reject_mask
        return self

    def transform(self, y_probs: np.ndarray) -> ProbsWReject:
        if (self.y_probs != y_probs).any():
            raise ValueError('Cannot transform y_probs we did not fit on.')

        return [REJECT if reject else probs
                for probs, reject in zip(y_probs, self.reject_mask)]


# NULL-LABELING

def null_label_w_repetitions(y_true: Classes,
                             repetition_rejections: List[Union[ClassesWReject, ProbsWReject]],
                             min_rejections: int) -> Classes:
    """Returns a copy of the y_true classes with NULL_CLASS instead for
    any record that is rejected at least min_rejections times in
    repetition_rejections."""
    rejection_masks = [is_reject_mask(preds_or_probs_w_reject).astype(int)
                       for preds_or_probs_w_reject in repetition_rejections]
    summed_mask = reduce(np.add, rejection_masks)
    final_mask = summed_mask >= min_rejections
    return mask_iterable(y_true, final_mask, NULL_CLASS)


def cross_val_predict_wo_class_warnings(*args, **kwargs):
    # Note, use of catch_warnings is not thread-safe as it alters
    # global state, but this should be fine in the multi-processing we
    # perform.
    with warnings.catch_warnings(record=True) as warns:
        y_probs = cross_val_predict(*args, **kwargs)
    for warn in (warns or []):
        is_least_populated_message = (
            isinstance(warn.message, UserWarning) and
            str(warn.message).startswith('The least populated class in y has only ')
        )
        if (is_least_populated_message):
            # Ignore warnings about insufficient members for classes,
            # as we will simply do the best possible attempt at
            # classification we can with the data we have at this
            # point in the relabeling process.
            pass
        else:
            warnings.warn(warn.message,
                          getattr(warn, 'category', None),
                          getattr(warn, 'stacklevel', 1),
                          getattr(warn, 'source', None))
    return y_probs


def train_and_null_label(classifier, X, y,
                         incorrect_quantile=1, correct_quantile=0,
                         cv_splits=4, repetitions=1,
                         random_state=1, n_jobs=-1, verbose=10,
                         sample_weight=None):
    """Use the given classifier and cross-validation (CV) config to
    produce predictions for all of X. Apply the rejector to those
    predictions, and return a null-labeled copy of y (where NULL_CLASS
    replaces the true class value for rejected records).  Returns a
    dictionary of of possible null-labelings for different minimum
    rejections over the given number of repetitions.
    """
    rng = np.random.RandomState(seed=random_state)
    repetition_rejections = []
    for repetition in range(repetitions):
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                             random_state=rng.randint(1_000_000))
        y_probs = cross_val_predict_wo_class_warnings(classifier, X, y, cv=cv,
                                                      n_jobs=n_jobs,
                                                      verbose=verbose,
                                                      method='predict_proba',
                                                      fit_params={'sample_weight': sample_weight})
        # np.unique() returns classes in sorted order, and is also
        # used by sklearn to order the classes in probabilities
        # returned by cross_val_predict().
        rejector = IncorrectRejector(classes=np.unique(y),
                                     incorrect_quantile=incorrect_quantile,
                                     correct_quantile=correct_quantile)
        rejector.fit(y_probs, y)
        y_probs_w_reject = rejector.transform(y_probs)
        repetition_rejections.append(y_probs_w_reject)
    null_labelings = {
        min_rejections: null_label_w_repetitions(y, repetition_rejections, min_rejections)
        for min_rejections in range(1, repetitions+1)
    }
    return null_labelings
