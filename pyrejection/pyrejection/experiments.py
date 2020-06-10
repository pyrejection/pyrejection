from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
import numpy as np
import os.path
from sklearn.metrics import accuracy_score
import traceback

import pyrejection.monkey_patches # noqa
from .rejection import (QuantileRejector, probs_to_preds,
                        rejects_to_nulls, train_and_null_label,
                        NULL_CLASS)
from .evaluation import (coverage_score, covered_metric, nl_rate_pair_to_key)
from .utils import (proportion_range,
                    make_dir_if_not_exists,
                    drop_keys)
from .classifiers import CLASSIFIERS
from .datasets import DATASETS, prepare_dataset

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
NL_RANDOM_SEED = 1000

METRICS = {
    # NOTE: Many evaluations in evaluation.py currently expect the
    # selected metric to be "accuracy".
    'accuracy': accuracy_score,
}


def test_confidence_thresholding_rejection(metric, classifier_factory,
                                           train_X, train_y, test_X, test_y,
                                           confidence_rejection_quantiles,
                                           get_feature_weights=None,
                                           get_extras=None):
    """Train a classifier, and return a dictionary mapping values from
    confidence_rejection_quantiles to a summary of performance with
    rejection based on that confidence threshold (evaluated according
    to the given metric function).
    """
    if get_feature_weights is None:
        def get_feature_weights(model):
            None
    if get_extras is None:
        def get_extras(model):
            None

    model = classifier_factory()
    model.fit(train_X, train_y)
    test_probs = model.predict_proba(test_X)

    conf_rejection_results = {}
    for q in confidence_rejection_quantiles:
        rejector = QuantileRejector(q)
        rejector.fit(test_probs)
        test_probs_rej = rejector.transform(test_probs)
        test_preds_rej = probs_to_preds(model.classes_, test_probs_rej)

        conf_rejection_results[q] = {
            'quantile': q,
            'coverage': coverage_score(test_y, test_preds_rej),
            'metric': covered_metric(metric)(test_y, test_preds_rej),
            'test_preds': rejects_to_nulls(test_preds_rej),
        }
        # To save space, only include these when q=0
        if q == 0:
            conf_rejection_results[q]['feature_weights'] = get_feature_weights(model)
            conf_rejection_results[q]['extras'] = get_extras(model)

    return conf_rejection_results


class NotEnoughClassesError(ValueError):
    pass


def check_enough_classes_for_training(train_y):
    """Some models (such as logreg) will not run when there is only the
    single NULL_CLASS left in the training set, so we need a check
    for this case.
    """
    class_counts = train_y.value_counts()
    # If a class only appears once, then we don't consider it
    # as still existing, given these errors can be generated
    # when the training set in cross-validation ends up with
    # just the NULL_CLASS because the one record is in the
    # current evaluation fold.
    greater_than_one_class_counts = class_counts[class_counts > 1]
    only_unknown_class = set(greater_than_one_class_counts.index) == {NULL_CLASS}
    if only_unknown_class:
        raise NotEnoughClassesError()


def test_null_labeling_rejection(metric, classifier_factory,
                                 train_X, train_y, test_X, test_y,
                                 nl_rate_pairs,
                                 nl_iterations,
                                 nl_repetitions,
                                 nl_rep_min_consensus,
                                 confidence_rejection_quantiles,
                                 nl_static_random_state,
                                 cv_folds=5,
                                 get_feature_weights=None,
                                 get_extras=None):
    """Iteratively train/evaluate a model with cross-validation on the
    training set, null-labeling the training set based on incorrect
    classifications after each iteration (up to nl_iterations). At
    each iteration, train a model on the full training set and perform
    evaluation with different confidence_rejection_quantiles.
    """
    if get_feature_weights is None:
        def get_feature_weights(model):
            None
    if get_extras is None:
        def get_extras(model):
            None

    nl_results = {}
    for nl_rate_pair in nl_rate_pairs:
        nl_rate_key = nl_rate_pair_to_key(nl_rate_pair)
        # Set up variables we will change on each iteration.

        nl_train_y = train_y
        nl_results[nl_rate_key] = {}
        for i in range(nl_iterations):
            iteration = i + 1
            try:
                # Perform cross-validated training and null-labeling.
                check_enough_classes_for_training(nl_train_y)
                nl_rate_incorrect, nl_rate_correct = nl_rate_pair
                random_state = (NL_RANDOM_SEED if nl_static_random_state
                                else iteration * NL_RANDOM_SEED)
                null_labelings = train_and_null_label(classifier_factory(),
                                                      train_X, nl_train_y,
                                                      incorrect_quantile=nl_rate_incorrect,
                                                      correct_quantile=nl_rate_correct,
                                                      cv_splits=cv_folds,
                                                      repetitions=nl_repetitions,
                                                      random_state=random_state,
                                                      n_jobs=4, verbose=0)
                nl_train_y = null_labelings[nl_rep_min_consensus]

                # Build a model with the null-labeled dataset that we will
                # evaluate.
                check_enough_classes_for_training(nl_train_y)
                nl_model = classifier_factory()
                nl_model.fit(train_X, nl_train_y)
                nl_test_probs = nl_model.predict_proba(test_X)
                nl_model_classes = nl_model.classes_
                nl_model_feature_weights = get_feature_weights(nl_model)
                nl_model_extras = get_extras(nl_model)
            except NotEnoughClassesError:
                # If a model cannot be trained because only the unknown
                # class is left, then set all predictions to "NULL_CLASS"
                # with 100% confidence.
                nl_test_probs = np.ones((test_y.shape[0], 1))
                nl_model_classes = [NULL_CLASS]
                nl_model_feature_weights = None
                nl_model_extras = None

            nl_results[nl_rate_key][iteration] = {}
            for q in confidence_rejection_quantiles:
                # Perform confidence-rejection to create evaluation
                # results.
                conf_rejector = QuantileRejector(q)
                conf_rejector.fit(nl_test_probs)
                nl_rej_test_probs = conf_rejector.transform(nl_test_probs)
                nl_rej_test_preds = probs_to_preds(nl_model_classes,
                                                   nl_rej_test_probs)
                nl_results[nl_rate_key][iteration][q] = {
                    'nl_rate_pair': nl_rate_pair,
                    'nl_rate_key': nl_rate_key,
                    'iteration': iteration,
                    'quantile': q,
                    'coverage': coverage_score(test_y, nl_rej_test_preds),
                    'metric': covered_metric(metric)(test_y, nl_rej_test_preds),
                    'test_preds': rejects_to_nulls(nl_rej_test_preds),
                }
                # To save space, only include these when q=0
                if q == 0:
                    nl_results[nl_rate_key][iteration][q]['feature_weights'] = nl_model_feature_weights
                    nl_results[nl_rate_key][iteration][q]['extras'] = nl_model_extras

    return nl_results


def execute_experiment(metric_name, classifier_name, dataset_name, random_state,
                       confidence_rejection_quantiles=proportion_range(10),
                       nl_rate_pairs=None,
                       nl_iterations=9,
                       nl_repetitions=1,
                       nl_rep_min_consensus=1,
                       nl_static_random_state=False,
                       test_size=0.3):
    """Perform an experiment with confidence rejection and null-labeling."""
    if nl_rate_pairs is None:
        # Default value.
        nl_rate_pairs = [(1, 0)]

    metric = METRICS[metric_name]
    classifier = CLASSIFIERS[classifier_name]
    dataset = DATASETS[dataset_name]

    dataset_parts = prepare_dataset(classifier, dataset, random_state, test_size)
    df = dataset_parts['df']
    train_X, train_y = dataset_parts['train_X'], dataset_parts['train_y']
    test_X, test_y = dataset_parts['test_X'], dataset_parts['test_y']
    classifier_factory = classifier.factory

    # Performing experimentation
    return {
        'cr_results': test_confidence_thresholding_rejection(
            metric, classifier_factory, train_X, train_y, test_X, test_y,
            confidence_rejection_quantiles=confidence_rejection_quantiles,
            get_feature_weights=classifier.feature_weights,
            get_extras=classifier.extras,
        ),
        'nl_results': test_null_labeling_rejection(
            metric, classifier_factory, train_X, train_y, test_X, test_y,
            nl_rate_pairs=nl_rate_pairs,
            nl_iterations=nl_iterations,
            nl_repetitions=nl_repetitions,
            nl_rep_min_consensus=nl_rep_min_consensus,
            confidence_rejection_quantiles=confidence_rejection_quantiles,
            nl_static_random_state=nl_static_random_state,
            get_feature_weights=classifier.feature_weights,
            get_extras=classifier.extras,
        ),
        'dataset_attributes': {
            'shape': df.shape,
            'classes': len(df['class'].unique()),
            'feature_names': train_X.columns.tolist(),
        },
        'config': {
            'metric': metric_name,
            'classifier': classifier_name,
            'dataset': dataset_name,
            'confidence_rejection_quantiles': confidence_rejection_quantiles,
            'nl_rate_pairs': nl_rate_pairs,
            'nl_iterations': nl_iterations,
            'nl_static_random_state': nl_static_random_state,
            'random_state': random_state,
            'test_size': test_size,
        },
    }


def drop_experiment_test_preds(exp_result):
    """Return a copy of the exp_result with all test_preds removed to save
    storage space."""
    result = {
        'cr_results': {
            quantile: drop_keys(summary, ['test_preds'])
            for quantile, summary in exp_result['cr_results'].items()
        },
        'nl_results': {
            nl_rate_key: {
                iteration: {
                    quantile: drop_keys(summary, ['test_preds'])
                    for quantile, summary in summaries.items()
                }
                for iteration, summaries in iterations.items()
            }
            for nl_rate_key, iterations in exp_result['nl_results'].items()
        },
        **drop_keys(exp_result, ['conf_rejection_results', 'nl_results'])
    }
    return result


def experiment_filename(metric_name, classifier_name, dataset_name, random_state):
    """Generate a filename for caching an experiment's result."""
    return f'{metric_name}-{classifier_name}-{dataset_name}-rand{random_state}.json'


def run_experiment(metric_name, classifier_name, dataset_name, random_state,
                   cache_dir=None, clear_cache=False, log=print,
                   drop_test_preds=False, discard_results=False,
                   load_limited_iterations=9, **execute_kwargs):
    """Wraps execute_experiment with caching logic."""
    filename = experiment_filename(metric_name, classifier_name, dataset_name, random_state)
    filepath = os.path.join((cache_dir or ''), filename)

    if not clear_cache and cache_dir is not None and os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            log(f'Loading {filename} from cache')
            result = json.load(f)
    else:
        log(f'Starting execution for: {filename} - {datetime.now()}')
        result = execute_experiment(metric_name,
                                    classifier_name,
                                    dataset_name,
                                    random_state=random_state,
                                    **execute_kwargs)
        log(f'Finished execution for: {filename} - {datetime.now()}')

        if drop_test_preds:
            result = drop_experiment_test_preds(result)

        json_result = json.dumps(result)
        if cache_dir is not None:
            make_dir_if_not_exists(cache_dir)
            with open(filepath, 'w') as f:
                f.write(json_result)
        # Always return result loaded from JSON to avoid
        # inconsistencies (e.g. numeric dictionary keys converted to
        # strings in JSON).
        result = json.loads(json_result)

    for iterations in result['nl_results'].values():
        to_remove = set(iterations.keys()) - set([str(i) for i in range(1, load_limited_iterations+1)])
        for iteration in to_remove:
            del iterations[iteration]

    if discard_results:
        return None
    return result


def run_experiments(metric_names, classifier_names, dataset_names, random_states,
                    worker_count=3, **kwargs):
    """Performs run_experiment for all permutations of given metrics,
    classifiers, and datasets. Makes use of multi-processing."""
    print(datetime.now())
    configs = itertools.product(metric_names, classifier_names, dataset_names, random_states)

    if worker_count <= 1:
        results = [run_experiment(*config, **kwargs) for config in configs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures_to_configs = {
                executor.submit(run_experiment, *config, **kwargs): config
                for config in configs
            }
            for future in as_completed(futures_to_configs.keys()):
                config = futures_to_configs[future]
                try:
                    results.append(future.result())
                except Exception:
                    print('Exception raised for config {}:'.format(config))
                    traceback.print_exc()

    print(datetime.now())
    return results
