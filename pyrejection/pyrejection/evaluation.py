# coding: utf-8

from collections import namedtuple
import ipywidgets as widgets
from IPython.core.display import display, display_svg
import itertools
import numpy as np
import os.path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import seaborn as sns
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import check_consistent_length
import xml.etree.ElementTree as ET

from .utils import (select_keys, make_dir_if_not_exists,
                    corrected_resampled_t_test)
from .rejection import (Classes, ClassesWReject, NULL_CLASS,
                        is_reject_or_null_mask)
from .classifiers import CLASSIFIERS
from .datasets import DATASETS, prepare_dataset, NOISE_FEATURE_NAME

"""

NOTE: Many evaluations in this file assume the "metric" is accuracy.

exp_result['Cong_rejection_results'] is a dictionary of
confidence-rejection quantiles (e.g. '0.0') that map to result
summaries.

exp_result['relabel_results'] is a dictionary of
incorrect_relabel_quantile values mapping to dictionaries of iteration
numbers (starting from 1, as strings) that map to dictionaries of
confidence-rejection quantiles (e.g. '0.0') that map to result
summaries.

"""


# METRICS

def coverage_score(y_true: Classes, y_pred: ClassesWReject,
                   sample_weight=None):
    """Return the proportion of records that are not REJECTs or
    NULL_CLASS in y_pred."""
    check_consistent_length(y_true, y_pred, sample_weight)
    weights = sample_weight if sample_weight else np.ones(len(y_true))
    covered_weight = np.sum(weights[~is_reject_or_null_mask(y_pred)])
    total_weight = np.sum(weights)
    return covered_weight / total_weight


def covered_metric(metric, empty_value=0):
    """Return a decorated version of the given metric function that will
    only be applied to records that are not REJECTs or NULL_CLASS
    in y_pred. If nothing is covered, the empty_value will be returned
    instead.
    """
    def metric_over_covered(y_true, y_pred, **kwargs):
        y_df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
        })
        covered_mask = ~is_reject_or_null_mask(y_df['pred'])
        masked_y_df = y_df[covered_mask]
        if masked_y_df.empty:
            return empty_value
        return metric(masked_y_df['true'],
                      masked_y_df['pred'],
                      **kwargs)
    return metric_over_covered


# SUMMARIES

EXPERIMENT_CT_KEY = 'cr_results'
EXPERIMENT_NL_KEY = 'nl_results'
CT_SUMMARY_KEY = 'confidence-thresholding'
NL_SUMMARY_KEY_PREFIX = 'null-labeling'
ZERO_QUANTILE_KEY = '0.0'


def is_nl_summary_key(key):
    return key.startswith(NL_SUMMARY_KEY_PREFIX)


def nl_rate_pair_to_key(nl_rate_pair):
    return f'nlrm-{nl_rate_pair[0]}-nlrc-{nl_rate_pair[1]}'


def nl_summary_key(nl_rate_key, iteration):
    return f'{NL_SUMMARY_KEY_PREFIX}-{nl_rate_key}-iteration-{iteration}'


def get_summary_oneliner(summary):
    oneliner = f'q={summary["quantile"]} (perf={summary["metric"]:2f}, cov={summary["coverage"]:2f})'
    if 'iteration' in summary:
        oneliner = f'iter={summary["iteration"]} ' + oneliner
    return oneliner


def get_experiment_base_summary(exp_result):
    return exp_result[EXPERIMENT_CT_KEY][ZERO_QUANTILE_KEY]


def get_all_summaries(exp_result):
    return {CT_SUMMARY_KEY: exp_result[EXPERIMENT_CT_KEY],
            **{nl_summary_key(nl_rate_key, iteration): summaries
               for nl_rate_key, iterations in exp_result[EXPERIMENT_NL_KEY].items()
               for iteration, summaries in iterations.items()}}


def get_ct_summaries(exp_result):
    return list(exp_result[EXPERIMENT_CT_KEY].values())


def get_nl_summaries(exp_result):
    return [summaries[ZERO_QUANTILE_KEY]
            for iterations in exp_result[EXPERIMENT_NL_KEY].values()
            for summaries in iterations.values()]


def get_nlct_summaries(exp_result):
    nlct_summaries = []
    for iterations in exp_result[EXPERIMENT_NL_KEY].values():
        for summaries in iterations.values():
            for summary in summaries.values():
                nlct_summaries.append(summary)
    return nlct_summaries


# CAPACITY

CapacityPoint = namedtuple('CapacityPoint', ['coverage', 'unconditional_error'])


def get_conditional_error(summary):
    """The conditional error, as used in:

    * Hanczar, B. (2019). Performance visualization spaces for classification with rejection option. Pattern Recognition, 96, 106984.

    The proportion of covered records that were misclassified.
    """
    # NOTE: Assume metric is accuracy over covered records.
    accuracy = summary['metric']
    return 1 - accuracy


def get_unconditional_error(summary):
    """The unconditional error, as used in:

    * Ferri, C., & Hernández-Orallo, J. (2004). Cautious Classifiers. ROCAI, 4, 27-36.
    * Hanczar, B. (2019). Performance visualization spaces for classification with rejection option. Pattern Recognition, 96, 106984.

    The proportion of all records that were misclassified (not just of
    all covered records).
    """
    # Scale down the error to represent the error out of all records,
    # not just covered records.
    return get_conditional_error(summary) * summary['coverage']


def get_summaries_capacity_points(summaries, base_summary):
    """Return a list of all capacity points for a set of summaries (each
    representing a different classifier) Includes points for the
    base classifier with no rejection and zero error at full rejection.

    A capacity point is a (coverage, unconditional-error) pair.

    We do not implement the random classifier interpolation suggested
    in (Ferri, C., & Hernández-Orallo, J. (2004). Cautious
    Classifiers. ROCAI, 4, 27-36.), as this is only really appropriate
    for balanced classes.

    """
    capacity_points = [
        # Point for base classifier with no rejection.
        CapacityPoint(1, get_unconditional_error(base_summary)),
        # Point for full rejection with zero error.
        CapacityPoint(0, 0),
    ]
    for summary in summaries:
        capacity_points.append(CapacityPoint(summary['coverage'], get_unconditional_error(summary)))
    return capacity_points


def get_capacity_convex_hull(capacity_points):
    """Create a convex hull for a (flat) list of capacity points
    (coverage, unconditional-error pairs)."""
    # Collapse points with zero abstention to just the one with
    # minimum error metric.
    min_full_coverage_error_metric = np.min([point.unconditional_error
                                             for point in capacity_points
                                             if point.coverage == 1])
    capacity_points = [point for point in capacity_points
                       if ((point.coverage != 1) or (point.unconditional_error == min_full_coverage_error_metric))]
    # Add points to complete the polygon around the top and right
    # sides of the tradeoff square.
    hull_points = [*capacity_points, CapacityPoint(1, 1), CapacityPoint(0, 0), CapacityPoint(0, 1)]
    return ConvexHull(hull_points)


def find_point_on_capacity_curve(capacity_points, given_dimension, given_value):
    """Assumes given_dimension is either 'coverage' or
    'unconditional_error', and given_value is the point along that
    dimension we want the other dimension's value at on the capacity curve.
    """
    if given_dimension == 'coverage':
        target_dimension = 'unconditional_error'
    elif given_dimension == 'unconditional_error':
        target_dimension = 'coverage'
    else:
        raise ValueError('Unrecognised given_dimension')
    # Get CapacityPoints that are on the convex hull
    convex_hull = get_capacity_convex_hull(capacity_points)
    curve_points = [CapacityPoint(*point) for point
                    in convex_hull.points[convex_hull.vertices]
                    # Exclude hull points not on the curve.
                    if tuple(point) not in {CapacityPoint(1, 1), CapacityPoint(0, 1)}]

    # Find the maximum point that has a less-or-equal given_dimension
    # value than the given_value.
    points_below = [point for point in curve_points
                    if getattr(point, given_dimension) <= given_value]
    prev_point = points_below[np.argmax([getattr(point, given_dimension)
                                         for point in points_below])]
    prev_given_value = getattr(prev_point, given_dimension)
    prev_target_value = getattr(prev_point, target_dimension)
    # Short-circuit when given_value is exactly in the list.
    if prev_given_value == given_value:
        return prev_target_value

    # Find the minimum point that has a greater than given_dimension
    # value than the given_value.
    points_above = [point for point in curve_points
                    if getattr(point, given_dimension) > given_value]
    next_point = points_above[np.argmin([getattr(point, given_dimension)
                                         for point in points_above])]
    next_given_value = getattr(next_point, given_dimension)
    next_target_value = getattr(next_point, target_dimension)

    # Linearly interpolate between the target values of the prev and
    # next points by a weighted average based on p, which represents
    # the proportion of the distance between the prev and next
    # given_dimension values that given_value sits.
    p = (given_value - prev_given_value) / (next_given_value - prev_given_value)
    interpolated_target_value = (p * next_target_value) + ((1 - p) * prev_target_value)
    return interpolated_target_value


def get_summaries_stats(summaries, base_summary):
    """Return performance stats for a list of summaries."""
    fixed_coverage_point = 0.8
    fixed_relative_unconditional_error_point = 0.5
    # Fixed unconditional_error is relative to the base unconditional_error.
    fixed_unconditional_error_point = (fixed_relative_unconditional_error_point
                                       * get_unconditional_error(base_summary))
    capacity_points = get_summaries_capacity_points(summaries, base_summary)
    convex_hull = get_capacity_convex_hull(capacity_points)
    # Capacity is the volume of the convex hull (volume represents the
    # area for 2D points - hull.area is a different metric).
    capacity = convex_hull.volume
    return {
        'Capacity': capacity,
        f'E at {fixed_coverage_point:.0%} C': find_point_on_capacity_curve(
            capacity_points, 'coverage', fixed_coverage_point),
        f'C at {fixed_relative_unconditional_error_point:.0%} of Original E': find_point_on_capacity_curve(
            capacity_points, 'unconditional_error', fixed_unconditional_error_point),
    }


def get_experiment_stats(exp_result):
    """Returns performance stats that the different reject methods
    achieved in the given experiment."""
    base_summary = get_experiment_base_summary(exp_result)
    summary_sets = {
        'ct': get_ct_summaries(exp_result),
        'nl': get_nl_summaries(exp_result),
    }
    capacities = {}
    for rej_method, summaries in summary_sets.items():
        capacities[rej_method] = get_summaries_stats(summaries, base_summary)
    return capacities


# PLOTTING

def render_svg_fig(fig):
    display_svg(fig.to_image(format='svg').decode(), raw=True)


def standard_fig_style(fig):
    fig.update_layout({
        'width': 600,
        'height': 450,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l': 0, 'r': 50, 't': 0, 'b': 0},
        'xaxis': {
            'gridcolor': '#DDDDDD',
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': '#DDDDDD',
            'zeroline': False,
        },
        'font': {
            'size': 16,
            'color': '#000000',
        },
    })


def experiment_coverage_risk_plot(exp_result, include_nlct=False,
                                  render_svg=False):
    """Return a plot for the given experiment result. If include_nlct is
    True, then the results of combining null-labeling and
    confidence-thresholding will be plotted.
    """
    all_summaries = get_all_summaries(exp_result)

    # If not include_nlct, limit nl summaries to those with a quantile
    # of 0.0.
    if not include_nlct:
        all_summaries = {
            key: (select_keys(summaries, {'0.0'})
                  if is_nl_summary_key(key) else summaries)
            for key, summaries in all_summaries.items()
        }

    def format_label(label):
        if label == 'confidence-thresholding':
            return 'CT'
        nl_matches = re.match(r'null-labeling-nlrm-([0-9\.]+)-nlrc-([0-9\.]+)-iteration-([0-9]+)', label)
        if nl_matches:
            nlrm = float(nl_matches.group(1))
            nlrc = float(nl_matches.group(2))
            iteration = nl_matches.group(3)
            nl_label = f'NL {iteration}'
            if nlrm < 1:
                nl_label += f', θ<sub>m</sub>={nlrm}'
            if nlrc > 0:
                nl_label += f', θ<sub>c</sub>={nlrc}'
            return nl_label
        return label

    fig = go.Figure()
    colours = itertools.cycle(sns.color_palette("YlOrRd_r", len(all_summaries)).as_hex())
    max_metric = 0
    # Loop through each group of summaries in all_summaries.
    for ((label, summaries), colour) in zip(all_summaries.items(), colours):
        label = format_label(label)
        # Create result DataFrame from summaries.
        result = pd.DataFrame(summaries.values())
        result['rejection'] = 1 - result['coverage']
        result['cond_error'] = 1 - result['metric']
        # Filter out undefined points with full rejection.
        result = result[result['rejection'] < 1]
        # Plot result without confidence thresholding.
        no_confidence_threshold_result = result[result['quantile'] == 0]
        fig.add_scatter(x=no_confidence_threshold_result['rejection'],
                        y=no_confidence_threshold_result['cond_error'],
                        name=label, mode='markers',
                        marker={'symbol': 'square', 'color': colour, 'size': 10})
        # Plot result with confidence thresholding.
        confidence_threshold_result = result[result['quantile'] != 0]
        if confidence_threshold_result.shape[0] > 0:
            fig.add_scatter(x=confidence_threshold_result['rejection'],
                            y=confidence_threshold_result['cond_error'],
                            name=label, mode='markers',
                            marker={'symbol': 'circle', 'color': colour, 'size': 10},
                            showlegend=False)
        max_metric = max(max_metric, result['cond_error'].max())
    # Style figure
    standard_fig_style(fig)
    fig.update_layout({
        'xaxis': {
            'range': [-0.01, 1.01],
        },
        'xaxis_title': 'Rejection Rate',
        'yaxis': {
            'range': [-0.01, (max_metric + 0.01)],
        },
        'yaxis_title': 'Conditional Error Rate',
    })
    if render_svg:
        render_svg_fig(fig)
    else:
        return fig


def project_to_2d(df_X):
    """Expects X to be a DataFrame of numeric columns, and will return a
    DataFrame with the same index but reduced to only two columns.
    """
    tsne = TSNE(n_components=2, random_state=1)
    pca = PCA(n_components=2)
    col_count = df_X.shape[1]
    if col_count <= 1:
        # Not enough dimensions
        raise ValueError('Not enough dimensions')
    if col_count <= 2:
        # No need to project down
        return df_X
    elif col_count <= 50:
        # Small enough to apply T-SNE relatively quickly
        np_X_2d = tsne.fit_transform(df_X)
    else:
        # Reduce to 50 dimensions with PCA, then apply tsne.
        np_X_50d = pca.fit_transform(df_X)
        np_X_2d = tsne.fit_transform(np_X_50d)
    return pd.DataFrame(np_X_2d, index=df_X.index)


def prepare_visual_test_set(exp_result):
    dataset_parts = prepare_dataset(
        CLASSIFIERS[exp_result['config']['classifier']],
        DATASETS[exp_result['config']['dataset']],
        exp_result['config']['random_state'],
        exp_result['config']['test_size'],
        apply_preprocessing=True,
    )
    test_X, test_y = dataset_parts['test_X'], dataset_parts['test_y']
    test_X_2d = project_to_2d(test_X)
    return test_X_2d, test_y


# Add padding to some labels to prevent cutoff in Plotly SVG output"
PLOT_CORRECT_LABEL = 'Correct'
PLOT_NULL_LABEL = 'Rejected'
PLOT_INCORRECT_LABEL = 'Incorrect   '
PLOT_UNCHANGED_LABEL = 'Unchanged   '
PLOT_NEW_CORRECT_LABEL = 'New correct'
PLOT_NEW_NULL_LABEL = 'New rejected   '
PLOT_NEW_INCORRECT_LABEL = 'New incorrect'


def get_preds_correctness(test_y, test_preds):
    """Assumes inputs are Series."""
    correctness = pd.Series(PLOT_INCORRECT_LABEL, index=test_y.index)
    correctness[test_preds == test_y] = PLOT_CORRECT_LABEL
    null_mask = is_reject_or_null_mask(test_preds)
    correctness[null_mask] = PLOT_NULL_LABEL
    return correctness


def get_correctness_diff(correctness_a, correctness_b):
    diff = pd.Series(PLOT_UNCHANGED_LABEL, index=correctness_a.index)
    changed_mask = correctness_a != correctness_b
    diff[changed_mask & (correctness_b == PLOT_INCORRECT_LABEL)] = PLOT_NEW_INCORRECT_LABEL
    diff[changed_mask & (correctness_b == PLOT_CORRECT_LABEL)] = PLOT_NEW_CORRECT_LABEL
    diff[changed_mask & (correctness_b == PLOT_NULL_LABEL)] = PLOT_NEW_NULL_LABEL
    return diff


def classification_comparison(exp_result, test_X_2d, test_y,
                              highlight_incorrect_predictions=False, jitter=0):
    """Interactive component for comparing the classifications of two
    rejecting classifiers."""
    red = '#CC2929'
    green = '#6CD96C'
    blue = '#0000FF'
    base_palette = {
        PLOT_CORRECT_LABEL: green,
        PLOT_NULL_LABEL: blue,
        PLOT_INCORRECT_LABEL: red,
        PLOT_UNCHANGED_LABEL: 'grey',
        PLOT_NEW_CORRECT_LABEL: green,
        PLOT_NEW_NULL_LABEL: blue,
        PLOT_NEW_INCORRECT_LABEL: red,
        NULL_CLASS: blue,
    }

    all_summaries = get_all_summaries(exp_result)
    if jitter > 0:
        rng = np.random.RandomState(seed=1)
        test_X_2d = test_X_2d + rng.uniform(-jitter, jitter, test_X_2d.shape)

    model_a_widget = widgets.Dropdown(description='Model:',
                                      options=all_summaries.keys())
    quantile_a_widget = widgets.Dropdown(description='Quantile:',
                                         options=all_summaries[CT_SUMMARY_KEY].keys())
    model_b_widget = widgets.Dropdown(description='Model:',
                                      options=all_summaries.keys())
    quantile_b_widget = widgets.Dropdown(description='Quantile:',
                                         options=all_summaries[CT_SUMMARY_KEY].keys())
    controls = widgets.HBox([
        widgets.VBox([widgets.Label('Classifier A'), model_a_widget, quantile_a_widget],
                     layout={'width': '50%'}),
        widgets.VBox([widgets.Label('Classifier B'), model_b_widget, quantile_b_widget],
                     layout={'width': '50%'}),
    ])
    plots = [widgets.Output() for _ in range(6)]
    plot_outputs = [
        widgets.HBox([plots[0], plots[1]]),
        widgets.HBox([plots[2], plots[3]]),
        widgets.HBox([plots[4], plots[5]]),
    ]

    def update(change):
        summary_a = all_summaries[model_a_widget.value][quantile_a_widget.value]
        test_preds_a = pd.Series(summary_a['test_preds'], index=test_y.index)
        summary_b = all_summaries[model_b_widget.value][quantile_b_widget.value]
        test_preds_b = pd.Series(summary_b['test_preds'], index=test_y.index)
        correctness_a = get_preds_correctness(test_y, test_preds_a)
        correctness_b = get_preds_correctness(test_y, test_preds_b)
        # Extras only computed for the non-thresholded, because they
        # are the same values for all thresholds.
        extras_a = all_summaries[model_a_widget.value][ZERO_QUANTILE_KEY].get('extras', {})
        extras_b = all_summaries[model_b_widget.value][ZERO_QUANTILE_KEY].get('extras', {})
        labels_high_to_low_count = test_y.value_counts().index
        labels_palette = {label: colour for label, colour
                          in zip(labels_high_to_low_count,
                                 itertools.cycle(sns.color_palette().as_hex()))}
        palette = {**labels_palette, **base_palette}

        if highlight_incorrect_predictions:
            test_preds_a = test_preds_a.mask((correctness_a == PLOT_INCORRECT_LABEL), PLOT_INCORRECT_LABEL)
            test_preds_b = test_preds_b.mask((correctness_b == PLOT_INCORRECT_LABEL), PLOT_INCORRECT_LABEL)

        plot_configs = {
            'Classifier A Predictions': {'extras': extras_a, 'hue': test_preds_a},
            'Classifier B Predictions': {'extras': extras_b, 'hue': test_preds_b},
            'Classifier A Correctness': {'extras': extras_a, 'hue': correctness_a},
            'Classifier B Correctness': {'extras': extras_b, 'hue': correctness_b},
            'True Labels': {'hue': test_y},
            'Correctness Diff': {'hue': get_correctness_diff(correctness_a, correctness_b)},
        }

        x_col, y_col = test_X_2d.columns
        xmin, xmax = test_X_2d[x_col].min(), test_X_2d[x_col].max()
        ymin, ymax = test_X_2d[y_col].min(), test_X_2d[y_col].max()

        # Plot figures
        for plot in plots:
            plot.clear_output()
        for i, (title, config) in enumerate(plot_configs.items()):
            with plots[i]:
                fig = px.scatter(test_X_2d, x=x_col, y=y_col,
                                 color=config['hue'],
                                 color_discrete_map=palette,
                                 # Hide legend label.
                                 labels={'color': ''})
                standard_fig_style(fig)
                fig.update_layout({
                    'width': 450,
                    'height': 450,
                    'legend_orientation': 'h',
                    'legend': {'x': 0, 'y': 1.1},
                    'margin': {'r': 0},
                    'xaxis': {
                        'range': [xmin, xmax],
                    },
                    'yaxis': {
                        'range': [ymin, ymax],
                        'scaleanchor': 'x',
                        'scaleratio': 1,
                    },
                })
                try:
                    # Plotting of coefficient boundaries based on:
                    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html
                    classes = config['extras']['model_classes']
                    coefs = np.array(config['extras']['model_coefs'])
                    intercept = np.array(config['extras']['intercept'])
                    if coefs.shape[1] == 2:
                        for c in range(coefs.shape[0]):
                            def line(x):
                                return (-(x * coefs[c, 0]) - intercept[c]) / coefs[c, 1]
                            if coefs.shape[0] == 1:
                                color = 'black'
                            else:
                                color = palette[classes[c]]
                            fig.add_trace(go.Scatter(
                                x=[xmin, xmax],
                                y=[line(xmin), line(xmax)],
                                mode='lines',
                                showlegend=False,
                                line=dict(
                                    color=color,
                                    width=4,
                                )
                            ))
                except KeyError:
                    # Handle case where colour cannot be found for class.
                    pass
                render_svg_fig(fig)

    model_a_widget.observe(update, names='value')
    quantile_a_widget.observe(update, names='value')
    model_b_widget.observe(update, names='value')
    quantile_b_widget.observe(update, names='value')

    display(controls, *plot_outputs)
    update(None)


def experiment_capacity_plot(exp_result, show_legend=True, scale_error_improvement=False):
    """Plot the capacity curves for the given experiment."""
    base_summary = get_experiment_base_summary(exp_result)
    base_unconditional_error = get_unconditional_error(base_summary)

    def get_capacity_hull_rows(name, summaries, base_summary):
        """Return a list of plotable rows, each of which represents a point on
        the convex hull representing the capacity curve."""
        capacity_points = get_summaries_capacity_points(summaries, base_summary)
        hull = get_capacity_convex_hull(capacity_points)
        hull_points = set([tuple(hull.points[idx]) for idx in hull.vertices])
        rows = []
        for point in capacity_points:
            # Only plot points on the hull
            if point not in hull_points:
                continue
            coverage, unconditional_error = point
            rows.append({
                'Method': name,
                'coverage': coverage,
                'unconditional_error': unconditional_error,
            })
        return sorted(rows, key=lambda row: row['coverage'])

    def plot_classifier_curves(rows):
        df = pd.DataFrame(rows)
        if scale_error_improvement:
            y_col = 'unconditional_error_improvement'
            y_title = 'Unconditional Error Rate Improvement'
            y_axis_format = {
                'tickformat': ',.0%',
                'autorange': 'reversed',
            }
            df[y_col] = (base_unconditional_error - df['unconditional_error']) / base_unconditional_error
        else:
            y_col = 'unconditional_error'
            y_title = 'Unconditional Error Rate'
            y_axis_format = {
                'range': [-0.01, (df[y_col].max() + 0.01)],
            }

        fig = px.line(df, x='coverage', y=y_col, color='Method',
                      color_discrete_map={'CT': '#99aeea', 'NL': '#b72142'})
        standard_fig_style(fig)
        fig.update_traces(mode='lines+markers')
        fig.update_layout({
            'xaxis': {
                'range': [-0.01, 1.01],
            },
            'xaxis_title': 'Coverage Rate',
            'yaxis': y_axis_format,
            'yaxis_title': y_title,
            'legend_title_text': '',
            'showlegend': show_legend,
        })
        return fig

    ct_summaries = get_ct_summaries(exp_result)
    nl_summaries = get_nl_summaries(exp_result)

    return plot_classifier_curves((get_capacity_hull_rows('CT', ct_summaries, base_summary) +
                                   get_capacity_hull_rows('NL', nl_summaries, base_summary)))


def null_class_coefs_evolution(exp_result, nl_rate_pair=None):
    """Line plot showing the coefficients for the NULL_CLASS over the
    course of null-labeling iterations."""
    if nl_rate_pair is None:
        nl_rate_pair = [1, 0]
    nl_rate_key = nl_rate_pair_to_key(nl_rate_pair)
    feature_names = exp_result['dataset_attributes']['feature_names']
    summaries = [
        iteration_summaries[ZERO_QUANTILE_KEY]
        for iteration_summaries in exp_result[EXPERIMENT_NL_KEY][nl_rate_key].values()
    ]
    coef_dfs = []
    for summary in summaries:
        if summary['extras'] is None:
            # Handle case where no model was trained
            # (e.g. single-class). The scalar value will be expanded
            # in the DataFrame initialisation below.
            null_class_coefs = 0
        else:
            model_classes = summary['extras']['model_classes']
            model_coefs = summary['extras']['model_coefs']
            null_class_index = model_classes.index(NULL_CLASS)
            if len(model_classes) < 2:
                raise ValueError('Unable to get coefficients for a 1-class model.')
            if len(model_classes) == 2:
                # Handle a binary class model with only one set of
                # coefficients where positive values indicate one class
                # and negative indicate the other. Flatten because some
                # models will still have those coefficients as the first
                # item in an array.
                if null_class_index == 0:
                    null_class_coefs = np.array(model_coefs).flatten()
                elif null_class_index == 1:
                    null_class_coefs = -np.array(model_coefs).flatten()
                else:
                    raise ValueError('Unexpected null_class_index for a binary model')
            else:
                # Handle a multi class model where each set of
                # coefficients is for a different class.
                null_class_coefs = np.array(model_coefs[null_class_index])
            if null_class_coefs.shape[0] != len(feature_names):
                raise ValueError('Unexpected number of class coefficients')
        coef_dfs.append(pd.DataFrame({
            'feature_name': feature_names,
            'null_class_coef': null_class_coefs,
            'iteration': summary['iteration'],
        }))
    all_coefs_df = pd.concat(coef_dfs).sort_values(by=['feature_name', 'iteration'],
                                                   ascending=[False, True])
    fig = px.line(all_coefs_df, x='iteration', y='null_class_coef', color='feature_name',
                  color_discrete_map={**{feature: '#99aeea' for feature in feature_names},
                                      NOISE_FEATURE_NAME: '#b72142'})
    standard_fig_style(fig)
    fig.update_layout({
        'legend_orientation': 'h',
        'xaxis': {
            'nticks': 10,
        },
        'xaxis_title': 'NL Iteration',
        'yaxis_title': 'Null-Class Coefficient Value',
        'showlegend': False,
    })
    return fig


def plot_legend_svg(label_colour_dict, trace_type='markers'):
    """Generate SVG text for a legend with the given keys and colours."""
    fig = go.Figure()

    if trace_type == 'markers':
        trace_config = {
            'mode': 'markers',
            'marker_symbol': 'square',
            'marker_size': 10,
            'marker_line_width': 2,
        }
    elif trace_type == 'lines+markers':
        trace_config = {
            'mode': 'lines+markers',
        }
    else:
        raise ValueError(f'Unrecognised trace type: {trace_type}')

    def legend_trace(fig, **kwargs):
        fig.add_trace(go.Scatter(x=[1], y=[1],
                                 **trace_config,
                                 **kwargs))

    for label, colour in label_colour_dict.items():
        legend_trace(fig, name=label, marker_color=colour)
    standard_fig_style(fig)
    fig.update_layout(legend_orientation='h')
    fig_svg = fig.to_image(format='svg').decode()

    # Select only legend from figure SVG
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    fig_root = ET.fromstring(fig_svg)
    infolayer = fig_root.find("*[@class='infolayer']")
    # Remove translation from legend.
    legend = infolayer.find("*[@class='legend']")
    if legend.attrib.get('transform'):
        legend.attrib.pop('transform')
    legend_svg_string = ET.tostring(infolayer, encoding='unicode', method='html')
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg">
    {legend_svg_string}
    </svg>
    """


def save_tradeoff_reports(results, reports_base_dir):
    """Produce an HTML report containing tradeoff plots, capacity curves,
    and feature importances for each dataset in a set of (exp_)results."""
    result_classifiers = sorted(set([result['config']['classifier'] for result in results]))
    result_metrics = sorted(set([result['config']['metric'] for result in results]))
    for metric in result_metrics:
        for classifier in result_classifiers:
            print(f'Generating report for: {metric}-{classifier}')
            dataset_results = [result for result in results
                               if (result['config']['classifier'] == classifier)]
            dataset_results = sorted(dataset_results,
                                     key=lambda res: res['config']['dataset'])
            report_dir = os.path.join(reports_base_dir, f'{metric}-{classifier}')
            make_dir_if_not_exists(report_dir)

            # Capacity curve legend
            capacity_legend = plot_legend_svg({
                'CT': '#99aeea',
                'NL': '#b72142',
            }, trace_type='lines+markers')
            with open(os.path.join(report_dir, 'capacity-legend.svg'), 'w') as f:
                f.write(capacity_legend)
            # Null class coefficients legend
            null_coefs_legend = plot_legend_svg({
                NOISE_FEATURE_NAME: '#b72142',
                'Other features': '#99aeea',
            })
            with open(os.path.join(report_dir, 'null-class-coefs-legend.svg'), 'w') as f:
                f.write(null_coefs_legend)
            # Generate plot images
            for experiment in dataset_results:
                dataset = experiment['config']['dataset']
                # CT vs NL tradeoff
                ct_vs_nl_plot = experiment_coverage_risk_plot(experiment)
                ct_vs_nl_plot.write_image(os.path.join(report_dir, f'{dataset}-CTvsNL.svg'))
                # CT vs NLCT tradeoff
                ct_vs_nlct_plot = experiment_coverage_risk_plot(experiment, include_nlct=True)
                ct_vs_nlct_plot.write_image(os.path.join(report_dir, f'{dataset}-CTvsNL+CT.svg'))
                # Capacity Curve
                capacity_fig = experiment_capacity_plot(experiment, show_legend=False)
                capacity_fig.write_image(os.path.join(report_dir, f'{dataset}-capacity-curve.svg'))
                # Null class coefficients
                null_coefs_fig = null_class_coefs_evolution(experiment)
                null_coefs_fig.write_image(os.path.join(report_dir, f'{dataset}-null-class-coefs.svg'))

            # Generate report HTML
            report_rows = [
                f"""
                <tr>
                    <td>
                    {dataset}
                    </td>
                    <td>
                        <img src="{dataset}-capacity-curve.svg">
                        <img src="capacity-legend.svg">
                    </td>
                    <td>
                        <img src="{dataset}-CTvsNL.svg">
                    </td>
                    <td>
                        <img src="{dataset}-CTvsNL+CT.svg">
                    </td>
                    <td>
                        <img src="{dataset}-null-class-coefs.svg">
                        <img src="null-class-coefs-legend.svg">
                    </td>
                </tr>
                """
                for dataset in [dataset_result['config']['dataset'] for dataset_result in dataset_results]
            ]
            report_html = """
            <html>
              <head>
                <style type="text/css">
                 table {{ page-break-inside:auto; }}
                 td    {{ vertical-align: top; }}
                 tr    {{ page-break-inside:avoid; page-break-after:auto; }}
                 br    {{ page-break-before:avoid; page-break-after:auto; }}
                </style>
              </head>
              <body>
                  <h1>{metric}-{classifier}</h1>
                  <table>
                      {report_rows}
                  </table>
              </body>
            </html>
            """.format(metric=metric, classifier=classifier, report_rows='\n'.join(report_rows))
            with open(os.path.join(report_dir, 'report.html'), 'w') as f:
                f.write(report_html)


# CAPACITY ANALYSIS

def ct_vs_nl_statistical_test(test_size, ct_stats, nl_stats):
    return {
        'ct-mean': np.mean(ct_stats),
        'ct-std': np.std(ct_stats),
        'nl-mean': np.mean(nl_stats),
        'nl-std': np.std(nl_stats),
        'nl-gt-ct': np.mean(nl_stats) > np.mean(ct_stats),
        # 2-sided corrected paired T-test
        'ct-vs-nl-2t-p': corrected_resampled_t_test(nl_stats, ct_stats,
                                                    test_size),
        # 2-sided Wilcoxon signed-rank test
        'ct-vs-nl-2w-p': stats.wilcoxon(nl_stats, ct_stats)[1],
    }


# FEATURE NOISE INFLUENCE

def correctness_stats(correctness):
    rejection = correctness.value_counts().get(PLOT_NULL_LABEL, 0) / correctness.shape[0]
    coverage = 1 - rejection
    unconditional_error = correctness.value_counts().get(PLOT_INCORRECT_LABEL, 0) / correctness.shape[0]
    return {
        'rejection': rejection,
        'coverage': coverage,
        'unconditional_error': unconditional_error,
        'conditional_error': unconditional_error / coverage,
    }


def noise_feature_influence_plots(exp_result, feature_name=NOISE_FEATURE_NAME):
    """Interactive plotting tool for comparing the relationship between
    the noise_feature and classification correctness and rejections."""
    # Prepare dataset and result summaries.
    all_summaries = get_all_summaries(exp_result)
    dataset_parts = prepare_dataset(
        CLASSIFIERS[exp_result['config']['classifier']],
        DATASETS[exp_result['config']['dataset']],
        exp_result['config']['random_state'],
        exp_result['config']['test_size'],
        apply_preprocessing=False,
    )
    test_X, test_y = dataset_parts['test_X'], dataset_parts['test_y']

    # Classifier selection widgets.
    model_a_widget = widgets.Dropdown(description='Classifier:',
                                      options=all_summaries.keys())
    quantile_a_widget = widgets.Dropdown(description='CT t:',
                                         options=all_summaries[CT_SUMMARY_KEY].keys())
    model_b_widget = widgets.Dropdown(description='Classifier:',
                                      options=all_summaries.keys())
    quantile_b_widget = widgets.Dropdown(description='CT t:',
                                         options=all_summaries[CT_SUMMARY_KEY].keys())
    controls = widgets.HBox([
        widgets.VBox([widgets.Label('Classifier A'), model_a_widget, quantile_a_widget],
                     layout={'width': '50%'}),
        widgets.VBox([widgets.Label('Classifier B'), model_b_widget, quantile_b_widget],
                     layout={'width': '50%'}),
    ])

    # Plots
    plots = [widgets.Output(layout={'width': '50%'}) for _ in range(4)]
    plots_output = widgets.VBox([
        widgets.HBox([plots[0], plots[1]]),
        widgets.HBox([plots[2], plots[3]]),
    ])
    correctness_palette = {
        PLOT_CORRECT_LABEL: '#146614',
        PLOT_INCORRECT_LABEL: '#ff3333',
        PLOT_NULL_LABEL: '#2626bf',
    }

    def do_plot(correctness):
        """Plots a histogram of the correctness at different noise_feature values."""
        plot_df = test_X.assign(correctness=correctness, classes=test_y)
        fig = px.histogram(plot_df.sort_values(by='correctness'), x=feature_name, color='correctness',
                           color_discrete_map=correctness_palette)
        standard_fig_style(fig)
        fig.update_layout(
            width=470,
            height=470,
            showlegend=False,
            xaxis_title=feature_name,
            yaxis_title='Instances',
            legend={'traceorder': 'normal'}
        )
        render_svg_fig(fig)

    test_preds_base = pd.Series(
        get_experiment_base_summary(exp_result)['test_preds'],
        index=test_y.index
    )
    correctness_base = get_preds_correctness(test_y, test_preds_base)
    with plots[0]:
        # Base classifier (without rejection)
        print('Base classifier')
        do_plot(correctness_base)
    with plots[1]:
        # Legend
        display_svg(plot_legend_svg(correctness_palette), raw=True)

    def update(change):
        """Update correctness and plots for selected classifiers."""
        test_preds_a = pd.Series(
            all_summaries[model_a_widget.value][quantile_a_widget.value]['test_preds'],
            index=test_y.index
        )
        test_preds_b = pd.Series(
            all_summaries[model_b_widget.value][quantile_b_widget.value]['test_preds'],
            index=test_y.index
        )
        correctness_a = get_preds_correctness(test_y, test_preds_a)
        stats_a = correctness_stats(correctness_a)
        correctness_b = get_preds_correctness(test_y, test_preds_b)
        stats_b = correctness_stats(correctness_b)

        with plots[2]:
            plots[2].clear_output()
            print(f'{model_a_widget.value}, t={quantile_a_widget.value} (C={stats_a["coverage"]:.2%}; E\'={stats_a["conditional_error"]:.2%})')
            do_plot(correctness_a)
        with plots[3]:
            plots[3].clear_output()
            print(f'{model_b_widget.value}, t={quantile_b_widget.value} (C={stats_b["coverage"]:.2%}; E\'={stats_b["conditional_error"]:.2%})')
            do_plot(correctness_b)
        return

    # Update when selections change.
    model_a_widget.observe(update, names='value')
    quantile_a_widget.observe(update, names='value')
    model_b_widget.observe(update, names='value')
    quantile_b_widget.observe(update, names='value')

    display(controls, plots_output)
    update(None)
