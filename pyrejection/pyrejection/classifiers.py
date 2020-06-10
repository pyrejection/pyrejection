from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
CLASSIFIER_RANDOM_SEED = 100_000

ClassifierTuple = namedtuple('ClassifierTuple',
                             ['factory',
                              'preprocess_numeric',
                              'preprocess_categorical',
                              'feature_weights',
                              'extras'])


# Categorical feature preprocessors

def onehot_encode(df, column_names, drop_first=False):
    encoded_df = pd.get_dummies(df, columns=column_names,
                                prefix=column_names, drop_first=drop_first)
    # pd.get_dummies() does not guarantee column order, so arrange
    # columns by sort order.
    return encoded_df.reindex(sorted(encoded_df.columns), axis=1)


def onehot_drop_first_encode(df, column_names):
    return onehot_encode(df, column_names, drop_first=True)


# Numeric feature preprocessors

def standard_normalise(df, column_names):
    df = df.copy()
    for col in column_names:
        colmean, colstd = df[col].mean(), df[col].std()
        if colstd == 0:
            # Avoid devision by zero. Since there there is no
            # deviation from the mean, all values should be set to
            # zero.
            df[col] = 0
        else:
            df[col] = (df[col] - colmean) / colstd
    return df


def identity_transform(df, column_names):
    return df


MAX_ITERATIONS = 10000
LOGREG_CLASSIFIER = ClassifierTuple(
    # 'auto' uses multinomial logreg when multi-class, but still uses
    # a single logreg for binary-class datasets.
    factory=lambda: LogisticRegression(multi_class='auto', solver='lbfgs',
                                       max_iter=MAX_ITERATIONS,
                                       # As lbfgs solver is used,
                                       # random_state should not make
                                       # a difference, but we set it
                                       # statically to be sure.
                                       random_state=CLASSIFIER_RANDOM_SEED),
    preprocess_numeric=standard_normalise,
    preprocess_categorical=onehot_drop_first_encode,
    feature_weights=lambda model: np.sum(np.abs(model.coef_), axis=0).tolist(),
    extras=lambda model: {
        'model_classes': list(model.classes_),
        'model_coefs': model.coef_.tolist(),
        'intercept': model.intercept_.tolist(),
    },
)

CLASSIFIERS = {
    'logreg': LOGREG_CLASSIFIER,
    'unscaled-logreg': LOGREG_CLASSIFIER._replace(preprocess_numeric=identity_transform),
}
