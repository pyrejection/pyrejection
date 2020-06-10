from collections import namedtuple
from functools import wraps
from io import StringIO
import numpy as np
import os
import pandas as pd
import requests
from scipy.io.arff import loadarff
from scipy.stats import expon
from sklearn import datasets as sklearn_datasets
from sklearn.model_selection import train_test_split

DatasetTuple = namedtuple('DatasetTuple', ['df', 'numeric_columns', 'categorical_columns'])

# Use a different seed for each stage of an experiment to prevent
# overlaps and unintended correlation. Different orders of magnitude
# so that they can be repeated several times if needed.
TRAIN_TEST_RANDOM_SEED = 1
DATASET_DECORATOR_RANDOM_SEED = 10_000_000
DATASET_RANDOM_SEED = 1_000_000_000


def uci_url(path):
    return 'https://archive.ics.uci.edu/ml/machine-learning-databases/{}'.format(path)


# Caching

def cache_path(filename):
    return os.path.join('/home/jovyan/work/data', filename)


def cache_file_locally(local_path, remote_url, verify=True):
    """If local_path does not exist, downloads the remote_url and saves it
    to local_url."""
    if os.path.isfile(local_path):
        return
    r = requests.get(remote_url, verify=verify)
    with open(local_path, 'wb') as f:
        f.write(r.content)


def cache_and_unzip(local_compressed_path, local_target_dir_path,
                    remote_url, verify=True):
    """If local_compressed_path does not exist, downloads the remote_url,
    and ensures it is uncompressed inside local_target_dir_path."""
    cache_file_locally(local_compressed_path, remote_url, verify=verify)
    # Decompress
    if not os.path.isdir(local_target_dir_path):
        os.mkdir(local_target_dir_path)
        os.system('cd {} && unzip ../{}'.format(
            local_target_dir_path, os.path.basename(local_compressed_path)))


# DATASETS

# Each dataset function:
# * Makes use of caching to prevent redownloading each dataset.
# * Returns a DatasetTuple containing the DataFrame, a set of numeric
#   column names, and a set of categorical column names.
# * Stores the target feature for classification in a 'class' column.

def arem_dataset():
    compressed_local_file = cache_path('AReM.zip')
    local_dir = cache_path('arem')
    cache_and_unzip(compressed_local_file, local_dir, uci_url('00366/AReM.zip'))

    def get_activity_files(activity):
        activity_dir = os.path.join(local_dir, activity)
        activity_files = [os.path.join(activity_dir, filename)
                          for filename in os.listdir(activity_dir)]
        return sorted(activity_files)

    def load_activity_df(activity, activity_file):
        activity_df = pd.read_csv(activity_file,
                                  names=['time', 'avg_rss12', 'var_rss12', 'avg_rss13',
                                         'var_rss13', 'avg_rss23', 'var_rss23'],
                                  comment='#')
        # Remove time column
        activity_df = activity_df.drop(['time'], axis=1)
        # Add class column
        activity_df['class'] = activity
        return activity_df

    target_activities = ['cycling', 'lying', 'sitting', 'standing', 'walking']
    all_activity_files = [(activity, activity_file)
                          for activity in target_activities
                          for activity_file in get_activity_files(activity)]
    activity_file_dfs = [load_activity_df(activity, activity_file)
                         for activity, activity_file in all_activity_files]
    df = pd.concat(activity_file_dfs)
    df = df.sample(frac=1, random_state=DATASET_RANDOM_SEED).reset_index(drop=True)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def banknote_dataset():
    local_file = cache_path('banknote.data')
    cache_file_locally(local_file, uci_url('00267/data_banknote_authentication.txt'))
    df = pd.read_csv(local_file, sep=',', index_col=False,
                     names=['variance', 'skewness', 'curtosis',
                            'entropy', 'class'])
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)

    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def diabetic_dataset():
    local_file = cache_path('messidor_features.arff')
    cache_file_locally(local_file, uci_url('00329/messidor_features.arff'))
    with open(local_file, 'r') as f:
        arff_str = f.read()
        data, meta = loadarff(StringIO(arff_str))
        df = pd.DataFrame(data).rename(columns={
            '0': 'quality_assessment',
            '1': 'pre_screening',
            '2': 'ma_detection_1',
            '3': 'ma_detection_2',
            '4': 'ma_detection_3',
            '5': 'ma_detection_4',
            '6': 'ma_detection_5',
            '7': 'ma_detection_6',
            '8': 'exudates_1',
            '9': 'exudates_2',
            '10': 'exudates_3',
            '11': 'exudates_4',
            '12': 'exudates_5',
            '13': 'exudates_6',
            '14': 'exudates_7',
            '15': 'exudates_8',
            '16': 'euclidean_distance',
            '17': 'diameter_optic_disc',
            '18': 'am_fm_classification',
            'Class': 'class',
        })
        # Ensure class is a string
        df['class'] = 'c' + df['class'].apply(str)

    categorical_columns = {'quality_assessment', 'pre_screening', 'am_fm_classification'}
    return DatasetTuple(df=df,
                        numeric_columns=((set(df.columns) - {'class'}) - categorical_columns),
                        categorical_columns=categorical_columns)


def electrical_dataset():
    local_file = cache_path('electrical_grid.csv')
    cache_file_locally(local_file, uci_url('00471/Data_for_UCI_named.csv'))
    df = pd.read_csv(local_file, sep=',', index_col=False)
    df = df.rename(columns={'stabf': 'class'})
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def eye_dataset():
    local_file = cache_path('EEG Eye State.arff')
    cache_file_locally(local_file, uci_url('00264/EEG Eye State.arff'))
    with open(local_file, 'r') as f:
        arff_str = f.read()
        data, meta = loadarff(StringIO(arff_str))
        df = pd.DataFrame(data).rename(columns={'eyeDetection': 'class'})
        # Ensure class is a string
        df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def gas_dataset():
    compressed_local_file = cache_path('gas.zip')
    local_dir = cache_path('gas')
    cache_and_unzip(compressed_local_file, local_dir, uci_url('00224/Dataset.zip'))
    # Load data files.
    dfs = []
    for i in range(1, 11):
        local_file = os.path.join(local_dir, 'Dataset', 'batch{}.dat'.format(i))
        X, y = sklearn_datasets.load_svmlight_file(local_file)
        file_df = (pd.DataFrame.sparse
                   .from_spmatrix(X, columns=['f{}'.format(i) for i in range(128)])
                   .sparse.to_dense())
        file_df['class'] = y
        dfs.append(file_df)
    df = pd.concat(dfs)
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def handwritten_digits_dataset():
    digits = sklearn_datasets.load_digits()
    df = pd.DataFrame(data=np.c_[digits['data'], digits['target']],
                      columns=(['f{}'.format(i) for i in range(64)] + ['class']))
    # Ensure class is a string
    df['class'] = 'd' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def led_dataset():
    df = pd.DataFrame([
        [1, 1, 1, 0, 1, 1, 1, 'c0'],
        [0, 0, 1, 0, 0, 1, 0, 'c1'],
        [1, 0, 1, 1, 1, 0, 1, 'c2'],
        [1, 0, 1, 1, 0, 1, 1, 'c3'],
        [0, 1, 1, 1, 0, 1, 0, 'c4'],
        [1, 1, 0, 1, 0, 1, 1, 'c5'],
        [1, 1, 0, 1, 1, 1, 1, 'c6'],
        [1, 0, 1, 0, 0, 1, 0, 'c7'],
        [1, 1, 1, 1, 1, 1, 1, 'c8'],
        [1, 1, 1, 1, 0, 1, 1, 'c9'],
    ], columns=(['segment{}'.format(i) for i in range(7)] + ['class']))
    df = df.sample(n=10_000, replace=True, random_state=DATASET_RANDOM_SEED)
    return DatasetTuple(df=df,
                        numeric_columns=set(),
                        categorical_columns=(set(df.columns) - {'class'}))


def letter_recognition_dataset():
    local_file = cache_path('letter-recognition.data')
    cache_file_locally(local_file, uci_url('letter-recognition/letter-recognition.data'))
    df = pd.read_csv(local_file, sep=',', index_col=False,
                     names=['class', 'x-box', 'y-box', 'width', 'high',
                            'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar',
                            'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
                            'y-ege', 'yegvx'])
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def mushroom_dataset():
    local_file = cache_path('mushroom.data')
    cache_file_locally(local_file, uci_url('mushroom/agaricus-lepiota.data'))
    df = pd.read_csv(local_file, sep=',', index_col=False,
                     names=['class', 'cap-shape', 'cap-surface',
                            'cap-color', 'bruises', 'odor',
                            'gill-attachment', 'gill-spacing', 'gill-size',
                            'gill-color', 'stalk-shape', 'stalk-root',
                            'stalk-surface-above-ring',
                            'stalk-surface-below-ring',
                            'stalk-color-above-ring',
                            'stalk-color-below-ring', 'veil-type',
                            'veil-color', 'ring-number', 'ring-type',
                            'spore-print-color', 'population', 'habitat'])
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=set(),
                        categorical_columns=(set(df.columns) - {'class'}))


def phishing_dataset():
    local_file = cache_path('phishing.arff')
    cache_file_locally(local_file, uci_url('00327/Training%20Dataset.arff'))
    with open(local_file, 'r') as f:
        # Remove extra whitespace in column value definitions.
        arff_str = f.read().replace('{ ', '{').replace(' }', '}')
        data, meta = loadarff(StringIO(arff_str))
        df = pd.DataFrame(data).rename(columns={'Result': 'class'})
        # Ensure class is a string
        df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=set(),
                        categorical_columns=(set(df.columns) - {'class'}))


def segment_dataset():
    local_file = cache_path('segment.dat')
    cache_file_locally(local_file, uci_url('statlog/segment/segment.dat'))
    df = pd.read_csv(local_file, sep=' ', index_col=False,
                     names=['region-centroid-col', 'region-centroid-row',
                            'region-pixel-count', 'short-line-density-5',
                            'short-line-density-2', 'vedge-mean', 'vedge-sd',
                            'hedge-mean', 'hedge-sd', 'intensity-mean',
                            'rawred-mean', 'rawblue-mean', 'rawgreen-mean',
                            'exred-mean', 'exblue-mean', 'exgreen-mean',
                            'value-mean', 'saturation-mean', 'hue-mean',
                            'class'])
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def simple_synthetic_noise_dataset():
    rng = np.random.RandomState(seed=DATASET_RANDOM_SEED)
    x1 = 'x<sub>1</sub>'
    x2 = 'x<sub>2</sub>'

    # Randomly assign the class with probability proportional to the
    # second dimension, otherwise set it based on the sign of the
    # first dimension.
    def gen_class(row):
        if rng.uniform(0, 1) > row[x2]:
            return 'pos' if row[x1] >= 0.5 else 'neg'
        else:
            return rng.choice(['pos', 'neg'])

    # Create a dataset of uniformly distributed values.
    df = pd.DataFrame({
        x1: rng.uniform(0, 1, 2000),
        x2: rng.uniform(0, 1, 2000),
    })
    df['class'] = df.apply(gen_class, axis=1)

    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def radial_synthetic_noise_dataset(noise_angle_degrees=90):
    rng = np.random.RandomState(seed=DATASET_RANDOM_SEED)
    noise_angle_radians = np.radians(noise_angle_degrees)
    noise_vector = np.array([np.cos(noise_angle_radians), np.sin(noise_angle_radians)])

    def gen_class(row):
        # The noise prob is the magnitude of this row's point along
        # the noise_vector.
        noise_prob = np.dot(
            np.array([row['first'], row['second']]),
            noise_vector
        )
        if rng.uniform(-1, 1) > noise_prob:
            return 'pos' if row['first'] >= 0 else 'neg'
        else:
            return 'pos' if rng.uniform(-1, 1) >= 0 else 'neg'

    # Create a dataset distributed within a circle, so we generate
    # points in polar space before converting to cartesian.
    phi = rng.uniform(0, (2 * np.pi), 2000)
    u = pd.Series(rng.uniform(0, 1, 2000) + rng.uniform(0, 1, 2000))
    rho = u.mask((u > 1), (2 - u))
    df = pd.DataFrame({
        'first': rho * np.cos(phi),
        # Add 1 so the minimum value is zero.
        'second': rho * np.sin(phi),
    })
    df['class'] = df.apply(gen_class, axis=1)

    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def radial_synthetic_exp_noise_dataset(noise_angle_degrees=90, lmbda=0.8, pdf_mass=0.95):
    rng = np.random.RandomState(seed=DATASET_RANDOM_SEED)
    noise_angle_radians = np.radians(noise_angle_degrees)
    noise_vector = np.array([np.cos(noise_angle_radians), np.sin(noise_angle_radians)])

    # Convert lambda to scale as per docs for expon.pdf()
    scale = 1/lmbda
    # Cover enough of the Exponential PDF to cover the given mass
    # under the PDF.
    pdf_range = expon.ppf(pdf_mass, scale=scale)

    def gen_class(row):
        # The noise point is the magnitude of this row's point along
        # the noise_vector.
        noise_point = np.dot(
            np.array([row['first'], row['second']]),
            noise_vector
        )
        # Map noise_point from range [-1, 1] to the PDF range. First
        # rescale noise_point to [0,1] and invert so that a high
        # noise_point results in a high probability in the exponential
        # PDF.
        pdf_point = (1 - ((noise_point + 1) / 2)) * pdf_range
        probability_of_noise = expon.pdf(pdf_point, scale=scale)
        if rng.uniform(0, 1) > probability_of_noise:
            return 'pos' if row['first'] >= 0 else 'neg'
        else:
            return 'pos' if rng.uniform(-1, 1) >= 0 else 'neg'

    # Create a dataset distributed within a circle, so we generate
    # points in polar space before converting to cartesian.
    phi = rng.uniform(0, (2 * np.pi), 2000)
    u = pd.Series(rng.uniform(0, 1, 2000) + rng.uniform(0, 1, 2000))
    rho = u.mask((u > 1), (2 - u))
    df = pd.DataFrame({
        'first': rho * np.cos(phi),
        'second': rho * np.sin(phi),
    })
    df['class'] = df.apply(gen_class, axis=1)

    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


def skin_segmentation_dataset():
    local_file = cache_path('Skin_NonSkin.txt')
    cache_file_locally(local_file, uci_url('00229/Skin_NonSkin.txt'))
    df = pd.read_csv(local_file, sep='\t', names=['b', 'g', 'r', 'class'])
    # Ensure class is a string
    df['class'] = 'c' + df['class'].apply(str)
    return DatasetTuple(df=df,
                        numeric_columns=(set(df.columns) - {'class'}),
                        categorical_columns=set())


# DATASET VARIATIONS

def decorate_dataset(dataset_func, transformation):
    """Decorate a dataset function with a transformation function."""
    @wraps(dataset_func)
    def decorated_dataset_func():
        dataset = dataset_func()
        return transformation(dataset)
    return decorated_dataset_func


NOISE_FEATURE_NAME = 'noise_feature'


def add_exp_class_noise(dataset, lmbda=0.8, mass=0.95):
    """Add class noise to the dataset, with the rate of noise correlated
    (with an exponential distribution) to an injected noise feature."""
    rng = np.random.RandomState(DATASET_DECORATOR_RANDOM_SEED)
    classes = sorted(dataset.df['class'].unique())
    # Convert lambda to scale as per docs for expon.pdf()
    scale = 1/lmbda
    # Cover enough of the Exponential PDF to cover the given mass
    # under the PDF.
    pdf_range = expon.ppf(mass, scale=scale)

    def class_noise(row):
        # Map NOISE_FEATURE value to point in the PDF range. Inverted
        # so that a high NOISE_FEATURE results in a high probability
        # in the exponential PDF.
        pdf_point = (1 - row[NOISE_FEATURE_NAME]) * pdf_range
        probability_of_noise = expon.pdf(pdf_point, scale=scale)
        if rng.uniform(0, 1) > probability_of_noise:
            return row['class']
        else:
            return rng.choice(classes)

    dataset.df[NOISE_FEATURE_NAME] = rng.uniform(0, 1, dataset.df.shape[0])
    dataset.df['class'] = dataset.df.apply(class_noise, axis=1)
    dataset.numeric_columns.add(NOISE_FEATURE_NAME)
    return dataset


BASE_DATASETS = {
    'arem': arem_dataset,
    'banknote': banknote_dataset,
    'diabetic': diabetic_dataset,
    'electrical': electrical_dataset,
    'eye': eye_dataset,
    'handwritten-digits': handwritten_digits_dataset,
    'gas': gas_dataset,
    'led': led_dataset,
    'letter-recognition': letter_recognition_dataset,
    'mushroom': mushroom_dataset,
    'phishing': phishing_dataset,
    'segment': segment_dataset,
    'simple-synthetic-noise': simple_synthetic_noise_dataset,
    'radial-synthetic-noise-90': lambda: radial_synthetic_noise_dataset(noise_angle_degrees=90),
    'radial-synthetic-noise-65': lambda: radial_synthetic_noise_dataset(noise_angle_degrees=65),
    'radial-synthetic-noise-45': lambda: radial_synthetic_noise_dataset(noise_angle_degrees=45),
    'radial-synthetic-noise-25': lambda: radial_synthetic_noise_dataset(noise_angle_degrees=25),
    'radial-synthetic-noise-0': lambda: radial_synthetic_noise_dataset(noise_angle_degrees=0),
    'radial-synthetic-exp-noise-90': lambda: radial_synthetic_exp_noise_dataset(noise_angle_degrees=90),
    'radial-synthetic-exp-noise-65': lambda: radial_synthetic_exp_noise_dataset(noise_angle_degrees=65),
    'radial-synthetic-exp-noise-45': lambda: radial_synthetic_exp_noise_dataset(noise_angle_degrees=45),
    'radial-synthetic-exp-noise-25': lambda: radial_synthetic_exp_noise_dataset(noise_angle_degrees=25),
    'radial-synthetic-exp-noise-0': lambda: radial_synthetic_exp_noise_dataset(noise_angle_degrees=0),
    'skin-segmentation': skin_segmentation_dataset,
}

EXP_NOISY_DATASETS = {
    f'exp-noisy-{key}': decorate_dataset(BASE_DATASETS[key], add_exp_class_noise)
    for key in [
            'banknote',
            'gas',
            'letter-recognition',
            'led',
            'arem',
            'mushroom',
            'phishing',
            'segment',
            'eye',
            'handwritten-digits',
            'electrical',
            'diabetic',
    ]
}

DATASETS = {
    **BASE_DATASETS,
    **EXP_NOISY_DATASETS,
}


def prepare_dataset(classifier, dataset, random_state, test_size, apply_preprocessing=True):
    # dataset may be a DatasetTuple or a function that generates one.
    if callable(dataset):
        dataset = dataset()

    # Dataset pre-processing as required by given classifier
    df = dataset.df
    if apply_preprocessing:
        if dataset.numeric_columns and classifier.preprocess_numeric:
            df = classifier.preprocess_numeric(df, list(dataset.numeric_columns))
        if dataset.categorical_columns and classifier.preprocess_categorical:
            df = classifier.preprocess_categorical(df, list(dataset.categorical_columns))

    # Train/test set preparation
    train_df, test_df = train_test_split(df,
                                         random_state=(random_state * TRAIN_TEST_RANDOM_SEED),
                                         test_size=test_size,
                                         shuffle=True,
                                         stratify=df['class'])
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    return {
        'df': df,
        'train_X': train_df.loc[:, train_df.columns != 'class'],
        'train_y': train_df['class'],
        'test_X': test_df.loc[:, test_df.columns != 'class'],
        'test_y': test_df['class'],
        'random_state': random_state,
    }
