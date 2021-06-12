from pathlib import Path
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold


def _get_files_from_dir(directory, file_format):
    """
    Returns the list of file paths with the given format contained in the given directory and its subdirectories.

    :param directory: Path to the root directory.
    :type directory: str
    :param file_format: Format of the files to retrieve, specified as '*.format'. It can specify any pattern that
        matches the desired files as long as it is compatible with Path().rglob().
    :type file_format: str
    :return: A list with the paths to all the files found in 'dir' with format 'file_format'.
    :rtype: list[Path]
    """
    return list(Path(directory).rglob(file_format))


def _load_key_value_csv(csv_path, key_type=str, value_type=str, ignore_header=False):
    """
    Loads a CSV file that contains key, value pairs as rows into a dictionary.

    :param csv_path: Path to the CSV file.
    :type csv_path: str
    :param key_type: Type of the keys (first column), to cast them when creating the dictionary. Optional, default: str
    :type key_type: type
    :param value_type: Type of the values (second column), to cast them when creating the dictionary. Optional,
        default: str
    :type value_type: type
    :para, ignore_header: Set to true to ignore the first row of the CSV file, which is assumed to be a header
        with metadata (usually the name of the columns). Optional, default: False
    :raises ValueError: If the type conversion is not possible.
    :raises OSError: If there is a problem reading the file.
    :return: A dictionary that maps each element of the first column in the CSV to its corresponding pair in the
        second column for the same row. The type of the keys and of the values is defined by the corresponding
        parameters.
    :rtype: dict
    """
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        if ignore_header:
            # Skip first row.
            next(reader)
        csv_dict = {key_type(row[0]): value_type(row[1]) for row in reader}

    return csv_dict


class Dataset:
    def __init__(
            self, train_set_dir, test_set_dir, file_format,
            train_labels_file, label_type=int, train_labels_file_has_header=True):
        """
        A dataset that consists of a directory with a set of train examples and a directory with a set of test examples.
        Each example is stored in a different file, with a given file format.
        There is a CSV file that matches each example with its label. Each example is identified by its file name
        without the extension.

        :param train_set_dir: Path to the directory that contains the training examples.
        :type train_set_dir: str
        :param test_set_dir: Path to the directory that contains the test examples.
        :type test_set_dir: str
        :param file_format: Format of the example files, specified as `*.format`. It can specify any pattern that
            matches the desired files as long as it is compatible with Path().rglob().
        :type file_format: str
        :param train_labels_file: Path to the CSV file that maps each file name to its label. It must contain one row
            per training example, with two fields: example ID (file name without extension) and label.
        :type train_labels_file: str
        :param label_type: Type of the label, usually an integer. Optional, default: int
        :type label_type: type
        :param train_labels_file_has_header: Set to True when the labels CSV file contains a first row that acts as a
            header with metadata or the names of the columns. It is used to discard this column. Optional, default: True
        :type train_labels_file_has_header: bool
        """
        # Keep label type for further parsing.
        self._label_type = label_type

        # Load file paths (each file contains one sample in the given format).
        train_set_files = _get_files_from_dir(train_set_dir, file_format)
        test_set_files = _get_files_from_dir(test_set_dir, file_format)

        # Load labels (given in a "key, value" CSV file that maps each sample ID to its label).
        train_labels = _load_key_value_csv(train_labels_file, str, label_type, train_labels_file_has_header)

        # Map train files to labels using the previous mapping.
        # It is assumed that the sample ID is the file name without extension (train_file.stem).
        train_files_to_labels = {train_file: train_labels[train_file.stem] for train_file in train_set_files}

        # Create train and test arrays.
        # Each row of the array is a sample. The first column is the sample file path and the second column
        # (only in train set) is the target label.
        self._train_set = np.array([[key, value] for key, value in train_files_to_labels.items()])
        self._test_set = np.array(test_set_files)[..., np.newaxis]

    def train_set_in_splits_generator(self, n_splits, shuffle=True, seed=42):
        """
        Generator that yields one split of the train set each time. The train set is divided in N splits while
        keeping the same label distribution as in the complete set.

        :param n_splits: Number of partitions the train set will be split into.
        :type n_splits: int
        :param shuffle: Whether to shuffle the dataset prior to partitioning or not. Optional, default: True
        :type shuffle: bool
        :param seed: Seed to initialize the random number generator for the shuffling. It must be set to the same
            value between executions to ensure the generated subsets are the same. Optional, default: 42
        :type seed: int

        :return: Yields a split of the original train set on each iteration, with shape
            (~ N_EXAMPLES // N_SPLITS, 2). The actual number of examples differs among splits if the total number
            of examples in the training set is not a multiple of the number of splits. The two columns are the
            example's file path and its manual label.
        :rtype: numpy.ndarray
        """
        # Seed must be None when not shuffling.
        if not shuffle:
            seed = None
        # Generate the K-Fold division. This object will split a dataset using its labels to generate the folds.
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        # The train set is split maintaining the label distribution (column 1 of the array).
        # The split method returns the row indices of the train and validation subsets for each fold.
        # KFold divides the dataset in N splits and, for each fold, assigns N-1 splits
        # to the training subset and the remaining split to the validation subset, leaving out
        # a different validation subset on each fold.
        # Thus, to divide the data in N splits, we use the validation subset of each fold.
        for fold_id, (_, val_ids) in enumerate(
                kfold.split(self._train_set[:, 0], self._train_set[:, 1].astype(self._label_type))):
            yield self._train_set[val_ids]

    def test_set_in_splits_generator(self, n_splits, shuffle=True, seed=42):
        """
        Generator that yields one split of the test set each time. The test set is divided in N splits with equal size.

        :param n_splits: Number of partitions the test set will be split into.
        :type n_splits: int
        :param shuffle: Whether to shuffle the dataset prior to partitioning or not. Optional, default: True
        :type shuffle: bool
        :param seed: Seed to initialize the random number generator for the shuffling. It must be set to the same
            value between executions to ensure the generated subsets are the same. Optional, default: 42
        :type seed: int

        :return: Yields a split of the original test set on each iteration, with shape
            (~ N_EXAMPLES // N_SPLITS, 1). The actual number of examples differs among splits if the total number
            of examples in the test set is not a multiple of the number of splits. The only column is the
            example's file path, since the test set has no labels.
        :rtype: numpy.ndarray
        """
        # The test set does not need to be split using K-Fold because it has no known labels and will
        # only be used for inference. In this case, it is split evenly in equally sized chunks.
        indices = np.arange(len(self._test_set))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for fold_id, val_ids in enumerate(np.array_split(indices, n_splits)):
            yield self._test_set[val_ids]
