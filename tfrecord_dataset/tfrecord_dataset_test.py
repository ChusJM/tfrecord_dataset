import unittest
import numpy as np
import csv
from pathlib import Path
import shutil
import dataset
import tfrecords


class MyTestCase(unittest.TestCase):
    DUMMY_DATASET_PATH = Path('./tfrecord_dataset_test_dummy_dataset')
    TRAIN_SET = np.random.rand(10, 240, 250, 6)
    TEST_SET = np.random.rand(5, 240, 250, 6)
    LABELS = np.array([[idx, np.random.rand() < 0.75] for idx in range(len(TRAIN_SET))])

    @classmethod
    def setUpClass(cls) -> None:
        """
        Creates the dummy original dataset for tests.
        """
        # Create root directory and train and test subdirectories.
        cls.DUMMY_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        (cls.DUMMY_DATASET_PATH / 'train').mkdir(exist_ok=True)
        (cls.DUMMY_DATASET_PATH / 'test').mkdir(exist_ok=True)

        # Populate the directory with random examples. Each example is saved in a different *.npy file.
        for idx, example in enumerate(cls.TRAIN_SET):
            np.save(str(cls.DUMMY_DATASET_PATH / 'train' / f'{idx}.npy'), example)
        for idx, example in enumerate(cls.TEST_SET):
            np.save(str(cls.DUMMY_DATASET_PATH / 'test' / f'{idx}.npy'), example)

        # Create labels file with random targets.
        with open(cls.DUMMY_DATASET_PATH / 'labels.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'target'])
            writer.writeheader()
            for row in cls.LABELS:
                writer.writerow({'id': row[0], 'target': row[1]})

    def _test_dataset(self, label_type):
        """
        Tests dataset.py and tfrecords.py basic functionality using the dummy dataset.

        It uses dataset.py to read and split the dummy dataset, and tfrecords.py to save each partition in a
        different tfrecord file. Finally, tfrecords.py is used to read the tfrecord files and example data and
        labels are compared to the corresponding ones in the original dataset, to check that the process of
        saving and loading is correct.
        :param label_type: Type of the label in the labels file. Labels can be interpreted as int or as str.
        :type label_type: type
        """
        # Load dataset.
        dummy_dataset = dataset.Dataset(
            str(self.DUMMY_DATASET_PATH / 'train'), str(self.DUMMY_DATASET_PATH / 'test'), '*.npy',
            str(self.DUMMY_DATASET_PATH / 'labels.csv'), label_type, True)

        # Partition the train set in equally distributed splits, and save each split in a different tfrecord file.
        data_type = None
        data_shape = None
        # Create 5 splits after shuffling the examples using the specified seed for the random number generator.
        for idx, train_set_split in enumerate(dummy_dataset.train_set_in_splits_generator(5, True, 42)):
            # Get the data type and shape from the first example.
            if idx == 0:
                _, data_type, data_shape, _ = tfrecords.npy_data_preprocessor(train_set_split[0, 0])
            # Save the split into a tfrecord file.
            tfrecords.write_dataset_to_file(
                train_set_split,
                str(self.DUMMY_DATASET_PATH / f'train_{idx}.tfrecord'),
                tfrecords.npy_data_preprocessor
            )

        # The same is done for test set.
        for idx, test_set_split in enumerate(dummy_dataset.test_set_in_splits_generator(2, True, 42)):
            tfrecords.write_dataset_to_file(
                test_set_split,
                str(self.DUMMY_DATASET_PATH / f'test_{idx}.tfrecord'),
                tfrecords.npy_data_preprocessor
            )

        # Load the train set from the tfrecords.
        tf_train_set = tfrecords.load_dataset_from_files(
            list(map(str, self.DUMMY_DATASET_PATH.glob('train_*.tfrecord'))),
            data_shape, data_type, label_type
        )

        # Read each recovered example and check it is equal to the original one.
        for recovered in tf_train_set:
            data, label, example_id = recovered['data'], recovered['label'], recovered['example_id']
            print(example_id, label, np.abs(self.TRAIN_SET[int(example_id)] - data.numpy()).mean())
            # Check the data is close enough to the original one (some error is allowed due to floating
            # point precision conversion (original data are double and tfrecord uses float).
            self.assertTrue(np.abs(self.TRAIN_SET[int(example_id)] - data.numpy()).mean() < 1e-8)
            # Check the label is the same, parsing it with the selected data type.
            self.assertEqual(
                self.LABELS[int(example_id), 1].astype(label_type), label.numpy().decode()
                if label_type is str else label.numpy()
            )

        # Do the same for the test set
        tf_test_set = tfrecords.load_dataset_from_files(
            list(map(str, self.DUMMY_DATASET_PATH.glob('test_*.tfrecord'))),
            data_shape, data_type, None
        )
        for recovered in tf_test_set:
            data, example_id = recovered['data'], recovered['example_id']
            print(example_id, np.abs(self.TEST_SET[int(example_id)] - data.numpy()).mean())
            self.assertTrue(np.abs(self.TEST_SET[int(example_id)] - data.numpy()).mean() < 1e-8)

    def test_dataset_int_label(self):
        """
        Test dataset and tfrecords for int labels.
        """
        self._test_dataset(label_type=int)

    def test_dataset_str_label(self):
        """
        Test dataset and tfrecords for str labels.
        """
        self._test_dataset(label_type=str)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Performs cleanup by deleting the root directory of the dummy dataset and all its contents.
        """
        shutil.rmtree(cls.DUMMY_DATASET_PATH)


if __name__ == '__main__':
    unittest.main()
