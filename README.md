# tfrecord_dataset
**TFRecord Dataset** is a simple tool written in [Python](https://www.python.org/) for handling [datasets](https://en.wikipedia.org/wiki/Data_set)
that are composed of several files, each file containing the data of one example, 
and exporting them to [TensorFlow's TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files.

## What does it do?

This tool works with datasets that follow a structure that is similar to this one:
- `\path\to\train_set\`
  - `.\1.bin`
  - `.\2.bin`
  - `.\3.bin`
  - ...
  - `.\1234.bin`  
- `\path\to\test_set\`
  - `.\1.bin`
  - `.\2.bin`
  - `.\3.bin`
  - ...
  - `.\543.bin` 
- `\path\to\labels.csv`

And will generate a TFRecord dataset like this one:
- `\output\path\train_1.tfrecord`
- `\output\path\train_2.tfrecord`
- ...
- `\output\path\train_25.tfrecord`
- `\output\path\test_1.tfrecord`
- `\output\path\test_2.tfrecord`
- ...
- `\output\path\test_12.tfrecord`


### Relevant details to notice

- It is not necessary that
  [the train set and the test set](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets)
  share the same dataset root directory.
- Each file must contain the data for one single example of the dataset
  (e.g. one image, one sample of binary data...).
- Files can be stored in an arbitrary format (represented here by `*.bin`)
  as long as a function to properly load and preprocess the data is provided. An example of
  such a function has been included to work with single [NumPy](https://numpy.org/) arrays 
  stored as `*.npy` files.
- The labels [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) can have an arbitrary name, but it must be a CSV file with one row per training
  example and two columns: the example ID and its [label](https://en.wikipedia.org/wiki/Labeled_data)
  (which commonly will be the target for [classification](https://en.wikipedia.org/wiki/Statistical_classification)
  in [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) problems). It can include a _header_ first row that will be ignored when specified.
- Each example is identified by its example ID, which is its file name without the
  extension. This ID must be unique among the examples of each subset.
  File names do not need to be consecutive integers, as long as they can be interpreted as
  unique IDs in `string` format.
- Output `*.tfrecord` files can have arbitrary names. In the previous example, they are named after
its subset with a consecutive number for simplicity.
  
## How does it work?
The tool has two parts:
- The dataset handler (`dataset` module), which lists all files in both subsets (train and test) and splits each
subset in partitions with (almost) the same number of examples (if possible). The training
  set is split in a [stratified](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold) way,
  which means that each partition keeps the same label distribution as in the
  original set. This is useful to perform
  [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation)
  or distributed processing.
- The TFRecord handler (`tfrecords` module), which saves dataset _chunks_ to `*.tfrecord` files
and loads datasets from sets of `*.tfrecord` files.
  
Putting the two pieces together it is possible to load both train and test sets, split them into
equally sized _chunks_ (keeping label distribution in train set), and save each _chunk_
in a `*.tfrecord` file. The data stored in TFRecord format contains serialized preprocessed data
from each example, its label (if any) and its example ID.
These `*.tfrecord` files can be later loaded into a `tf.data.Dataset` to handle the
data efficiently using [TensorFlow](https://www.tensorflow.org/guide/data) (maybe [using TPUs](https://www.tensorflow.org/guide/tpu)).

### Examples
To load the file structure of a dataset for posterior splitting, just do
```python
import tfrecord_dataset.dataset as dataset
import tfrecord_dataset.tfrecords as tfrecords
from pathlib import Path  # Used for easy path handling in the example, not needed for tfrecord_dataset.


# Example dataset with *.npy files.
DUMMY_DATASET_PATH = Path('./tfrecord_dataset_test_dummy_dataset')
# Both int and str are supported as the type of the label in the CSV file.
label_type = int
# Used to ignore the first row of the CSV, which is just the names of the columns.
labels_file_has_header = True
# Load dataset structure.
dummy_dataset = dataset.Dataset(
    train_set_dir=str(DUMMY_DATASET_PATH / 'train'),
    test_set_dir=str(DUMMY_DATASET_PATH / 'test'),
    file_format='*.npy',
    train_labels_file=str(DUMMY_DATASET_PATH / 'labels.csv'),
    label_type=label_type,
    train_labels_file_has_header=labels_file_has_header)
```
Next, we can partition the train set in equally distributed splits,
and save the serialized examples of each split in a different `*.tfrecord` file. The `data_preprocessing_function`
will perform the data specific processing needed to load and preprocess each example
from its file.

```python
data_type = None
data_shape = None

# Create 5 splits after shuffling the examples using the specified seed for the random number generator.
for idx, train_set_split in enumerate(dummy_dataset.train_set_in_splits_generator(n_splits=5, shuffle=True, seed=42)):
    # Get the data type and shape from the first example, using the specific data preprocessing function.
    if idx == 0:
        _, data_type, data_shape, _ = tfrecords.npy_data_preprocessor(train_set_split[0, 0])
    # Save the split into a tfrecord file.
    tfrecords.write_dataset_to_file(
        dataset=train_set_split,
        file_path=str(DUMMY_DATASET_PATH / f'train_{idx}.tfrecord'),
        data_preprocessing_function=tfrecords.npy_data_preprocessor
    )
```

The procedure is the same for the test set.
```python
for idx, test_set_split in enumerate(dummy_dataset.test_set_in_splits_generator(n_splits=2, shuffle=True, seed=42)):
    tfrecords.write_dataset_to_file(
        dataset=test_set_split,
        file_path=str(DUMMY_DATASET_PATH / f'test_{idx}.tfrecord'),
        data_preprocessing_function=tfrecords.npy_data_preprocessor
    )
```

The code to load the train set from the TFRecord files and to retrieve each example is also simple:
```python
# Load the train set from the tfrecords.
tf_train_set = tfrecords.load_dataset_from_files(
    file_paths=list(map(str, self.DUMMY_DATASET_PATH.glob('train_*.tfrecord'))),
    data_shape=data_shape, data_type=data_type, label_type=label_type
)

# Read each recovered example.
for recovered in tf_train_set:
    data, label, example_id = recovered['data'], recovered['label'], recovered['example_id']
    # ...Do the TensorFlow processing with these Tensors...
```

The test set is treated similarly, but here there is no label type since test examples
are not labelled.
```python
tf_test_set = tfrecords.load_dataset_from_files(
    file_paths=list(map(str, DUMMY_DATASET_PATH.glob('test_*.tfrecord'))),
    data_shape=data_shape, data_type=data_type, label_type=None
)
for recovered in tf_test_set:
    data, example_id = recovered['data'], recovered['example_id']
    # ...Do the TensorFlow processing with these Tensors...
```

This tool includes a data preprocessing function to work with `*.npy` files containing a single array each one.
To work with different data types, a custom function must be developed, meeting the following signature:

```python
def custom_data_preprocessing_function(example_path):
    """
    Loads and preprocesses an example, given its path.
    This function is to be applied on the first column of each example row of the dataset (file path).
    
    :param example_path: Path to the file.
    :type example_path: str
    :return: A tuple with
        - loaded and preprocessed data in any of the supported formats for serialization (bytes or list of numbers)
        - data type after loading and preprocessing (to allow for serialization and later parsing),
        - data shape after loading and preprocessing but before flattening (to allow for reshaping),
        - and example ID used to identify the example (e.g. filename without extension). The ID is not used
        in the serialization process but it is useful for further usages of the dataset.
    :rtype: tuple[Union[bytes, list[float], list[int]], type, tuple[int], str]
    """
    # ...
```

### More information

For a complete usage example on a real dataset, please see this [Kaggle notebook](https://www.kaggle.com/chusjm/seti-bl-preprocessed-dataset-to-tfrecord).

For more information about the API, please read the automatically generated [documentation](https://chusjm.github.io/tfrecord_dataset/).

## Why?

I developed this tool to work with [datasets](https://www.kaggle.com/datasets)
on [Kaggle](https://www.kaggle.com/) [competitions](https://www.kaggle.com/competitions)
(more specifically, for the [seti-breakthrough-listen competition](https://www.kaggle.com/c/seti-breakthrough-listen)),
in order to export the original dataset to TFRecord format for faster data handling and
enabling [TPU](https://www.kaggle.com/docs/tpu) processing.

I realized this was a repetitive task that appears frequently when working with datasets that have the structure
defined above. My intention was to create a tool that does the job in an easy and straightforward
way, while allowing for a little flexibility.

However, bear in mind that this a personal tool only tested for my problem in question, and probably there
are better tools out there. I release it for
anyone that might find it useful, with no guarantees. If anyone ends up using this tool, please feel free
to send feedback or contribute with fixes, functionality extensions, new data preprocessing functions,
new test cases, etc.

## Requirements
This tool has been developed and tested using Python 3.8 and it should work with higher versions too. It has
the following dependencies:

- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [ScikitLearn](https://scikit-learn.org/stable/)

To use it, install the dependencies and just [clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository)
this repo into your working directory.

## Acknowledgements

I would like to give credit to the following pieces of work, which served me as a
source of inspiration, code snippets and learning resources to develop this code:

- [Chris Deotte's](https://www.kaggle.com/cdeotte) excellent work in this [notebook](https://www.kaggle.com/cdeotte/how-to-create-tfrecords).
- [awsaf49](https://www.kaggle.com/awsaf49) with this [notebook](https://www.kaggle.com/awsaf49/seti-bl-256x256-tfrec-data/),
from which I took the idea of splitting the dataset using stratified K-fold to
ensure that all records maintain the original distribution of targets.
- [xhlulu](https://www.kaggle.com/xhlulu) with these notebooks
([train](https://www.kaggle.com/xhlulu/seti-create-training-tf-records),
[test](https://www.kaggle.com/xhlulu/seti-create-test-tf-records)) which serve also
as a great explanation of the process and describe how to upload the results as a Kaggle Dataset.
- The ways of operating with TFRecords that differ from the aforementioned notebooks
have been taken directly from
[Tensorflow's documentation on TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).

All the pieces of code referenced above are released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license by
their respective authors.

## License
This tool is released under the [Apache 2.0](LICENSE) license.
