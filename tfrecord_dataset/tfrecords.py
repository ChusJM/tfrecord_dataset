import tensorflow as tf
import numpy as np
from pathlib import Path


# See https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """
    Returns a bytes_list from a string / bytes. A string / bytes is considered a scalar by Tensorflow.

    :param value: A String or a Byte list.
    :type value: Union[str, bytes]
    :return: A bytes list feature.
    :rtype: tensorflow.train.Feature
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value_list):
    """
    Returns a float_list from a list of float / double.
    :param value_list: List of float values
    :type value_list: list[float]
    :return: A float list feature
    :rtype: tensorflow.train.Feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _int64_feature(value_list):
    """
    Returns an int64_list from a list of bool / enum / int / uint.
    :param value_list: List of integer values
    :type value_list: list[int]
    :return: A int64 list feature
    :rtype: tensorflow.train.Feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def _auto_feature(value, data_type):
    """
    Returns a feature list automatically selecting the correct data type as specified by the parameter.

    :param value: Value to be converted to feature. It must be a str or bytes, a list of float (or equivalent type)
        or a list of int (or equivalent type).
    :type value: Union[Union[str, byte], list[float], list[int]]
    :param data_type: Type of the data (type of value in case of str of bytes, or type of the elements of the list in
        the case of list of float and list of int).
    :type data_type: type
    :raise ValueError: When the specified data type is not recognised as a type that can be converted to feature.
    :return: A list feature of the correct type.
    :rtype: tensorflow.train.Feature
    """
    if data_type in (np.float32, np.float64, float):
        feature = _float_feature(value)
    elif data_type in (int, bool, np.uint32, np.uint64, np.int64):
        feature = _int64_feature(value)
    elif data_type in (str, bytes):
        # The feature must be bytes, so it is necessary to encode the string as a bytes string.
        # Default parameters are used (utf-8 encoding).
        if data_type is str:
            value = value.encode()
        feature = _bytes_feature(value)
    else:
        raise ValueError(f'Unrecognised data_type {data_type}')

    return feature


def _auto_tf_type(data_type):
    """
    Returns the tensorflow data type that corresponds to an the data type specified by the parameter.
    The returned type is the type that must be used to parse a feature of 'data_type' that has been converted
    with _auto_feature() (see above).

    :param data_type: Python native type or numpy type that is to be converted to tensorflow type.
    :type data_type: type
    :raise ValueError: When the specified data type is not recognised as a type that has a tensorflow equivalent.
    :return: Tensorflow type that corresponds to the specified data type.
    :rtype: tensorflow.python.framework.dtypes.DType
    """
    if data_type in (np.float32, np.float64, float):
        tf_type = tf.float32
    elif data_type in (int, bool, np.uint32, np.uint64, np.int64):
        tf_type = tf.int64
    elif data_type in (str, bytes):
        tf_type = tf.string
    else:
        raise ValueError(f'Unrecognised data_type {data_type}')

    return tf_type


def npy_data_preprocessor(example_path):
    """
    Loads and preprocesses an example, given its path. The example must be a *.npy file that contains an array
    that can (and will) be casted to float32. It also returns the data type and the shape before flattening the array,
    to allow serialization and later reconstruction, and a string to serve as example identifier (e.g. the file name).

    :param example_path: Path to the *.npy file.
    :type example_path: str
    :return: Flattened numpy array of type np.float32 (it must be compatible with a list of float), data type (to allow
        for serialization and later parsing) and data shape before flattening (to allow for unflattening).
    :rtype: tuple[numpy.ndarray, type, tuple[int], str]
    """
    data = np.load(example_path).astype(np.float32)
    return data.ravel(), data.dtype, data.shape, Path(example_path).stem


def _serialize_example(
        example,
        data_preprocessing_function=npy_data_preprocessor,
        label_type=int):
    """
    Serializes a single example.
    :param example: An example consists of a file path (str) and optionally a label (str).
    :type example: tensorflow.Tensor
    :param data_preprocessing_function: Function to be applied on the first column of each example (file path), which
        is supposed to load the example data and perform any additional preprocessing. It must return the data
        in any of the supported formats for serialization (str, bytes or list of numbers), along with the
        data type (after loading and preprocessing) and the original data shape (after loading and preprocessing but
        before flattening). It will also return an example ID, used to identify the example (for instance, the original
        file name). This ID is not used internally but it is useful for further usages of the dataset.
        Optional, default: npy_data_preprocessor (see above).
    :type data_preprocessing_function: (str) -> tuple[Union[str, bytes, list[float], list[int]], type, tuple[int], str]
    :param label_type: Type of the label, usually an integer. If there is no label, it is ignored.
        Optional, default: int
    :type label_type: type
    :raises ValueError: If the label type conversion is not possible.
    :raises OSError: If there is a problem reading the example data from the file path.

    :return: A binary string representation of the example.
    :rtype: bytes
    """
    # Create a dictionary mapping the feature name to its Feature of the right type, obtained from the
    # original values.
    feature = {}

    # Recover path and label from the example.
    # To get the path as a str, it is necessary to get the numpy() value of the Tensor, and to decode the bytes
    # string with default parameters (utf-8).
    path = example[0].numpy().decode()
    if len(example) > 1:
        # The same applies to the label.
        label = example[1].numpy().decode()
        # Parse label.
        label = label_type(label)
        # If the label is not specified as a str or bytes, it is a scalar int or float. In that case
        # it must be passed as a list for serialization.
        if label_type not in (str, bytes):
            label = [label]
        # This feature is to be used as the target for classification.
        feature['label'] = _auto_feature(label, label_type)

    # Load the data pointed by the path
    data, data_type, _, example_id = data_preprocessing_function(path)

    # This feature is the data from the example.
    feature['data'] = _auto_feature(data, data_type)
    # This feature is an unique ID that can be useful when using the dataset. It is always a string.
    feature['example_id'] = _auto_feature(example_id, str)

    # Create the Features message.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _tf_serialize_example(
        example,
        data_preprocessing_function=npy_data_preprocessor,
        label_type=int):
    """
    Tensorflow wrapper for "serialize_example" to be used with the tf.data interface.

    :param example: An example consists of a file path (str) and optionally a label (str).
    :type example: tensorflow.Tensor
    :param data_preprocessing_function: Function to be applied on the first column of each example (file path), which
        is supposed to load the example data and perform any additional preprocessing. It must return the data
        in any of the supported formats for serialization (str, bytes or list of numbers), along with the
        data type (after loading and preprocessing) and the original data shape (after loading and preprocessing but
        before flattening). It will also return an example ID, used to identify the example (for instance, the original
        file name). This ID is not used internally but it is useful for further usages of the dataset.
        Optional, default: npy_data_preprocessor (see above).
    :type data_preprocessing_function: (str) -> tuple[Union[str, bytes, list[float], list[int]], type, tuple[int], str]
    :param label_type: Type of the label, usually an integer. If there is no label, it is ignored.
        Optional, default: int
    :type label_type: type
    :raises ValueError: If the label type conversion is not possible.
    :raises OSError: If there is a problem reading the example data from the file path.

    :return: A binary string representation of the example.
    :rtype: tensorflow.Tensor
    """
    tf_string = tf.py_function(
        lambda ex: _serialize_example(ex, data_preprocessing_function, label_type),
        (example,),
        tf.string
    )
    return tf.reshape(tf_string, ())


def write_dataset_to_file(
        dataset,
        file_path,
        data_preprocessing_function=npy_data_preprocessor):
    """
    Writes a dataset to a file in TFRecord format. The dataset must be an array with one row per example and at least
    one column with the file path to each example data, and one optional column for its label.
    The data for each example is read and saved serialized into the TFRecord file along with its label (if any).

    :param dataset: Input dataset array.
    :type dataset: numpy.ndarray
    :param file_path: Path where the output file will be placed. It shall have *.tfrecord extension.
    :type file_path: str
    :param data_preprocessing_function: Function to be applied on the first column of each example (file path), which
        is supposed to load the example data and perform any additional preprocessing. It must return the data
        in any of the supported formats for serialization (str, bytes or list of numbers), along with the
        data type (after loading and preprocessing) and the original data shape (after loading and preprocessing but
        before flattening). It will also return an example ID, used to identify the example (for instance, the original
        file name). This ID is not used internally but it is useful for further usages of the dataset.
        Optional, default: npy_data_preprocessor (see above).
    :type data_preprocessing_function: (str) -> tuple[Union[str, bytes, list[float], list[int]], type, tuple[int], str]
    :raises ValueError: If the label type conversion is not possible.
    :raises OSError: If there is a problem reading the example data from the file path.
    """
    # If the labels column is present, guess the data type of the labels.
    if len(dataset) > 0 and dataset.shape[1] > 1:
        label_type = type(dataset[0, 1])
    else:
        label_type = None
    # Prepare to write to a TFRecord file.
    writer = tf.data.experimental.TFRecordWriter(file_path)
    # Serialize dataset examples. The dataset must have all columns of the same type. They are converted to
    # str since the paths will be str, and the label type is passed as a parameter to recover the label to its
    # original type inside the function.
    serialized_dataset = tf.data.Dataset.from_tensor_slices(dataset.astype(str))\
        .map(lambda ex: _tf_serialize_example(ex, data_preprocessing_function, label_type),
             num_parallel_calls=tf.data.AUTOTUNE)
    # Write to file.
    writer.write(serialized_dataset)


def _parse_example(example_proto, data_shape, data_type=np.float32, label_type=int):
    """
    Parses a single serialized example.

    :param example_proto: Serialized example as a binary string.
    :type example_proto: tensorflow.string
    :param data_shape: Original shape of the example data.
    :type data_shape: tuple[int]
    :param data_type: Type of each data point in the example or the example itself if it is bytes or str.
        Optional, default: np.float32
    :type data_type: type
    :param label_type: Type of the label (if any). If there is no label, this must be specified by setting this
        parameter to None. Optional, default: int.
    :type label_type: type
    :return: Dictionary with 'data' and 'example_id' fields and an optional 'label' field.
    :rtype: tuple[tensorflow.Tensor]
    """
    # Basic example description with data field and ID.
    feature_description = {
        'data': tf.io.FixedLenFeature(data_shape, _auto_tf_type(data_type)),
        'example_id': tf.io.FixedLenFeature([], _auto_tf_type(str))
    }
    # Add the label field if a label type is specified.
    if label_type is not None:
        feature_description['label'] = tf.io.FixedLenFeature([], _auto_tf_type(label_type))

    # Parse the example.
    example = tf.io.parse_example(example_proto, feature_description)

    return example


def load_dataset_from_files(file_paths, data_shape, data_type=np.float32, label_type=int):
    """
    Loads a tensorflow dataset from a list of *.tfrecord files with serialized examples.
    This function is the counterpart for write_dataset_to_file() (see above) and its only valid for TFRecords that
    follow the serialization format of that function (two fixed length features, 'data' (with shape = data_shape)
    and 'label' (scalar value)).

    :param file_paths: List of paths to the *.tfrecord files that contain the serialized examples.
    :type file_paths: list[str]
    :param data_shape: Original shape of the example data. Since they are serialized, this metadata is needed
        to reconstruct the tensor.
    :type data_shape: tuple[int]
    :param data_type: Type of each data point in the example or the example itself if it is bytes or str, before
        serialization. It is needed to parse the example. Optional, default: np.float32
    :type data_type: type
    :param label_type: Type of the label (if any). If there is no label, this must be specified by setting this
        parameter to None. Optional, default: int.
    :type label_type: type
    :return: Tensorflow dataset of parsed examples (as tuples of tensors with example data and optionally a label).
    :rtype: tf.data.Dataset
    """
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    return raw_dataset.map(
        lambda raw_ex: _parse_example(raw_ex, data_shape, data_type, label_type),
        num_parallel_calls=tf.data.AUTOTUNE
    )
