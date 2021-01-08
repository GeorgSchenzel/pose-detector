import os
from datetime import datetime

import tensorflow as tf
from pose_detector.training.CustomCallback import CustomCallback

from tensorflow.keras import layers
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.models import Sequential
from classification_models.tfkeras import Classifiers


def run(images_directory, save_path, base_model_name="resnet18"):
    """Trains a CNN using a previously created dataset and transfer learning.

    Args:
        images_directory: Directory where the dataset is be stored.
        save_path: Directory where the final model will be stored.
        base_model_name: Name of the model architecture to use as a baseline.
    """

    img_size = (128, 128)
    img_shape = img_size + (3,)

    train_dataset, val_dataset = _create_dataset(images_directory)

    print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_dataset))

    Model = Classifiers.get(base_model_name)[0]
    base_model = Model(input_shape=img_shape, weights='imagenet', include_top=False)
    model = create_model(base_model)

    log_dir = "logs/fit/" + "PoseDetection_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

    test_callback = CustomCallback(log_dir, val_dataset)


    model.fit(train_dataset,
              epochs=20,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback, test_callback])

    # model must actually be in a subdir indicating the version
    save_path = save_path / "1"
    model.save(str(save_path.resolve()))


def _freeze(base_model):
    """Freeze some layers of the model.

    Args:
        base_model: The model to freeze layers on.

    Returns:
        The model with frozen layers.
    """

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    conv_layers = [layer for layer in base_model.layers if "conv" in layer.name]
    for layer in conv_layers[:]:
        layer.trainable = True

    return base_model


def create_model(base_model):
    """Creates the complete model.

    Adds layers to the base model and uses "mean squared error" as the loss function.

    Returns:
        The complete model.
    """

    base_model = _freeze(base_model)
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.25))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mean_absolute_error'])

    return model


def _create_dataset(data_dir):
    """Creates a dataset from all images in a directory.

    The images are expected to have a name of the format: "<img_num>_<label>.png".
    The numerical value stored at "label" will be used as the label for this image.
    A 80/20 training/validation split is used.

    Args:
        data_dir: The directory containing all images.

    Returns:
        The dataset split into training and validation.
    """

    image_count = sum(1 for _ in data_dir.glob("*.png"))

    list_ds = tf.data.Dataset.list_files(str(data_dir / "*.png"), shuffle=True)

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    # Set "num_parallel_calls" so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(_process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_process_path, num_parallel_calls=AUTOTUNE)

    train_ds = _configure_for_performance(train_ds, shuffle=True)
    val_ds = _configure_for_performance(val_ds)

    return train_ds, val_ds


def _process_path(file_path):
    """Maps a path to an decoded image and a label.

    Args:
        file_path: The path of the image.

    Returns:
        A tuple of (image, label)
    """

    label = _extract_label(file_path)
    img = _load_img(file_path)

    return img, label


def _extract_label(file_path):
    """Extract the label from an image path.
    """

    # convert the path to a list of path components
    label = tf.strings.split(file_path, os.path.sep)[-1]

    # expecting file name of the format <num>_<training>.png
    label = tf.strings.split(label, "_")[1]

    # remove file extension
    label = tf.strings.split(label, ".")[0]
    label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    return label


def _load_img(file_path):
    """Loads and decodes an image.
    """

    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)

    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)

    return img


def _configure_for_performance(ds, shuffle=False):
    """Enables caching and prefetching on a dataset.

    Args:
        shuffle: If the dataset should be shuffled each iteration.
    """

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    ds = ds.batch(64)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
