import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Used to create a custom histogram of errors in the validation stage.
    """

    def __init__(self, log_dir, pred_data):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir + "/prediction")
        self.pred_data = pred_data

    def on_epoch_end(self, epoch, logs=None):
        def iterate(ds):
            for _, labels in ds:
                for label in labels:
                    yield label

        pred = self.model.predict(self.pred_data).ravel()
        truth = np.fromiter(iterate(self.pred_data), float)
        error = pred - truth

        with self.writer.as_default():
            plt.figure(dpi=200)
            plt.hist(error, range=(-50, 50), bins=100)
            plt.xlabel('Error')
            plt.ylabel('Count')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            plt.clf()

            tf.summary.image("Error", image, max_outputs=1, step=epoch)

            tf.summary.histogram("Prediction", pred, step=epoch)