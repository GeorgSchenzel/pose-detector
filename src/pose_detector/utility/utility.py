import matplotlib.pyplot as plt
from pose_detector.training.training import _create_dataset


def visualize_dataset(images_directory, save_path):

    train_dataset, val_dataset = _create_dataset(images_directory)

    fig = plt.figure(figsize=(10, 10))
    plt.margins(y=10)
    fig.suptitle('Sample of the arms dataset', fontsize=16, weight="bold")
    for images, labels in train_dataset.take(1):
        labels = labels.numpy()
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(str(labels[i]) + "% open")
            plt.axis("off")

    plt.savefig(save_path)
