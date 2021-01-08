import os
import random
from timeit import default_timer as timer
import cv2
import numpy as np


def process_images(backgrounds, output_path, delete_tmp=True):
    """Processes rendered images.

    Adds backgrounds and performs some simple transformations.

    Args:
        backgrounds: List of paths of images used as backgrounds for the final images.
        output_path: Where the rendered images are stored and will be stored to.
        delete_tmp: Whether we should delete the temporary output from rendering after
          the image processing is finished.
    """

    print("Starting processing")
    start = timer()

    bg = cv2.imread(str(backgrounds[0]))

    for tmp_dir in output_path.glob("tmp_*"):
        run, count_per_run = _extract_dir_data(tmp_dir)
        for img_path in tmp_dir.glob("*.png"):

            hand_orig = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

            randomized_hand = _flip_and_rotate(hand_orig)
            bg_crop = _get_random_background_crop(bg)
            processed_image = _overlay(bg_crop, randomized_hand)

            num, open = _extract_img_data(img_path, run, count_per_run)
            cv2.imwrite(str(output_path.joinpath("{}_{}.png".format(num, open))), processed_image)

            if delete_tmp:
                os.remove(img_path)

    if delete_tmp:
        for tmp in output_path.glob("tmp*"):
            if tmp.is_dir():
                os.rmdir(tmp)
            else:
                os.remove(tmp)

    end = timer()
    print("Image processing completed in {}s".format(end - start))


def _get_random_background_crop(img):
    """Randomly crops an image to a 128x128 rectangle.

    Args:
        img: The image to crop.

    Returns: The cropped image.
    """

    # crop and scale
    height, width, channels = img.shape
    min_dim = min(height, width) / 4

    # dont crop smaller than target size
    scaled_size = max(int(random.random() * min_dim), 128)

    left = random.randint(0, width - scaled_size)
    up = random.randint(0, height - scaled_size)

    img = img[up:up + scaled_size, left:left + scaled_size]
    img = cv2.resize(img, (128, 128))

    return img


def _flip_and_rotate(img):
    """Randomizes an image by rotating and flipping.

    Args:
        img: The image used.

    Returns: The randomized image.
    """

    # half of the rendered images get flipped
    if random.random() > .5:
        img = cv2.flip(img, 1)

    # ROTATE_180 = 1
    # ROTATE_90_CLOCKWISE = 0
    # ROTATE_90_COUNTERCLOCKWISE = 2
    # don't rotate with 3
    rot = random.randint(0, 3)
    if rot != 3:
        img = cv2.rotate(img, rotateCode=rot)

    return img


def _translate(img):
    """Randomly translates an image.

    Args:
        img: The image used.

    Returns: The randomized image.
    """

    max_dim = max(img.shape[0], img.shape[1])
    x_trans = random.randint(-max_dim, max_dim)
    y_trans = random.randint(-max_dim, max_dim)

    img2 = np.zeros_like(img)
    if x_trans < 0 and y_trans < 0:
        img2[:x_trans, :y_trans] = img[-x_trans:, -y_trans:]

    elif x_trans >= 0 and y_trans < 0:
        img2[x_trans:, :y_trans] = img[:-x_trans, -y_trans:]

    elif x_trans < 0 and y_trans >= 0:
        img2[:x_trans, y_trans:] = img[-x_trans:, :-y_trans]

    elif x_trans >= 0 and y_trans >= 0:
        img2[x_trans:, y_trans:] = img[:-x_trans, :-y_trans]

    return img


def _overlay(img, overlay_img):
    """Overlays an image over another image.

    Args:
        img: The background image.
        overlay_img: The image being put on top.

    Returns: The background image with the overlay.
    """

    alpha_overlay = overlay_img[:, :, 3] / 255.0
    alpha_img = 1.0 - alpha_overlay

    for c in range(0, 3):
        img[:, :, c] = (alpha_overlay * overlay_img[:, :, c] +
                        alpha_img * img[:, :, c])

    return img


def _extract_img_data(path, run, count_per_run):
    """

    Expects the image name to be encoded like: "<img_num>_<label>.png".
    Calculates the number of this image across all runs.

    Args:
        path: Path of the image.
        run: In which run index this image was rendered.
        count_per_run: How many images are rendered per run.

    Returns:
        The number of this image across all runs and the corresponding label.
    """

    split = path.stem.split("_")
    num = int(split[1])
    num = run * count_per_run + num

    label = int(split[0])

    return num, label


def _extract_dir_data(path):
    """

    Expects the directory to be named like: "tmp_<run>_with_<images in run>"

    Args:
        path: Path of the directory.

    Returns:
        The run index and image count per run.
    """

    split = path.stem.split("_")
    run = int(split[1])
    count_per_run = int(split[3])

    return run, count_per_run
