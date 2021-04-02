import cv2
import numpy as np
import os


def convert(fname: str) -> None:
    """
    creates a mask with class id values as pixel values
    from the color ground truth mask image.

    Args:
        fname (str): fname

    Returns:
        None:
    """
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # label_id_mask file name
    label_id_fname = fname.replace("_gtFine_color", "_gtFine_labelIds")

    color_mask = np.zeros(img.shape, dtype=np.uint8)
    color_mask[np.where((img == [234, 30, 39]).all(axis=2))] = np.array([0])
    color_mask[np.where((img == [101, 190, 110]).all(axis=2))] = np.array([1])
    color_mask[np.where((img == [24, 92, 163]).all(axis=2))] = np.array([2])
    color_mask[np.where((img == [224, 212, 28]).all(axis=2))] = np.array([3])

    color_mask = color_mask[:, :, 0]
    cv2.imwrite(label_id_fname, color_mask)


if __name__ == "__main__":
    convert("./assets/data/gtFine_trainvaltest/gtFine/train/a/1_gtFine_color.png")
    convert("./assets/data/gtFine_trainvaltest/gtFine/train/a/2_gtFine_color.png")
    convert("./assets/data/gtFine_trainvaltest/gtFine/train/a/3_gtFine_color.png")
    convert("./assets/data/gtFine_trainvaltest/gtFine/train/a/4_gtFine_color.png")
    convert("./assets/data/gtFine_trainvaltest/gtFine/val/b/5_gtFine_color.png")
    convert("./assets/data/gtFine_trainvaltest/gtFine/val/b/6_gtFine_color.png")
