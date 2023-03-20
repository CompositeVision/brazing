import argparse
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_IMAGES = False
POROSITY_LEVEL = 110


def process_image(image_path: str) -> None:
    """Perform multi-stage X-ray image analysis"""

    # Step 1: load image

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f"Couldn't read {image_path}. Check file path")
        return

    # Step 2: Segment foreground with OTSU thresholding

    _, binary_image = cv2.threshold(image, 0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Step 3: Separate individual tubes by selecting largest components in binary image

    MIN_TUBE_AREA = 50_000

    components = []
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_TUBE_AREA:
            cc = np.where(labels == i, 255, 0).astype("uint8")
            # close the holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
            close = cv2.morphologyEx(cc, cv2.MORPH_CLOSE, kernel, iterations=3)
            components.append(close)

    logger.info(f"{len(components)} tubes detected in {image_path}")

    # Step 4: Crop and process individual tubes

    tube_crops = []
    for component in components:
        tube_masked = cv2.bitwise_and(component, image)
        xmin, ymin, w, h = cv2.boundingRect(tube_masked)
        crop = tube_masked[ymin : ymin + h, xmin : xmin + w]
        tube_crops.append(crop)

    porosities = [process_tube(tube_crop) for tube_crop in tube_crops]

    # Step 5: Output results

    if DEBUG_IMAGES:
        tube_crop_debug_images = [image for (porosity, image) in porosities]
        debug_image = np.concatenate(tube_crop_debug_images, axis=1)
        save_to = os.path.join("debug_images", image_path)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

        if cv2.imwrite(save_to, debug_image):
            logger.debug(f"Saved debug image {save_to}")
        else:
            logger.debug(f"Failed to save debug image {save_to}")

    print(f"Estimated porosities for {image_path}:")
    for i, (porosity, _) in enumerate(porosities, start=1):
        print(f"Tube {i}: {porosity * 100:.4f}%")


def fill_holes(image: npt.NDArray["uint8"]) -> npt.NDArray["uint8"]:
    image = np.pad(image, 1)
    cv2.floodFill(image, None, (0, 0), 1)
    image = image[1:-1, 1:-1]
    image = np.where(image != 1, 255, 0).astype("uint8")
    return image


def leave_largest_component(image) -> npt.NDArray["uint8"]:
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, numLabels)], key=lambda x: x[1])
    image = np.where(labels == max_label, 255, 0).astype("uint8")
    return image


def process_tube(image: npt.NDArray["uint8"]) -> tuple[float, Optional[npt.NDArray["uint8"]]]:
    """Compute porosity level of the brazing area of the tube"""

    image_equalized = cv2.equalizeHist(image)

    # Cut off dark background (inner double wall)
    _, mask = cv2.threshold(image_equalized, 45, 255, type=cv2.THRESH_TOZERO)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Cut off bright outside-of-brazing area
    _, mask = cv2.threshold(mask, 180, 255, type=cv2.THRESH_TOZERO_INV)
    # Cut vertical adjacent (outer double walls)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 3)), iterations=3)

    mask = fill_holes(mask)

    brazing_mask = leave_largest_component(mask)

    _, porosity_mask = cv2.threshold(image, POROSITY_LEVEL, 255, cv2.THRESH_BINARY)
    porosity_mask = cv2.bitwise_and(porosity_mask, brazing_mask)

    whole_area = np.count_nonzero(brazing_mask)
    porosity_area = np.count_nonzero(porosity_mask)

    porosity = porosity_area / whole_area

    # Optionally output debug images

    debug_image = None

    if DEBUG_IMAGES:
        # Invert image, highlight brazing and porosity areas
        inv_image = (255 - image).astype("uint8")
        inv_image = cv2.addWeighted(inv_image, 0.8, brazing_mask, 0.2, 0)
        inv_image = cv2.cvtColor(inv_image, cv2.COLOR_GRAY2RGB)
        porosity_mask = cv2.merge((np.zeros_like(porosity_mask), np.zeros_like(porosity_mask), porosity_mask))
        inv_image = cv2.addWeighted(inv_image, 0.8, porosity_mask, 0.2, 0)
        debug_image = inv_image

    return porosity, debug_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", metavar="image", nargs="+", help="Paths to tiff images")
    parser.add_argument("-d", "--debug", action="store_true", help="Save debug segmentation images")

    args = parser.parse_args()

    if args.debug:
        DEBUG_IMAGES = True

    for image in args.images:
        process_image(image)
