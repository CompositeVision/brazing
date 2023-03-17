import argparse
import logging
import os
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_IMAGES = False
POROSITY_LEVEL = 100


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
        tube_masked = np.where(component, image, 0).astype("uint8")
        xmin, ymin, w, h = cv2.boundingRect(tube_masked)
        crop = tube_masked[ymin : ymin + h, xmin : xmin + w]
        tube_crops.append(crop)

    porosities = []
    with ProcessPoolExecutor(max_workers=len(tube_crops)) as pool:
        for i, crop in enumerate(tube_crops, start=1):
            fut = pool.submit(process_tube, crop, image_path=image_path, tube_id=i)
            porosities.append(fut)

    porosities = [p.result() for p in porosities]

    print(f"Estimated porosities for {image_path}:")
    for i, porosity in enumerate(porosities, start=1):
        print(f"Tube {i}: {porosity * 100:.4f}%")


def draw_rectangle(image: npt.NDArray["uint8"], x0, y0, h, w, angle):
    x1 = int(x0 - np.sin(angle) * w / 2 + np.cos(angle) * h / 2)
    x2 = int(x0 + np.sin(angle) * w / 2 + np.cos(angle) * h / 2)
    x3 = int(x0 + np.sin(angle) * w / 2 - np.cos(angle) * h / 2)
    x4 = int(x0 - np.sin(angle) * w / 2 - np.cos(angle) * h / 2)
    y1 = int(y0 - np.cos(angle) * w / 2 - np.sin(angle) * h / 2)
    y2 = int(y0 + np.cos(angle) * w / 2 - np.sin(angle) * h / 2)
    y3 = int(y0 + np.cos(angle) * w / 2 + np.sin(angle) * h / 2)
    y4 = int(y0 - np.cos(angle) * w / 2 + np.sin(angle) * h / 2)

    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    cv2.fillConvexPoly(image, points, 255)

    return image


def process_tube(image: npt.NDArray["uint8"], **kwargs) -> Optional[float]:
    """Compute porosity level of the brazing area of the tube

    The region of interest on a single tube image is the brazing are.
    In order to locate it, we fit a rotated rectangle over the tube image
    so it covers most of the brazing area and doesn't cover the double walls.
    """

    # 1. Estimate rectangular brazing area based on optimal pixel values

    def rel_to_abs_coords(x0, y0, h, w):
        """Helper function used to map [0, 1] to [0, H(W)]"""
        x0 = image.shape[1] * x0
        y0 = image.shape[0] * y0
        h = image.shape[0] * h
        w = image.shape[1] * w
        return x0, y0, h, w

    lut = np.arange(256)
    lut[0] = -20
    lut[(1 <= lut) & (lut <= 60)] = -30
    lut[(60 <= lut) & (lut <= 99)] = 10
    lut[(99 <= lut) & (lut <= 255)] = -5

    def cost(x0, y0, h, w, angle):
        x0, y0, h, w = rel_to_abs_coords(x0, y0, h, w)
        mask = draw_rectangle(np.zeros_like(image), x0, y0, h, w, angle)

        pixel_cost = cv2.LUT(image, lut)
        pixel_cost = np.where(mask, pixel_cost, 0)

        return -np.sum(pixel_cost)

    def cost_utility(x):
        return cost(*x)

    bounds = opt.Bounds([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, np.deg2rad(180)])
    solution = opt.dual_annealing(cost_utility, bounds)

    if not solution.success:
        logger.warning("Couldn't find brazing area for tube #{}", kwargs["tube_id"])
        return

    # 2. Compute porosity value

    x0, y0, h, w, angle = solution.x
    brazing_mask = draw_rectangle(np.zeros_like(image), *rel_to_abs_coords(x0, y0, h, w), angle)
    porosity_mask = ((np.where(brazing_mask, image, 0) > POROSITY_LEVEL) * 255).astype("uint8")

    whole_area = np.count_nonzero(brazing_mask)
    porosity_area = np.count_nonzero(porosity_mask)

    porosity = porosity_area / whole_area

    # 3. Optionally output debug images

    if DEBUG_IMAGES:
        # Invert image, highlight brazing and porosity areas
        inv_image = (255 - image).astype("uint8")
        inv_image = cv2.addWeighted(inv_image, 0.8, brazing_mask, 0.2, 0)
        inv_image = cv2.cvtColor(inv_image, cv2.COLOR_GRAY2RGB)
        porosity_mask = cv2.merge((np.zeros_like(porosity_mask), np.zeros_like(porosity_mask), porosity_mask))
        inv_image = cv2.addWeighted(inv_image, 0.7, porosity_mask, 0.3, 0)

        save_to = os.path.join("debug_images", kwargs["image_path"], f"tube{kwargs['tube_id']:0>2}.png")
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

        if cv2.imwrite(save_to, inv_image):
            logger.debug(f"Saved debug image {save_to}")
        else:
            logger.debug(f"Failed to save debug image {save_to}")

    return porosity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", metavar="image", nargs="+", help="Paths to tiff images")
    parser.add_argument("-d", "--debug", action="store_true", help="Save debug segmentation images")

    args = parser.parse_args()

    if args.debug:
        DEBUG_IMAGES = True

    for image in args.images:
        process_image(image)
