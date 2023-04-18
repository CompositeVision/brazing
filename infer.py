import argparse
import os

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as rt
import tqdm


BATCH_SIZE = 32
PATCH_SHAPE = (128, 128)
PROBA_THRESHOLD = 0.3886

POROSITY_LEVEL = 110

ort_sess = rt.InferenceSession("model.onnx")

def preprocess_image(image: npt.NDArray["uint8"]) -> npt.NDArray["float32"]:
    image = image.astype("float32") / 255.0
    return image

def get_patches(image: npt.NDArray, shape=(128, 128)) -> list[list[np.ndarray]]:
    """Patch indexing reproduces array indexing"""
    patches = []
    for x in range(0, image.shape[0], shape[0]):
        row = []
        for y in range(0, image.shape[1], shape[1]):
            patch = image[x:x+shape[0], y:y+shape[1]]
            row.append(patch)
        patches.append(row)
    return patches

def from_patches(patches: list[list[np.ndarray]]) -> np.ndarray:
    """Assumes pathces indexing that reproduces array indexing"""
    rows = [np.concatenate(row, axis=1) for row in patches]
    image = np.concatenate(rows, axis=0)
    return image


def pad_image(image: npt.NDArray, multiple_of_shape):
    pad_x = (multiple_of_shape[0] - image.shape[0]) % multiple_of_shape[0]
    pad_y = (multiple_of_shape[1] - image.shape[1]) % multiple_of_shape[1]
    return np.pad(image, ((0, pad_x), (0, pad_y)))


def infer_images(images: list[npt.NDArray["uint8"]], batch_size, patch_shape=(128, 128)) -> list[npt.NDArray["uint8"]]:
    patches_per_image = []

    for i, image in enumerate(images):
        image = preprocess_image(image)
        image = pad_image(image, patch_shape)
        patches_per_image.append(get_patches(image, patch_shape))

    all_patches = []
    for image in patches_per_image:
        for row in image:
            for patch in row:
                all_patches.append(patch)
    

    all_preds = []
    for i in tqdm.trange(0, len(all_patches), batch_size, desc="Batch inference...", unit="batch"):
        batch = all_patches[i: i+batch_size]

        current_batch_size = len(batch)
        pad_b = batch_size - current_batch_size
        batch = np.stack(batch, axis=0)                       # bxHxW
        batch = np.pad(batch, ((0, pad_b), (0, 0), (0, 0)))   # BxHxW
        batch = np.expand_dims(batch, axis=1)                 # Bx1xHxW

        batch_pred = ort_sess.run(None, {"input": batch})[0]  # Bx1xHxW
        batch_pred = (batch_pred > PROBA_THRESHOLD).astype("uint8") * 255
        
        batch_pred = np.squeeze(batch_pred)                   # BxHxW
        all_preds.extend([batch_pred[i] for i in range(current_batch_size)])

    iterate_preds = iter(all_preds)
    for image in patches_per_image:
        for row in image:
            for patch in row:
                pred_patch = next(iterate_preds)
                patch[...] = pred_patch.view("uint8")

    segmentation_maps = []
    for image, original_image in zip(patches_per_image, images):
        segmentation_maps.append(from_patches(image)[:original_image.shape[0], :original_image.shape[1]])

    return segmentation_maps

def postprocess_mask(mask: npt.NDArray["uint8"]) -> tuple[npt.NDArray["uint8"], list[npt.NDArray["uint8"]]]:
    MIN_TUBE_AREA = 50_000

    components = []
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_TUBE_AREA:
            cc = np.where(labels == i, 255, 0).astype("uint8")
            # close the holes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(15, 15))
            close = cv2.morphologyEx(cc, cv2.MORPH_CLOSE, kernel, iterations=3)
            components.append(close)

    mask = np.zeros_like(mask)
    for component in components:
        mask += component

    return mask, components


def calculate_porosity(image, mask):
    _, porosity_mask = cv2.threshold(image, POROSITY_LEVEL, 255, cv2.THRESH_BINARY)
    porosity_mask = cv2.bitwise_and(porosity_mask, mask)

    mask_area = np.count_nonzero(mask)
    porosity_area = np.count_nonzero(porosity_mask)

    porosity = porosity_area / mask_area

    return porosity, porosity_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", metavar="image", nargs="+", help="Paths to tiff images")
    parser.add_argument("-m", "--model", default="model.onnx", help="Segmentation model in ONNX format, model.onnx by default")
    parser.add_argument("-d", "--debug", action="store_true", help="Save debug segmentation images")

    args = parser.parse_args()

    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in args.images]
    
    model = rt.InferenceSession(args.model)

    smaps = infer_images(images, BATCH_SIZE, PATCH_SHAPE)
    
    for image_path, image, smap in zip(args.images, images, smaps):
        smap, components = postprocess_mask(smap.astype("uint8"))
            
        print(f"{len(components)} tubes detected in {image_path}")
        
        porosity_mask = np.zeros_like(image)

        print(f"Estimated porosities for {image_path}:")

        for i, component in enumerate(components, start=1):
            porosity, porosity_mask_component = calculate_porosity(image, component)
            porosity_mask = cv2.bitwise_or(porosity_mask, porosity_mask_component)
            print(f"Tube {i}: {porosity * 100:.4f}%")

        # TODO: move lower
        if args.debug:
            os.makedirs("debug_images", exist_ok=True)
            inverted = 255 - image
            masked = cv2.addWeighted(inverted, 0.8, smap, 0.2, 0.0)
            masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
            porosity_mask = cv2.merge((np.zeros_like(porosity_mask), np.zeros_like(porosity_mask), porosity_mask))
            masked = cv2.addWeighted(masked, 0.8, porosity_mask, 0.2, 0)

            # paste component numbers
            for i, component in enumerate(components, start=1):
                xmin, ymin, w, h = cv2.boundingRect(component)
                xmin = max(xmin, 40)
                ymin = max(ymin, 40)
                cv2.putText(masked, str(i), (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)

            cv2.imwrite(os.path.join("debug_images", os.path.splitext(image_path)[0] + ".png"), masked)
