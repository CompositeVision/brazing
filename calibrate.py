from model import get_model, get_transforms_val
from dataset import Dataset
import torch.utils.data
import cv2
import numpy as np
import torchvision.transforms.functional as F
import glob
import torch.onnx
import tqdm
import plotly.express as px
from sklearn.metrics import precision_recall_curve


# THRESH = 0.394
# THRESH = 0.58
THRESH = 0.3886

model = get_model()
with open("model.bin", "rb") as f:
    model.load_state_dict(torch.load(f))

torch.onnx.export(model, torch.zeros((32, 1, 128, 128)), "model.onnx", verbose=True, input_names=["input"], output_names=["segmentation"])
exit()

model = model.cuda()
model.eval()

images = glob.glob("processed/images/*")
masks = glob.glob("processed/masks/*")

transforms = get_transforms_val()
dataset = Dataset(images, masks, transforms=get_transforms_val())
loader = torch.utils.data.DataLoader(dataset, 32)

def thresholds():
    gts = []
    predictions = []

    for i, batch in tqdm.tqdm(enumerate(loader)):
        image_b, mask_b = batch
        pred_b = model(image_b.cuda()).cpu().sigmoid()

        image_b = image_b.squeeze().detach().numpy()
        mask_b = mask_b.squeeze().detach().numpy()
        pred_b = pred_b.squeeze().detach().numpy()
        
        gts.append(mask_b.flat)
        predictions.append(pred_b.flat)

    gts = np.concatenate(gts)
    predictions = np.concatenate(predictions)

    indices = np.random.randint(0, len(gts), size=1000)
    gts = gts[indices]
    predictions = predictions[indices]

    precision, recall, thresholds = precision_recall_curve(gts, predictions)
    
    plot = px.line(x=precision[:-1], y=recall[:-1], hover_name=thresholds)
    plot.write_html("pr.html")

def images():

    for i, batch in tqdm.tqdm(enumerate(loader)):
        image_b, mask_b = batch
        pred_b = model(image_b.cuda()).cpu().sigmoid()
        pred_b = (pred_b > THRESH).float()

        image_b = image_b.squeeze().detach().numpy()
        mask_b = mask_b.squeeze().detach().numpy()
        pred_b = pred_b.squeeze().detach().numpy()

        image_b = (image_b * 255).astype("uint8")
        mask_b = (mask_b * 255).astype("uint8")
        pred_b = (pred_b * 255).astype("uint8")

        masked_gts = []
        masked_preds = []
        for image, mask, pred in zip(image_b, mask_b, pred_b):
            masked_gts.append(cv2.addWeighted(image, 0.8, mask, 0.2, 0))
            masked_preds.append(cv2.addWeighted(image, 0.8, pred, 0.2, 0))

        masked_gts = np.concatenate(masked_gts, axis=0)
        masked_preds = np.concatenate(masked_preds, axis=0)

        full_pic = np.concatenate([masked_gts, masked_preds], axis=1)
        cv2.imwrite(f"preds/b{i:0>3}.png", full_pic)

# thresholds()
images()