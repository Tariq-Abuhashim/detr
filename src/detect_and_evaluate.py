
import math
import sys
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
import matplotlib.pyplot as plt

import time
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from torchvision.transforms import functional as F

from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Append the path to the DETR repository
sys.path.append("../detr")  #Replace with the path to your cloned DETR directory
import util.misc as utils
from models import build_model

# Model setup
class Args:
    def __init__(self):

        self.lr = 1e-4
        self.lr_backbone = 1e-5 #Learning rate for the backbone network.
        self.batch_size = 16
        self.weight_decay = 1e-4
        self.epochs = 300
        self.start_epoch = 0
        self.lr_drop = 200
        self.clip_max_norm = 0.1 #gradient clipping max norm
    
        # Model parameters
        self.frozen_weights = None #"weights/detr-r101-2c7b67e5.pth" #Path to the pretrained model. If set, only the mask head will be trained")
        self.distributed = False

        # * Backbone
        self.backbone = "resnet50" #Backbone architecture (e.g., "resnet50", "resnet101").
        self.dilation = False #Whether or not to use dilated convolutions in the backbone.
        self.position_embedding = 'sine' #Type of position embedding ('sine' or 'learned').

        # * Transformer
        self.enc_layers = 6 #Number of layers in the transformer encoder.
        self.dec_layers = 6 #Number of layers in the transformer decoder.
        self.dim_feedforward = 2048  #Dimension of the feedforward network model.
        self.hidden_dim = 256 #Size of the embeddings (dimension of the transformer)
        self.dropout = 0.1  #Dropout rate applied in the transformer for regularisation
        self.nheads = 8  #Number of attention heads in the multi-head attention mechanism.
        self.num_queries = 100  #Number of query slots.
        #self.pre_norm = 'store_true'  #Number of query slots.
        self.activation = 'relu' #Activation function used in the transformer (e.g., 'relu' or 'gelu').
        self.normalize_before = True #Whether normalization should be done before or after the self-attention/ffn blocks.
        self.pre_norm = False #denotes whether to use normalization before other operations, such as attention or feed-forward layers.

        # * Segmentation
        self.masks = False #Whether or not to use segmentation masks.

        # * Loss
        self.aux_loss = True #indicates whether or not to use auxiliary decoding losses (in addition to the usual decoding loss). These can help improve the training stability of the DETR model.

        # * Matcher
        self.set_cost_class = 1.0 #Class coefficient in the matching cost
        self.set_cost_bbox = 5.0 #L1 box coefficient in the matching cost
        self.set_cost_giou = 2.0 #GIoU box coefficient in the matching cost

        # * Loss coefficients
        self.mask_loss_coef = 1.0
        self.dice_loss_coef = 1.0
        self.bbox_loss_coef = 5.0 #Coefficient for the bounding box loss in the total loss function.
        self.giou_loss_coef = 2.0
        self.eos_coef = 0.1 #Relative classification weight of the no-object class")

        # dataset parameters
        self.num_classes = 2  #Number of object classes (90 for COCO, 1 for window only, 3 for person, car, window).
        self.dataset_file = 'custom_dataset' #'coco'  or 'custom_dataset'#indicate that we are working with the COCO dataset
        self.coco_path = None #'coco'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  #Use GPU if available, otherwise use CPU  
        self.seed = 42
        self.output_dir = './'
        self.resume = None
        self.eval = None

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    start_time = time.time()
    img = transform(im).unsqueeze(0).cuda()
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    #assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    start_time = time.time()
    outputs = model(img)
    #print("--- %s seconds ---" % (time.time() - start_time))

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def read_coco_annotations(annotation_path):
    """
    Reads the COCO-style annotation JSON file and extracts image file paths.
    """
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        image_info = annotations.get("images", [])
        image_paths = [img['file_name'] for img in image_info]
        print(f"Found {len(image_paths)} images in annotation file.")
        #return image_paths
        return annotations, image_info
    except FileNotFoundError:
        print(f"Annotation file not found: {annotation_path}")
        return []
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        return []

def collect_predictions(image_id, scores, boxes):
    """
    Formats predictions for COCO evaluation.
    """
    predictions = []
    for score, (x_min, y_min, w, h) in zip(scores, boxes.tolist()):
        # Convert box from [x, y, w, h] to [x_min, y_min, width, height]
        #x_min, y_min, w, h = box
        cl = score.argmax()
        category_id = score.argmax().item()

        pred = {
            "image_id": image_id,
            "category_id": category_id,  # Assuming single-class detection, adjust as needed
            "bbox": [x_min, y_min, w, h],
            "score": float(cl)
        }
        predictions.append(pred)
    return predictions

def run_detection_with_metrics(annotation_path, image_info, model, device, transform, output_dir):
    """
    Loops through images, performs detection, collects predictions, and computes COCO metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []  # Store predictions in COCO results format
    coco = COCO(annotation_path)  # Load manual annotations from JSON file

    for img_info in tqdm(image_info, desc="Processing images"):
        image_id = img_info['id']
        file_name = img_info['file_name']
        image_path = "/media/mrt/Whale/data/detr/pcc5/person-car/all_images/" + file_name
        #print(image_path)
        try:
            # Load image
            im = Image.open(image_path)
            if im.mode == 'RGBA':
                im = im.convert('RGB')

            # Run detection
            with torch.no_grad():
                scores, boxes = detect(im, model, transform)
                #print(f"Scores shape: {scores.shape}")
                #print(f"Boxes shape: {boxes.shape}")

            if len(scores) == 0 or len(boxes) == 0:
                print(f"No detections for image {image_path}")
                continue

            # Collect predictions
            preds = collect_predictions(image_id, scores, boxes)
            predictions.extend(preds)

            # Optionally save detection results as images with bounding boxes
            #plot_path = output_dir / f"{Path(image_path).stem}_detections.jpg"
            #plot_results(im, scores, boxes, save_path=str(plot_path))

        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    # Save predictions to file
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {predictions_path}")

    # Evaluate using COCO metrics
    if len(predictions) == 0:
        print("No predictions were made; skipping evaluation.")
        return

    evaluate_coco_metrics(coco, predictions)

def evaluate_coco_metrics(coco, predictions):
    """
    Evaluate predictions using COCO metrics.
    """
    if len(predictions) == 0:
        print("No predictions available for evaluation.")
        return

    # Load predictions into COCO format
    coco_preds = coco.loadRes(predictions)
    coco_eval = COCOeval(coco, coco_preds, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def run_detection_on_images(image_paths, model, device, transform, output_dir):
    """
    Loops through the list of image paths, performs detection, and saves results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess the image
            image_path = "/media/mrt/Whale/data/detr/pcc5/person-car/all_images/" + image_path
            im = Image.open(image_path)
            if im.mode == 'RGBA':  # Convert RGBA to RGB if necessary
                im = im.convert('RGB')

            # Run detection
            with torch.no_grad():
                scores, boxes = detect(im, model, transform)
                print("Scores shape:", scores.shape)
                print("Boxes shape:", boxes.shape)

            # Save detection results
            plot_path = output_dir / f"{Path(image_path).stem}_detections.jpg"
            #plot_results(im, scores, boxes, save_path=str(plot_path))
            #print(f"Results saved to {plot_path}")
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

def main(args):
    print("Arguments:", args)

    # Set the device
    device = torch.device(args.device)

    # Build the model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    model.eval()  # Set to evaluation mode

    # Load checkpoint
    try:
        checkpoint = torch.load("/media/mrt/Whale/data/detr/pcc5/person-car/outputs/solider-humvee-checkpoint1099.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Checkpoint file not found: {args.checkpoint_path}")
        return
    except KeyError:
        print("Invalid checkpoint format.")
        return

    # Read annotation file

    #image_paths = read_coco_annotations("/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json")

    _, image_info = read_coco_annotations("/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json")
    if not image_info:
        return

    # Run detection on all images
    #run_detection_on_images(image_paths, model, device, transform, args.output_dir)

    # Run detection and evaluate metrics
    run_detection_with_metrics("/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json", image_info, model, device, transform, args.output_dir)

if __name__ == '__main__':
    args = Args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

