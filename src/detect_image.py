
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
        self.backbone = "resnet101" #Backbone architecture (e.g., "resnet50", "resnet101").
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
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    #assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    start_time = time.time()
    outputs = model(img)
    print("--- %s seconds ---" % (time.time() - start_time))

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.4

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=1))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            #text = f'{p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=8)
            #        bbox=dict(facecolor='yellow', alpha=0.25))
        plt.axis('off')
        plt.savefig('results.png')  # Save the plot to a file
        plt.close()

# weapon classes
CLASSES = [
    'N/A','person', 'car', 'N/A'
]
#CLASSES = [
#    'N/A','person', 'car', 'window', 'N/A'
#]
#CLASSES = [
#    'N/A', 'window', 'N/A'
#]

# COCO classes
#CLASSES = [
#    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#    'toothbrush'
#]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def main(args):

    print(args)
    device = torch.device(args.device)

    # the model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # load checkpoint
    checkpoint = torch.load("/home/mrt/src/detr/weights/pcc5/person-car/20250226-res101/checkpoint0099.pth", map_location='cpu')
    #checkpoint = torch.load("../weights/detr-r101-2c7b67e5.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # detect
    #filename = '../../samples/fov.png'
    filename = '/media/mrt/Whale/data/detr/pcc5/person-car/all_images/Screenshot_2025-02-03_12-17-26.png'
    im = Image.open(filename)
    
    # Check if the image has an alpha channel
    if im.mode == 'RGBA':
        # Convert the image to RGB mode
        im = im.convert('RGB')
    
    with torch.no_grad():
        scores, boxes = detect(im, model, transform)
        print(f"Scores shape: {scores.shape}")
        print(f"Boxes shape: {boxes.shape}")

    #print(scores)
    #print(boxes)

    # plot
    plot_results(im, scores, boxes)

if __name__ == '__main__':
    args = Args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
