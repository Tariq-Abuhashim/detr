import sys
import argparse
from pathlib import Path

import torch
import torchvision

# Append the path to the DETR repository
sys.path.append("../detr")  # Replace with the path to your cloned DETR directory
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
        self.num_classes = 1  #Number of object classes (e.g., 91 for COCO, 3 for car,person,window).
        self.dataset_file = 'custom_dataset' #'coco'  or 'custom_dataset'#indicate that we are working with the COCO dataset
        self.coco_path = None #'coco'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  #Use GPU if available, otherwise use CPU  
        self.seed = 42
        self.output_dir = './'
        self.resume = None
        self.eval = None

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Tracks windows in images and computes their normals.')
        parser.add_argument('--w', type=int, default=1273, help='Width of the input images') #vulcan 1274, vulcan-mono-slam 1273
        parser.add_argument('--h', type=int, default=800, help='Height of the input images')
        parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
        parser.add_argument('--checkpoint', type=str, default="../weights/window_res101_cp1399.pth", help='Path to the trained model checkpoint')
        return parser.parse_args()


def main(args):

    # the model
    model, criterion, postprocessors = build_model(args)
    
    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # Dummy input to the model
    x = torch.randn(1, 3, args.h, args.w, requires_grad=True)

    # Export the model to ONNX format
    # https://pytorch.org/docs/stable/onnx.html
    torch.onnx.export(
        model,               # model being run
        x,                   # model input (or a tuple for multiple inputs)
        "model.onnx",        # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],   # the model's input names
        output_names=['output'], # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

if __name__ == "__main__":
    inargs = Args.parse_args()

    args = Args()
    args.w = inargs.w
    args.h = inargs.h
    args.checkpoint = inargs.checkpoint
    args.num_classes = inargs.num_classes #over-writes number of classes using user input
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
