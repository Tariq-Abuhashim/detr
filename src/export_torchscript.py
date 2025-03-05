import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from build_model import Args
# Append the path to the DETR repository
sys.path.append("../detr")  # Replace with the path to your cloned DETR directory
from models import build_model


# Wrapper class that converts output of DETR from dictionary to tuple
class FlatModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, *args):
        outputs = self.model.forward(*args)
        flat_outputs = (outputs['pred_logits'], outputs['pred_boxes'])
        return flat_outputs
    
def main(args):

    # Dummy input to the model for tracing
    x = torch.randn(1, 3, args.h, args.w, requires_grad=False)

    # save the Torchscript model
    model_save_path = Path(args.model_save_path)
    if not model_save_path.is_file():
        model, criterion, postprocessors = build_model(args)
        model.eval()
        flat_model = FlatModel(model) # wrap model for compatibility with ONNX
        prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        traced_model = torch.jit.trace(flat_model, x, strict=False)
        torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)
        traced_model.save(args.model_save_path)
        print("Model saved at " + str(model_save_path), file=sys.stderr, flush=True)
    else:
        print("Model already exists at " + str(model_save_path), file=sys.stderr, flush=True)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Exports DETR model as Torchscript')
    parser.add_argument('--w', type=int, default=1274, help='Width of the input images')
    parser.add_argument('--h', type=int, default=800, help='Height of the input images')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--model_save_path', type=str, default=Path.home() / "src/detr/detr/outputs/detr_model_flattened_trace.pth")
    inargs = parser.parse_args()

    args = Args()
    args.w = inargs.w
    args.h = inargs.h
    args.model_save_path = inargs.model_save_path
    args.num_classes = inargs.num_classes #over-writes number of classes using user input
    main(args)