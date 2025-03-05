import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import tensorrt as trt
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
tensorrt_version = trt.__version__

# Colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class TensorRTInference:

    # FIXME choose number of classes
    CLASSES = [
        'N/A', 'person', 'car', 'window', 'N/A'
    ]
    #CLASSES = [
    #    'N/A', 'window', 'N/A'
    #]

    def __init__(self, engine_path="/home/mrt/src/detr/src/model.engine", num_classes=1):
        cuda.init()  # Ensure CUDA driver is initialized
        self.device = cuda.Device(0)  # Assuming use of the first GPU device
        self.context = self.device.make_context()  # Create and make this context current

        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_path)
        self.trt_context = self.engine.create_execution_context()
        if tensorrt_version.startswith('8'):
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers_trt8()
        elif tensorrt_version.startswith('10'):
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers_trt10()

        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.num_classes=num_classes

    def load_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def cleanup(self):
        del self.trt_context
        del self.engine
        del self.runtime
        self.context.pop()
        self.context.detach()

    def allocate_buffers_trt10(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # allocate hose and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings.
            # When cast to int, it's a linear index into the context's memory (like memory address).
            bindings.append(int(device_mem))
            
            # Append to the appropriate input/output list.
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})

        return inputs, outputs, bindings, stream

    def allocate_buffers_trt8(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        #for binding in range(self.engine.num_bindings):
        #    binding_shape = self.context.get_binding_shape(binding)
        #    size = trt.volume(binding_shape) * self.engine.max_batch_size if self.engine.has_implicit_batch_dimension else trt.volume(binding_shape)
        #    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings.
            # When cast to int, it's a linear index into the context's memory (like memory address).
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        b = np.zeros_like(x)
        b[:, 0] = x_c - 0.5 * w  # x0
        b[:, 1] = y_c - 0.5 * h  # y0
        b[:, 2] = x_c + 0.5 * w  # x1
        b[:, 3] = y_c + 0.5 * h  # y1
        return b

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b[:, [0, 2]] = b[:, [0, 2]] * img_w  # Scale x0 and x1
        b[:, [1, 3]] = b[:, [1, 3]] * img_h  # Scale y0 and y1
        return b

    def process_outputs(self, pred_logits, pred_boxes, image_size, conf_threshold=0.5):
        # Assuming pred_logits and pred_boxes are NumPy arrays

        # Apply softmax to convert logits into probabilities
        probas = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=-1, keepdims=True)

        # Filter out low confidence predictions
        max_probas = np.max(probas[:, :-1], axis=-1)  # Exclude the last class (usually 'no-object' class)
        keep = max_probas > conf_threshold

        filtered_probas = probas[keep]
        filtered_boxes = pred_boxes[keep]

        # Rescale bounding boxes
        bboxes_scaled = self.rescale_bboxes(filtered_boxes, image_size)

        return filtered_probas, bboxes_scaled
    
    def plot_results(self, pil_img, prob, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=1))
            cl = p.argmax()
            #text = f'{self.CLASSES[cl]}: {p[cl]:0.2f}'
            text = f'{p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=8)
            #        bbox=dict(facecolor='yellow', alpha=0.25))
        plt.axis('off')
        plt.savefig('results.png')  # Save the plot to a file
        plt.close()

    def infer(self, image):
        self.context.push()  # Make sure to push context to current thread before CUDA operations
        
        # Load and preprocess image
        start_time = time.time()
        image_pil = Image.fromarray(image)
        img_tensor = self.transform(image_pil).unsqueeze(0).numpy()
        #print(f"Image tensor shape: {img_tensor.shape}")
        
        if tensorrt_version.startswith('10'):
            for i in range(self.engine.num_io_tensors):
                self.trt_context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        
        [cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream) for inp in self.inputs]
        if tensorrt_version.startswith('8'):
            self.trt_context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        elif tensorrt_version.startswith('10'):
            self.trt_context.execute_async_v3(stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream) for out in self.outputs]
        self.stream.synchronize()
        
        # Postprocessing : Assuming output[0] is pred_logits and output[1] is pred_boxes
        start_time = time.time()
        pred_logits = np.asarray(self.outputs[0]['host']).reshape(100, 2+self.num_classes) #(#queries x #CLASSES)  3-if only windows, 5-if car,person, window
        pred_boxes = np.asarray(self.outputs[1]['host']).reshape(100, 4)  #(#queries x 4corners)
        #print(f"Predicted logits: {pred_logits}")
        #print(f"Predicted boxes: {pred_boxes}")
        probas, bboxes_scaled = self.process_outputs(pred_logits, pred_boxes, image_pil.size)

        self.context.pop()  # Pop context when done if this thread won't immediately continue with more CUDA operations

        return probas, bboxes_scaled

# cuda-memcheck python3 detect_trt.py
# cuda-gdb python3 detect_trt.py
# Example usage
def main(args):
    trt_inference = TensorRTInference(args.engine, args.num_classes)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Image at path {image} could not be loaded")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    probas, bboxes = trt_inference.infer(image)
    trt_inference.plot_results(image, probas, bboxes)
    print(f"Bounding boxes: {bboxes}")
    trt_inference.cleanup() # delete objects to avoid memory segmentation fault
    print("--- finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict and recreate images with missing red channel.')
    parser.add_argument('--engine', type=str, default='model.engine', help='Directory containing images with missing red channel')
    parser.add_argument('--image', type=str, default='../samples/20230210T081213.646665.png', help='Path to the trained model file')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
    args = parser.parse_args()
    main(args)
    


