import torch
import torchvision.models as models
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import tensorrt as trt
import torch
import torch.nn.utils.prune as prune
import torchvision.models as models

# Load a pre-trained model (ResNet-50)
model = models.resnet50(pretrained=True)
model.eval()

# Input tensor for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "resnet50.onnx")

"""
import tensorflow as tf

# Load a pre-trained TensorFlow model
model = tf.keras.applications.MobileNetV2()

# Convert the model to ONNX
import tf2onnx
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
output_path = "mobilenetv2.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
"""


# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create a builder and network
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Create an ONNX parser
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load the ONNX model
with open("resnet50.onnx", "rb") as model:
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse the ONNX model.')

# Set the optimization profile (batch size, precision)
builder.max_batch_size = 8  # Example batch size
builder.fp16_mode = True    # Enable FP16 precision

# Create the engine
engine = builder.build_cuda_engine(network)

# Save the TensorRT engine
with open("resnet50_engine.trt", "wb") as f:
    f.write(engine.serialize())

# Load a pre-trained model (ResNet-50)
model = models.resnet50(pretrained=True)

# Apply structured pruning to Conv2d layers
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)

# Export the pruned model to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "pruned_resnet50.onnx")

# Load the TensorRT engine
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        return trt_runtime.deserialize_cuda_engine(f.read())

# Allocate memory for input/output
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

# Run inference
def run_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()

# Example usage
engine = load_engine(trt.Runtime(TRT_LOGGER), "resnet50_engine.trt")
context = engine.create_execution_context()

inputs, outputs, bindings, stream = allocate_buffers(engine)

# Load input data (example random data)
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
inputs[0]['host'] = input_data.ravel()

# Run inference
run_inference(context, bindings, inputs, outputs, stream)

# Get the output
output = outputs[0]['host'].reshape(1, 1000)  # Reshape as per model output
print("Inference output:", output)