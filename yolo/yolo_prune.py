
import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model,amount=0.3):
    for module in model.modules():
        if isinstance(module,torch.nn.Conv2d):
            prune.l1_unstructured(module,name="weight",amount=amount)
            prune.remove(module,"weight")
    return model

model = YOLO("yolov8n.pt")
#results= model.val(data="coco.yaml")

#print(f"mAP50-95: {results.box.map}")
torch_model = model.model
print(torch_model)

print("Prunning model...")
pruned_torch_model = prune_model(torch_model,amount=0.1)
print("Model pruned.")

model.model =pruned_torch_model

print("Saving pruned model...")
model.save("yolov8n_trained_pruned.pt")

print("Pruned model saved.")