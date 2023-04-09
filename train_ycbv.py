from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
model = YOLO("runs/detect/train7/weights/best.pt")  # load a pretrained model (recommended for training)

freeze_layers = 22
# Freeze 
freeze = [f'model.{x}.' for x in range(freeze_layers)]  # layers to freeze 
for k, v in model.named_parameters(): 
     v.requires_grad = True  # train all layers 
     if any(x in k for x in freeze): 
         print(f'freezing {k}') 
         v.requires_grad = False 


# Use the model
# model.train(data="ycbv-22.yaml", epochs=12, imgsz=512, batch=-1, device='0')  # train the model
model.train(data="ycbv-22.yaml", epochs=12, imgsz=512, batch=-1)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
