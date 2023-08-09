from ultralytics import YOLO

# Load a model
model = YOLO("voc_yolov8s_C2fSCConv2.yaml")  # build a new model from scratch
# model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco.yaml", epochs=100,batch=32)  # train the model
model.train(data="driverVOC2007.yaml", epochs=300,batch=32)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
success = model.export(format="onnx")  # export the model to ONNX format