from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("model_weights/best_v11_E100.pt")  # load a custom trained model

# Export the model
model.export(format="engine", int8=True, keras = True,device = 0, batch = 1, half = True)