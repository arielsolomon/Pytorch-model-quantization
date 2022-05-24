# Pytorch-model-quantization
Part of AI/ML model deployment on edge devices includes reducing its memory size and weights type in order to ease on resources needed for using the model.
This two short scripts are used for doing it semi automatically using derivative of export.py from yolov5 and quantize_torch_model.py
The end part requires cli (command written at the end of quantize_torch_model.py)
I executed it in yolov5 docker enviornment(https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) and in the yolov5 folder (git clone https://github.com/ultralytics/yolov5  # clone, cd yolov5, pip install -r requirements.txt  # install) to simplify and adjust enviornment requirments.
