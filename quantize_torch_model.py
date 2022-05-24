from onnx_tf.backend import prepare
import tensorflow as tf
import pathlib
import onnx
import export_unparsed as ex

"""
The process includes .pt-->.onnx-->.tflite-->quantization-->.onnx (to be used directly)
"""

#loading torch model
root = 'folder of your model'
model_name = 'model.pt'
model_path = root+ model_name
types = ['tflite', 'onnx']
onnx_model1 = model_path.replace('pt', 'onnx')
#to convert model to onnx
ex.main() # I found it works better for me to directly add parameters in export_unparsed.py
          # you can parse it as you wish
#Load the onnx model  
onnx_model = onnx.load(onnx_model1)

#conversion to tflite model (tflite model needed to quantization)
tf_rep = prepare(onnx_model)
tf_rep.export_graph('/'.join(model_path.split('/')[:-1])+'/tf_model_pb')
tf_model = model_path.replace('pt', 'pb')

# quantize tf model
converter = tf.lite.TFLiteConverter.from_saved_model(root+'tf_model_pb/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# save quantized model
tflite_models_dir = pathlib.Path(root)
tflite_model_quant_file = tflite_models_dir/model_name.replace('pt', 'tflite')
tflite_model_quant_file.write_bytes(tflite_quant_model)

#use cli to convert the quantized tflite file to onnx model

python3.8 -m tf2onnx.convert --tflite 'tflite_quantized_model_path --output onnx_quantized_model_path --opset 13
# now you have quantized model in form of onnx
