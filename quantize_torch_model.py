from onnx_tf.backend import prepare
import tensorflow as tf
import pathlib
import onnx
import export_unparsed as ex


# first define model class
#loading model
root = '/Data/Signals/Sat_proj/yolov5/'
model_name = 'epoch2275_best_SGD_square_1.pt'
model_path = root+ model_name
types = ['tflite', 'onnx']
onnx_model1 = model_path.replace('pt', 'onnx')
ex.main()
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

#back path to pytorch model

tflite_path = root+ model_name.replace('pt', 'tflite')

# onnx_quantized_export_model_path = root+ model_name.replace('pt', 'onnx')
"""
using script command it did not work and using cli it does work
python3.8 -m tf2onnx.convert --tflite 'tflite_path --output onnx_quantized_export_model_path --opset 13
now you have quantized model back in form of onnx
"""
