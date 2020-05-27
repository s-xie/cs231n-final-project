# Imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
import os
import pathlib
import argparse

def clip():
	parser = argparse.ArgumentParser(description = 'Specify post-training quantization details')
	parser.add_argument('-m', required = True, type = str, help = 'path to model file')
	parser.add_argument('-q', required = True, type = int, choices = [8, 16], help = 'quantized weight precision (# of bits)')
	args = parser.parse_args()
	return args

# Collect command line arguments
args = clip()

# Load model
model_path = args.m
with tfmot.quantization.keras.quantize_scope():
    model = load_model(filepath = model_path, compile = True)
print('Model Loaded!')

# Convert model to TFLite
precision = args.q
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
if precision == 16:
	converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

# Save converted model
model_dir = os.path.dirname(model_path)
model_filename = os.path.basename(model_path)
tflite_filename = model_filename[:model_filename.index('.hdf5')] + '_q' + str(precision) + '.tflite'
tflite_filepath = pathlib.Path(model_dir)/tflite_filename
tflite_filepath.write_bytes(tflite_quant_model)
