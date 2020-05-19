import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os
import pydot
import graphviz 

model_folder = 'models/least_overfit_model/'
model_filename = 'model.hdf5'
diagram_filename = 'modified_vgg_model.png'
model = load_model(filepath = os.path.join(model_folder, model_filename), compile = False)
plot_model(model, to_file=os.path.join(model_folder, 'plots/', diagram_filename), show_shapes=True, show_layer_names=True)