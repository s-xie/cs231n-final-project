import argparse 
import os
import tempfile
import zipfile

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-m', required = True, type = str, help = 'path to model file')
	args = parser.parse_args()
	return args

def get_model_size(file):
	 _, zipped_file = tempfile.mkstemp('.zip')
	 with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
	   f.write(file)
	 return os.path.getsize(zipped_file)

args = clip()
print("Size of gzipped model: %.2f bytes" % (get_model_size(args.m)))