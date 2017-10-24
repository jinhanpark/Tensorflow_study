from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile

MODEL_FILE = './output_graph.pb'
IMAGE_DIRECTORY = 'D:/Datasets/cats_vs_dogs/predict/'
NUM_OF_IMAGES = 50
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
FINAL_TENSOR_NAME = 'final_result:0'
LOAD_ARBITRARY = 0


def create_image_list(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    extensions = ['jpg', 'jpeg']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))
    if not file_list:
        print('No files found')
    result = []
    for file_name in file_list:
        base_name = os.path.basename(file_name)
        result.append(base_name)
    return result

def create_image_list_in_order():
    result = []
    for i in range(NUM_OF_IMAGES):
        base_name = str(i+1) + '.jpg'
        result.append(base_name)
    return result

def get_image_path(image_list, index, image_dir):
    base_name = image_list[index]
    path = os.path.join(image_dir, base_name)
    return path

def create_output_graph():
    with tf.Graph().as_default() as graph:
        model_filename = MODEL_FILE
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            jpeg_data_tensor, final_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    JPEG_DATA_TENSOR_NAME, FINAL_TENSOR_NAME    
                ])
            )
    return graph, jpeg_data_tensor, final_tensor

def final_on_image(sess, image_data, image_data_tensor, final_tensor):
    final_values = sess.run(
        final_tensor,
        {image_data_tensor: image_data}
    )
    final_values = np.argmax(final_values)
    return final_values

def calculate_final(sess, image_list, index, image_dir,
                         jpeg_data_tensor, final_tensor):
    image_path = get_image_path(image_list, index, image_dir)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        final_values = final_on_image(
            sess, image_data, jpeg_data_tensor, final_tensor
        )
    except:
        raise RuntimeError('Error during processing file %s' % image_path)
    return final_values

def get_predictions(sess, image_list, image_dir, jpeg_data_tensor, final_tensor):
    filenames = []
    predictions = []
    for i in range(len(image_list)):
        image_name = get_image_path(image_list, i, image_dir)
        prediction = calculate_final(sess, image_list, i, image_dir,
                                     jpeg_data_tensor, final_tensor)
        predictions.append(prediction)
        filenames.append(image_name)
    return predictions, filenames

def main():
    graph, jpeg_data_tensor, final_tensor = (create_output_graph())
    image_dir = IMAGE_DIRECTORY
    if LOAD_ARBITRARY:
        image_list = create_image_list(image_dir)
    else:
        image_list = create_image_list_in_order()

    with tf.Session(graph=graph) as sess:
        predictions, filenames = get_predictions(sess, image_list, image_dir,
                                                 jpeg_data_tensor, final_tensor)
        classes = ['cat', 'dog']
        for i in range(len(filenames)):
            print("[%2d] image path : %s\n     prediction : %s"
                  %(i+1, filenames[i], classes[predictions[i]]))
    
if __name__ == "__main__":
    main()
