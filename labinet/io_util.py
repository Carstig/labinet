from utils import label_map_util
import tensorflow as tf
import numpy as np

def load_label_map(path_to_labels, num_classes):
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return categories, category_index


def load_model(path_to_model):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
def load_image_into_numpy_array(image):
    '''
    reshape image to [1, None, None, 3] so all pixels are in one column
    might only work with jpg (as png have 4 channels)
    @param image image loaded with PIL - maybe the latter is not important
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
