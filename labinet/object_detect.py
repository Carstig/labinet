from __future__ import  absolute_import
from . import io_util

import tensorflow as tf
import numpy as np
from utils import visualization_utils as vis_util

def get_tensor_dict(graph):
    tensor_dict = {}
    with graph.as_default():
        ops = graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_names = ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks', 'image_tensor']
        for key in tensor_names:
            tensor_name = key + ':0'
            #print(f'get tensor:{tensor_name}')
            if tensor_name in all_tensor_names:
              print(f'set: {tensor_name}')
              tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
    return tensor_dict        

def set_detection_masks(image_width, image_height, tensor_dict):
    print(f"tensor:0 for image.shape=(x={image_width},y={image_height})")
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_height, image_width)
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        print("reframed : ")
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0) 

def get_tensor_dict_with_masks(image_width, image_height, detection_graph):
  tensor_dict = get_tensor_dict(detection_graph)
  set_detection_masks(image_width, image_height, tensor_dict)
  return tensor_dict



def convert_output_dict(output_dict):
  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]


# from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
def run_inference_for_single_image(image, graph):
  '''
  runs inference (object detection) on image and returns dict 
  with num_detections, detection_classes, detection_masks, detection_boxes, detection_scores
  @param image loaded as is
  '''
  image_np_exp = np.expand_dims(io_util.load_image_into_numpy_array(image), axis=0) 
  tensor_dict = get_tensor_dict(graph)
  set_detection_masks(image_width=image_np_exp.shape[2], image_height=image_np_exp.shape[1], tensor_dict=tensor_dict)
  image_tensor = tensor_dict['image_tensor']

  with tf.Session(graph=graph) as sess:
    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: image_np_exp})

    convert_output_dict(output_dict)
  return output_dict


def visualize_boxes_after_detection(image, inference_output_dict, category_index):
    '''
      returns an image with the detected boxes visualized
      that can be displayed with matplotlib.pyplot.imshow()
      @param inference_output_dict return value from run_inference_for_single_image
      @param image 
      @param score the minimum score so that a box is visualized
    '''
    #image_np = io_util.load_image_into_numpy_array(image)    
    image_np = image

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        inference_output_dict['detection_boxes'],
        inference_output_dict['detection_classes'],
        inference_output_dict['detection_scores'],
        category_index,
        instance_masks=inference_output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4 )

    return image_np


def get_boxes_to_use(detection_boxes, detection_scores, min_score_threshold=0.8):
    '''
    returns list of boxes that have score higher than min_score_threshold
    @param detection_boxes result from inference
    @param detection_scores result from inference
    '''
    boxes = detection_boxes
    scores = detection_scores
    #print(output_dict['detection_boxes'])
    num_boxes = boxes.shape[0]    
    boxes_to_use = []
    for i in range(num_boxes):
        if scores is None or scores[i] > min_score_threshold:
            box = boxes[i]
            boxes_to_use.append(box)       
            #score = int(scores[i] * 100)
            #print(f"Use box[{i}]={boxes[i]}. (score is {score}%)")
    return boxes_to_use        

