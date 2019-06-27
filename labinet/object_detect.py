from __future__ import  absolute_import
from . import io_util

import tensorflow as tf
import numpy as np
from utils import visualization_utils as vis_util

# from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
def run_inference_for_single_image(image, graph):
  '''
  runs inference (object detection) on image and returns dict 
  with num_detections, detection_classes, detection_masks, detection_boxes, detection_scores
  @param image loaded as is
  '''
  image_np_exp = np.expand_dims(io_util.load_image_into_numpy_array(image), axis=0) 
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np_exp.shape[1], image_np_exp.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image_np_exp})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def visualize_boxes_after_detection(image, inference_output_dict, category_index):
    '''
      returns an image with the detected boxes visualized
      that can be displayed with matplotlib.pyplot.imshow()
      @param inference_output_dict return value from run_inference_for_single_image
      @param image 
      @param score the minimum score so that a box is visualized
    '''
    image_np = io_util.load_image_into_numpy_array(image)

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

