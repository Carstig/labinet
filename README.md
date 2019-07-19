# LabiNet
A fun project to detect my Labrador dogs going onto our couch by applying AI image processing.

Directory structure

notebooks 
_contains prototyping code and jupyter notebooks using the framework_

labinet
_actual sources implementing model and inference code_

data
_will have tf records and csv files_

images/train
images/test
_will contain our dataset (during work - not in git). I have not labeled all images, the ones I did have a `<imagename>.xml`_ I will not publish my full picture set due to privacy reasons. (my family is also visible ;-) - the dogs did not care about their privacy :-) )
images_train should contain images for training and images_test should contain images for evaluation (aka test).

scripts
_well - contains scripts to manage conversion and training_

training
_used to save our pretrained model, config file and trained model and checkpoints_

eval
_save results of evaluation on our trained model_

test
_python unit tests_


## Getting Started

### Installing Tensorflow
_this was so far the hardest part_ I recommend you follow [Jeff Heaton's install instruction](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class01_intro_python.ipynb). Also just search for his youtube video "Jeff Heaton `<your os>`" as he updates his video, repository along with the description at least once a year.

_In case you play with the thought to train with GPU power: it might be worthwhile to use a cloudservice like Watson Studio or AWS that come with already working TF & GPU software stack. Getting this to run on your own might be a real pain (believe Jeff Heaton's comment - I have been there!)_

### Transfer Learning
The idea is to reuse an already pretrained model and adapt it's architecture to my needs (only detect the dogs and maybe the couch) and retrain it with my own images (Transfer learning is the key word).

After some googling I decide to follow [TensorFlow Object Detection API tutorial — Training and Evaluating Custom Object Detector](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) from medium.com

### Step 1 - Labeling and CSV creation

#### Creating Images
I created images by using my Raspi and it's cam following [4]. 

I recommend to take a picture only every 5th (or even less). Either "objects" move so fast they become blurred (and more blurred pictures do not help), or they are so slow that within one second there is not much change in the object position or pose.

_I took 5 pictures per seconds, ended up with 3000 pictures, which I labeled only about 300 from to have a good variance in what the pictures show. In my second round I configured to have an img every 5th second - that was quite good, but when the dog is asleep you have lots of pretty identical images._

If you have configured your cam//motion as deamon, the raspi will save pictures as soon as it is booted. Use `sudo service motion start | stop` to toggle this.

In case not use this short command sequence to restart the image capturing:

1. Enable camera with `sudo raspi-config`
    1. go to `Interfaces`
    2. select camera and enable it
2. Take one picture `raspistill -o test.jpg`

It is quite handy to configure _motion_ to open the image stream on a webinterface.
You can then check the created images under `http://<raspi-ip>:8081/`

### Labeling with labelImg

ensure you copy first all your images into `images` directory and let LabelImg tool save the xml (label info) into the same directory. If you did the labeling before the move - this is ok. Do not care about the details contained in the xml files.

[Object Detection Tutorial youtoube video](https://www.youtube.com/watch?v=kq2Gjv_pPe8&index=4&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku) mentions to copy your labeled images into train and test directories... hm... I would rather want to let the splitting being data within the python scripts. _back from Training_ The retraining is based on a config file which wants us to do this. i.e. you need a `test.csv` and create the `test.record` file.

### Step 2 - Creating TFRecords from the LabelImg xml data

#### xml to csv
[Raccoon dataset](https://github.com/datitran/raccoon_dataset) downloaded xml_to_csv.py and put it into my repository - with adaptations for my repo. After conversion the aforementioned folder and path values are gone in the csv. 
Imho `xml_to_csv.py` is a bit "hardcoded". 
1. cd to `gitbase`
2. edit xml_to_csv.py to have the correct directory to the images
3. set the output filename (`train`)
4. invoke `python .\scripts\xml_to_csv.py`


#### csv to TFRecords
I use [this script](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) to create TFRecords. Change ` row_label == 'dog' ` 
Open a anaconda window and switch to your tensorflow virtual env. (you need the tensorflow package installed), run (from within `$gitbase`/images/train ) 

```
python ..\..\scripts\generate_tfrecord.py --csv_input=train.csv --output_path=..\..\data\train.record
``` 

_I needed to do this from within the images directory as tensorflow starts reading the different jpg files_

### Step 3 - Training
I download [ssd_mobilenet_v1_coco.config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config)
Save it into dir `training`. I also download the model itself (`ssd_mobilenet_v1_coco_2018_01_28.tar.gz`) and unpack it into `training/ssd_mobilenet_v1_coco`

Put `object-detection.pbtxt` into dir `data`

I follow the details steps from my selected [tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) - I leave batch_size (as I have tf w/ gpu. you might have to select a lower value when you run out of (GPU)-memory). But I experimented with size smaller than 24 and the loss did vary quite a lot wheras with 24 it became decreasing in a stable way.

Four round 2 (training again):
1. Now, copy `data/`, `images/train` (to images) directories to `models/research/object-detection` directory. If it prompts to merge the directory, merge it.

Then cd to the `models/research/object-detection` directory

from within your anaconda prompt, start training with (use forward slashes even on Windows)
```
python train.py --logtostderr --train_dir=training --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
 ```

*leave it train for couple of hours - the loss has to became less than 1*

Visualize the training process and progress by calling `tensorboard --logdir=training` from within the `object_detection` directory. You should see the loss starting to level.


### Step 4 Evaluate
copy `eval.py` from `legacy` dir into `object_detection` dir

- need to install pycocoapi
    - install [vs studio build env](https://go.microsoft.com/fwlink/?LinkId=691126)
    - install git
    - reopen conda prompt
    - `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

- modify tensorflow-models:
    - in `tensorflow-models\research\` delete or rename `pycocotools`
    - in `object_detection_evaluation.py` replace `unicode()` with `str()`

- start evaluation
``` 
python eval.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --checkpoint_dir=training/ --eval_dir=eval/
```  

In case you get an `Image with id... already added` Error, you forgot to update in the .config file with the right number of examples. (must match number of test = evaluation images)

```
eval_config: {
        #num of test images. In my case 71. Previously It was 8000
        num_examples: 71
``` 
---> did not help. Not yet solved.


### Step 5 Visualize Results
```
#To visualize the eval results
tensorboard --logdir=eval

#To visualize the training results
tensorboard --logdir=training
```

Eval results are horrible. Many boxes within one picture, and boxes with label=dog and a proba > 70% where no dog is.

After training it again (now this time four couple of hours - I checked the tensorboard and could observe that the loss started to flatten out) I got 10 pictures where with a 99% confidence one dog box was shown! (wow) and only in one picture he missed out the 2nd box to show the second dog just entering the picture (about 2/3rds of the dog were in the pic). This is awesome.

### Step 6 Export Inference Graph
I am now referencing to [2] and [10]

- Note the step count of the highest model checkpoint. For this go to the directory `.../object_detection/training`. 
- cd into `objection_detection`
- replace `XXXX` with the count from the model checkpoint chosen above
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
``` 
- check you have now `inference_graph/frozen_inference_graph.pb`
- copy this directory now to `$gitbase`


### Step 7 Use your object-detection classifier to label pictures

- place some new pictures (as created in step 1) and place them in a new directory
- i created a for loop to now run inference on the new graph and it will create xml files in the PascalVOC format (the one from LabelImg)
```
NEW_IMAGE_PATH = os.path.join(CWD_PATH, 'new_images')
files = os.listdir(NEW_IMAGE_PATH)

for fname in files:
    if fname.endswith('jpg'):
        image_file = os.path.join(NEW_IMAGE_PATH, fname)
        print(f'inference on: {fname}')
        image =  Image.open(image_file)
        output_dict = labinet.object_detect.run_inference_for_single_image(image, detection_graph)
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        boxes_to_use = labinet.object_detect.get_boxes_to_use(boxes, scores)
        boxes_normed = []
        print(f'found {len(boxes_to_use)} objects')
        for box in boxes_to_use:
            box_normed = labinet.box.get_normalized_coordinates(box, image.size)
            boxes_normed.append(box_normed)

        xml = labinet.box.box_to_labelimg_xml(fname, image.size, boxes_normed, imagepath=image_file)
        xmlfname = image_file[:-4]+".xml"
        xml.write(xmlfname)

``` 
- open LabelImg and click through the pictures to see how good we do
- in case of missing or bad box, correct it. Save the xml and note down the image name
- copy the images noted above into the `images\train` directory
- restart from Step 2

### Step 8 use new model on live stream
Object detection on Live Stream actually means, that we have a Loop to capture an image from a webcam and then apply _object detection on one image_ . 

To make that actually perform, you need to have a better implementation than the `labinet.object_detect.run_inference_for_single_image(image, detection_graph)`

Performance gains come from reusing a created tf.Session by:
- move loading model outside function (and loop)
- extracting tensors and preparing image tensors only once
- creating session only once
- reuse the session and call repeatedly ` output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_exp})`

```
# load model and labels
detection_graph = load_model(PATH_TO_MODEL)
categories, category_index = load_label_map(LABEL_MAP, NUM_CLASSES)

# prepare tensor dict for inference
tensor_dict = labinet.object_detect.get_tensor_dict_with_masks(IMAGE_SIZE[1], IMAGE_SIZE[0], detection_graph)
image_tensor = tensor_dict['image_tensor']
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Prepare the Cam!
video = cv2.VideoCapture(0)
ret, frame = video.read()
if frame is None:
    print("Error - did you connect your webcam?")

with tf.Session(graph=detection_graph, config=config) as sess:
    # capture
    while(True):
        ret, frame = video.read()
        #print(f"Captured frame.shape={frame.shape} - type(frame)={type(frame)}")
        image_np_exp = np.expand_dims(frame, axis=0) 
        #print(f'np-frame.shape={image_np_exp.shape}')
        # inference
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_exp})
        labinet.object_detect.convert_output_dict(output_dict)
        # visualize boxes
        image_with_boxes = labinet.object_detect.visualize_boxes_after_detection(frame, output_dict, category_index)
        # show image
        cv2.imshow('Detection Running...', cv2.resize(image_with_boxes,(IMAGE_SIZE[1],IMAGE_SIZE[0])))
        #cv2.waitKey(25)
        if cv2.waitKey(25) == ord('q'):
            break        

video.release()
cv2.destroyAllWindows()
``` 


# Resources and Links

[1]: [2019, Installing TensorFlow, Keras, & Python 3.7 in Windows](https://www.youtube.com/watch?v=59duINoc8GM) _these videos also exist for other OS. And he creates new ones every year - so search for the update_

[2]: [How To Train an Object Detection Classifier Using TensorFlow](https://www.youtube.com/watch?v=Rgpfk6eYxJA) - also see [10] for the associated git repo

[3]: [Intro - TensorFlow Object Detection API](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)

[4]: [RASPBERRY PI – KAMERAMODUL ALS ÜBERWACHUNGSKAMERA (LIVESTREAM)](https://www.datenreise.de/raspberry-pi-ueberwachungskamera-livestream/)

[5]: [TensorFlow Object Detection API tutorial — Training and Evaluating Custom Object Detector](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)

[6]: [A complete Transfer Learning Toolchain for Object Detection](https://medium.com/practical-deep-learning/a-complete-transfer-learning-toolchain-for-semantic-segmentation-3892d722b604)

[7]: [Tensorflow.org - How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

[8]: [Tensorflow.org - Transfer Learning Using Pretrained ConvNets](https://www.tensorflow.org/tutorials/images/transfer_learning)

[9]: [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

[10]: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 



# bad labeled images
20190414173231-00
20190414173232-00
20190414173242-00
20190414173243-00
20190414173244-00
20190414173245-00
20190414173727-00
20190414174049-00
20190414174104-00
20190414174108-00
20190414174146-00
20190414174154-00
20190414174156-00
20190414174202-00
20190414174206-00






