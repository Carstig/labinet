# LabiNet
A fun project to detect my Labrador dogs going onto our couch by applying AI image processing.

Directory structure

notebooks 
_contains prototyping code and jupyter notebooks using the framework_

labinet
_actual sources implementing model and inference code_

data
_will have tf records and csv files_

images
_will contain our dataset (during work - not in git). I have not labeled all images, the ones I did have a `<imagename>.xml`_ I will not publish my full picture set due to privacy reasons. (my family is also visible ;-) - the dogs did not care about their privacy :-) )

scripts
_well - contains scripts to manage conversion and training that are *not* python_

training
_used to save our trained model and checkpoints_

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

I recommend to take a picture only every second (or even less). Either "objects" move so fast they become blurred (and more blurred pictures do not help), or they are so slow that within one second there is not much change in the object position or pose.

_I took 5 pictures per seconds, ended up with 3000 pictures, which I labeled only about 300 from to have a good variance in what the pictures show._

### Labeling with labelImg

ensure you copy first all your images into `images` directory and let LabelImg tool save the xml (label info) into the same directory. If you did the labeling before the move - this is ok. Do not care about the details contained in the xml files.

[Object Detection Tutorial youtoube video](https://www.youtube.com/watch?v=kq2Gjv_pPe8&index=4&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku) mentions to copy your labeled images into train and test directories... hm... I would rather want to let the splitting being data within the python scripts. _back from Training_ The retraining is based on a config file which wants us to do this. i.e. you need a `test.csv` and create the `test.record` file.

#### xml to csv
[Raccoon dataset](https://github.com/datitran/raccoon_dataset) downloaded xml_to_csv.py and put it into my repository - with adaptations for my repo. After conversion the aforementioned folder and path values are gone in the csv.
Imho `xml_to_csv.py` is a bit "hardcoded". 
- It contains the directory to search for the xml files
- will put all xml into one csv (no train/test splitting)
- the csv outfile is saved to hardcoded name

### Step 2 - Creating TFRecords from the LabelImg xml data
I use [this script](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) to create TFRecords. Change ` row_label == 'dog' ` 
Open a anaconda window and switch to your tensorflow virtual env. (you need the tensorflow package installed), run (from within `$gitbase`/images ) 

```
python ..\scripts\generate_tfrecord.py --csv_input=..\data\train.csv --output_path=..\data\train.record
``` 

_I needed to do this from within the images directory as tensorflow starts reading the different jpg files_

### Step 3 - Training
I download [ssd_mobilenet_v1_coco.config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config)
Save it into dir `training`. I also download the model itself (`ssd_mobilenet_v1_coco_2018_01_28.tar.gz`) and unpack it into `training/ssd_mobilenet_v1_coco`

Put `object-detection.pbtxt` into dir `data`

I follow the details steps from my selected [tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) - I leave batch_size (as I have tf w/ gpu).

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

#TO visualize the training results
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


### Step 7 Use your object-detection classifier to label pictures

TODO: write own script that creates xml format of labelImg!
TODO: notebook to visualize pictures with boxes to "sort" out wrong labeled pictures

### Step 8 


# Resources and Links

[1]: [2019, Installing TensorFlow, Keras, & Python 3.7 in Windows](https://www.youtube.com/watch?v=59duINoc8GM) _these videos also exist for other OS. And he creates new ones every year - so search for the update_

[2]: [How To Train an Object Detection Classifier Using TensorFlow](https://www.youtube.com/watch?v=Rgpfk6eYxJA) - also see [10] for the associated git rep

[3]: [Intro - TensorFlow Object Detection API](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)

[4]: [RASPBERRY PI – KAMERAMODUL ALS ÜBERWACHUNGSKAMERA (LIVESTREAM)](https://www.datenreise.de/raspberry-pi-ueberwachungskamera-livestream/)

[5]: [TensorFlow Object Detection API tutorial — Training and Evaluating Custom Object Detector](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)

[6]: [A complete Transfer Learning Toolchain for Object Detection](https://medium.com/practical-deep-learning/a-complete-transfer-learning-toolchain-for-semantic-segmentation-3892d722b604)

[7]: [Tensorflow.org - How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)

[8]: [Tensorflow.org - Transfer Learning Using Pretrained ConvNets](https://www.tensorflow.org/tutorials/images/transfer_learning)

[9]: [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

[10]: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 

