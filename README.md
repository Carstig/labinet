# LabiNet
A fun project to detect my Labrador dogs going onto our couch by applying AI image processing.

Directory structure
(following https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510 )

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

### Labeling with labelImg
ensure you copy first all your images into `images` directory and let LabelImg tool save the xml (label info) into the same directory. If you did the labeling before the move - this is ok. Do not care about the details contained in the xml files.

[Object Detection Tutorial youtoube video](https://www.youtube.com/watch?v=kq2Gjv_pPe8&index=4&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku) mentions to copy your labeled images into train and test directories... hm... I would rather want to let the splitting being data within the python scripts.

#### xml to csv
[Raccoon dataset](https://github.com/datitran/raccoon_dataset) downloaded xml_to_csv.py and put it into my repository - with adaptations for my repo. After conversion the aforementioned folder and path values are gone in the csv.

### Creating TFRecords from the LabelImg xml data

