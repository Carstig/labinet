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
_will contain our dataset. I have not labeled all images, the ones I did have a `<imagename>.xml`_ 

scripts

training
_used to save our trained model and checkpoints_

eval
_save results of evaluation on our trained model_

test
_python unit tests_


## Getting Started

### Installing Tensforflow
_this was so far the hardest part_ I recommend you follow the [Jeff Heaton's install instruction ](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class01_intro_python.ipynb). Also just search for his youtube video "Jeff Heaton <your os>" as he updates video, repository along with the description at least once a year.

### Transfer Learning
The idea is to reuse an already pretrained model and adapt it's architecture to my needs (only detect the dogs and maybe the couch) and retrain it with my own images (Transfer learning is the key word).

After some googling I decide to follow [TensorFlow Object Detection API tutorial — Training and Evaluating Custom Object Detector](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) from medium.com

