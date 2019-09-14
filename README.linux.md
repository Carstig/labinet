# Runnning Labinet Stuff on Linux Ubuntu 18.0.4


# next todo:

- ssh setup / github
- debug CUDA problem
  - done NEXT TODO: rebuild docker image (and see if we can install cocoapi)
  - done then run docker image and use instead of train.py the script model_main.py >> same fail (CUDNN initilization fail)
  - TF 1.13/14 require CUDA 10.0 (i have 10.1) --> downgrade to 10.0
  - although I have now cuda-10 in /usr/lib/cuda - nvidia-smi show 10.1 ... 
  

### build the docker container
```
cd <path to dockerfile>
docker build --rm -t carstig/labinet_trainer . --no-cache=true
```
### run your container
either use `nvidia-docker` or `docker --runtime=nvidia` :

```
$> nvidia-docker run -u $(id -u):$(id -g) -it --mount type=bind,source=/home/cgreiner/python/object_detection/tensorflow-models,target=/home/docker/tensorflow-models --rm --runtime=nvidia carstig/labinet_trainer /bin/bash
``` 

from within this shell I do:
```
cd .../tensorflow-models/research/object-detection
python train.py --logtostderr --train_dir=training --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
```

Fails with 
```
see ../train.log
```



# installing tensorflow

The punch line for this task is : you don't - you are using a docker image. Due to the supported Linux versions from tensorflow I decide to 
use Ubuntu 18.0.2. I basically followed the following two blogs to first match the prerequisites and then to show how to run an interactive or jupyter container.

[Install Tensorflow Docker on Ubuntu 18.04](https://medium.com/@madmenhitbooker/install-tensorflow-docker-on-ubuntu-18-04-with-gpu-support-ed58046a2a56)

- includes nVidia driver install -- which version should I actually use?



[Running Tensorflow with Docker](https://winsmarts.com/easiest-way-to-setup-a-tensorflow-python3-environment-with-docker-5fc3ec0f6df1)

`$> nvidia-docker run tensorflow/tensorflow:latest-gpu-py3` 

built the docker file, ran it and started interactive using vscode:
also started image using nvidia-docker


result:

```
root@916bec87a1e8:/develop# ./starterfile.py 
WARNING: Logging before flag parsing goes to stderr.
W0908 21:04:10.556662 140574602528576 deprecation_wrapper.py:119] From ./starterfile.py:6: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2019-09-08 21:04:10.557069: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Could not dlopen library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2019-09-08 21:04:10.557095: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR (303)
2019-09-08 21:04:10.557124: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:163] no NVIDIA GPU device is present: /dev/nvidia0 does not exist
2019-09-08 21:04:10.557401: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```

## Docker Commands
- list running containers: docker ps 
- list all : docker ps -a
- stop running container: docker stop <container name(s)>
- remove container: docker rm <container name(s)>
- nvidia-docker -> use this for GPU (really? as I once run ` tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter` and seemed to train with GPU)

## Mount my NAS
```
sudo mount -t nfs 192.168.0.99:/volume1/documents /home/cgreiner/synology_nas_documents
```
### mount at boot
```
sudo nano /etc/fstab

add this line:
192.168.0.99:/volume1/documents /home/cgreiner/synology_nas_documents nfs  auto,nofail,noatime,nolock,intr,tcp,actimeo=1800 0 0

```







