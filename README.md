# MeasureDLS

This tools aims to faciliate analyses on various metrices (e.g., correctness and robustness) accross various machine learning frameworks (e.g., PyTorch and Keras).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please ensure you have anaconda installed in your Linux machine. 

### Installing virtual environments 

A step by step series of examples that tell you how to get a development env running

```
cd envs 
source ./setup.sh
```

### Preparing dataset (for measuring accurancy)

For preparing imagenet validation dataset, please download ILSVRC2012_img_val.tar from its official website and unzip those images in _data/img_val_. 

Several files (e.g., meta.mat or sysnet_words.txt) responsible for labeling are placed in _data/img_val_labels_. 

After moving unzipped images to _data/img_val_. 

```
python3 process_imagenet_val_and_store_as_npy.py
```

## Running the tests

We have prepared unit tests to verify correctness of functionalities for our tool. Run the following commands to activate unit tests. (Note that you switch to our virtual env before executing our tool)

```
conda activate measureDLS
python3 pytorch_test.py -v
python3 keras_test.py -v 
...
```

## Usage 

In Usage section, we will cover description for parameters and some examples for usage. 

### Parameters 

-  <b>measureDLS.measurement.AccurancyMeasurer</b>  
   Input: dataset_type, transform(optinal), is_input_flatten(optinal), preprocess(optional)
          
          dataset_type: 'MNIST', 'CIFAR10', or 'IMAGENET'
          transform: please input the transform used before forwarding (currently for PyTorch) 
          is_input_flatten: True or False. (If True, input will be collapsed into 2 dimensions. For instance, (1000, 3, 28, 28) -> (1000, 3*28*28))
          preprocess: please specify the preprocess used before forwarding (currently for Keras)

### Examples 

Use pretrained model (e.g., resnet34) in Pytorch and measure its accurancy with ImageNet dataset. 

``` python 
transform = transforms.Compose([...])

user_model = models.resnet34(pretrained=True)
wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=1000)
accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', transform, is_input_flatten=False)
acc = accurancy_measurer.measure_accurancy(wrapped_model)
```

Use pretrained model (e.g., vgg16) in Keras and measure its accurany with ImageNet dataset
``` python 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
        
user_model = VGG16(weights='imagenet', include_top=True)
wrapped_model = measureDLS.models.KerasModel(user_model, num_classes=1000)
accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=preprocess_input)
acc = accurancy_measurer.measure_accurancy(wrapped_model)
```

Use pretrained model (e.g., vgg19) in Keras and measure its top-5 accurany with ImageNet dataset
``` python 
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input 

k = 5
user_model = VGG19(weights='imagenet')
wrapped_model = measureDLS.models.KerasModel(user_model, num_classes=1000)
accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=preprocess_input, k_degree=k)
acc = accurancy_measurer.measure_accurancy(wrapped_model)
```

Use pretrained model (e.g., mobilenet) in tensorflow and measure its accurancy with ImageNet dataset
``` python 
pending
```

## Details of implementation 

### Recommandations (for enhancing productivity)

- Conduct unit tests periodically
- Use 'screen' in Linux, which enables you to place time-consuming tasks (e.g., data processing or model training) in an additional screen channel and continue to work on other tasks simultaneously. Besides, 'screen' allows you to run experiments even after the disconnection of ssh. 

### Something you need to pay attention

Please pay attention to the points described below during your implementation. 

- Be aware of the order of channels (colors for classification task), which can be RGB, BGR, or others. 
- Be aware that similar syntax (e.g., function with same/similar name) in the different framework may have different meanings. Please check the low-level implementation if needed. 
- Be aware of the hidden mechanism (e.g., default interpolation used in Resize function).
- Avoid hand-crafting any functions. There are many details in these frameworks that you could miss procedures in your hand-crafted functions. It is a critical lesson I learned by sacrificing ample time. Save your time and check the official documentation first. 
- Pay great attention to the preprocessing procedure for every pre-trained model released. It is somehow tedious work, but it matters to obtain the correct results of the analysis. 
- If time available, understanding some widely-utilized models (e.g., ResNet and MobileNet), different preprocessing process, and basic adversarial attacks (e.g., either gradient-based approach like FGSM, Fast Grident Sign Method, or optimization-based like Jacobian-based Saliency Map Attack) will be very helpful.

Due to the stochastic nature of machine learning program, it may be ambiguous for us to recognize some implementation errors. However, low accuracy should be a practical indication that there is something implemented inappropriately in our tool. 

### Current challenges and progress 

Challenges:
- It is still difficult to automatically conduct preprocessing for all frameworks. The way to handle preprocessing in a uniform method needed to be considered and appropriately tackled. 
- Currently, I am dealing with issues within the preprocessing procedure of TensorFlow pre-trained models. 

Progress (or future works needed):
- Accuracy measurement on mxnet/caffe is not yet implemented. However, it should be easy programming tasks. 
- Include top-5 error for accuracy measurement of Imagnet (I have doned that for Keras, please refer to measureDLS/measurement/accurancy.py).
- Include robustness (try local adversarial robustness first) for a simple task (MNIST classification). Call <b>Foolbox</b> for generating adversarial samples (I can sucessfully generate adversarial samples on Linux server, should not be a problem). 

## Acknowledgments

* This repository utilizes (calls) following tools during execution: foolbox (https://github.com/bethgelab/foolbox)
* <b>Jie M. Zhang, Mark Harman, Lei Ma, and Yang Liu. 2019. Machine Learning Testing: Survey, Landscapes and Horizons</b> is inspriational kickstart for people are interested in deep learning test yet not familiar with the topic. 
