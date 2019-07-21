# Project Title

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please ensure you have anaconda installed in your Linux machine. 

### Installing

A step by step series of examples that tell you how to get a development env running

```
cd envs 
source ./setup.sh
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
   Input: dataset_type, transform, is_input_flatten (optinal)
          
          dataset_type: 'MNIST', 'CIFAR10', or 'IMAGENET'
          transform: please input the transform used your training process 
          is_input_flatten: True or False. (If True, input will be collapsed into 2 dimensions. For instance, (1000, 3, 28, 28) -> (1000, 3*28*28))

### Examples 

Use pretrained model (e.g., resnet34) in Pytorch and measure its accurancy with ImageNet dataset. 

``` python 
transform = transforms.Compose([...])

user_model = models.resnet34(pretrained=True)
wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=1000)
accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', transform, is_input_flatten=False)
acc = accurancy_measurer.measure_accurancy(wrapped_model)
```

