import torchvision.models as models
import pytorch2timeloop
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import math
import torchvision.ops as ops

# Import the Converter class
from pytorch2timeloop.utils.interpreter import Converter

# Add module types to bypass
Converter.DEFAULT_BYPASSED_MODULES = (
    nn.BatchNorm2d,
    nn.Dropout,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.ReLU,
    nn.ReLU6,
    nn.AvgPool2d,         # Critical for DenseNet
    nn.AdaptiveAvgPool2d, # Used in many networks
    nn.MaxPool2d,
    nn.SiLU,              # Used in EfficientNet (swish activation)
    nn.Sigmoid            # Used in EfficientNet
)

# Extend the default ignored functions with our list
Converter.DEFAULT_IGNORED_FUNC.extend([
    torch.cat,           # Used in many networks for concatenation
    torch.add,           # Used in residual connections and element-wise operations
    torch.mul,           # Used in attention mechanisms and element-wise operations 
    F.avg_pool2d,        # Used in pooling layers
    F.adaptive_avg_pool2d, # Used in pooling layers
    F.max_pool2d,        # Used in pooling layers
    F.relu,              # Activation function
    F.sigmoid,           # Activation function
    torch.flatten,       # Used to flatten tensors before FC layers
    F.hardswish,         # MobileNetV3 activation
    F.hardsigmoid,       # MobileNetV3 activation
    F.silu,              # EfficientNet activation (also called swish)
    ops.stochastic_depth # Used in EfficientNet for stochastic depth regularization
])

def convert_alexnet():
    # AlexNet conversion
    net = models.alexnet()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'alexnet'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool']
    
    # We no longer need to pass ignored_func since we modified the class directly
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )


def convert_vgg16():
    # VGG16 conversion
    net = models.vgg16()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'vgg16'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'dropout']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )

def convert_resnet18():
    # ResNet18 conversion
    net = models.resnet18()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'resnet18'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'add']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )

def convert_densenet121():
    # Use DenseNet121 directly
    net = models.densenet121()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'densenet121'
    convert_fc = False
    
    # Additional module names to skip
    exception_module_names = ['view', 'pool', 'avgpool', 'concat', 'add', 'transition', 'AvgPool']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )



def convert_mobilenet_v3():
    # MobileNetV3 conversion (using small variant, change to "LARGE" if needed)
    net = models.mobilenet_v3_small()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'mobilenet_v3_small'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'hardsigmoid', 'hardswish', 'mul', 'add']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )

def convert_squeezenet():
    # SqueezeNet conversion (using version 1.1)
    net = models.squeezenet1_1()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'squeezenet'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'cat', 'concat']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )

def convert_googlenet():
    # GoogLeNet (Inception v1) conversion
    net = models.googlenet(aux_logits=False)  # Disable auxiliary outputs for inference
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'googlenet'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'cat', 'concat', 'Aux', 'dropout']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )

def convert_efficientnet_b0():
    # EfficientNet-B0 conversion
    net = models.efficientnet_b0()
    input_shape = (3, 224, 224)
    batch_size = 1
    top_dir = 'workloads'
    model_name = 'efficientnet_b0'
    convert_fc = False
    exception_module_names = ['view', 'pool', 'avgpool', 'add', 'cat', 'concat', 'sigmoid', 'mul', 'swish', 'SiLU']
    
    pytorch2timeloop.convert_model(
        model=net,
        input_size=input_shape,
        batch_size=batch_size,
        model_name=model_name,
        save_dir=top_dir,
        fuse=False,
        convert_fc=convert_fc,
        exception_module_names=exception_module_names
    )



# Main execution to convert all networks
if __name__ == "__main__":
    print("Converting AlexNet...")
    convert_alexnet()
    
    print("Converting VGG16...")
    convert_vgg16()
    
    print("Converting ResNet18...")
    convert_resnet18()
    
    print("Converting DenseNet121...")
    convert_densenet121()
    
    print("Converting MobileNetV3...")
    convert_mobilenet_v3()
    
    print("Converting SqueezeNet...")
    convert_squeezenet()
    
    print("Converting GoogLeNet...")
    convert_googlenet()
    
    print("Converting EfficientNet-B0...")
    convert_efficientnet_b0()
    
    print("All conversions completed!")