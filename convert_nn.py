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
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.MaxPool2d,
    nn.SiLU,
    nn.Sigmoid
)

# Extend the default ignored functions
Converter.DEFAULT_IGNORED_FUNC.extend([
    torch.cat,
    torch.add,
    torch.mul,
    F.avg_pool2d,
    F.adaptive_avg_pool2d,
    F.max_pool2d,
    F.relu,
    F.sigmoid,
    torch.flatten,
    F.hardswish,
    F.hardsigmoid,
    F.silu,
    ops.stochastic_depth
])

def convert_networks(batch_size, network_names, top_dir='workloads', input_shape=(3, 224, 224), convert_fc=False):
    """
    Convert specified neural networks to Timeloop format with batch size in model name.
    
    Args:
        batch_size (int): Batch size for conversion
        network_names (list): List of network names to convert
        top_dir (str): Output directory
        input_shape (tuple): Input shape for the networks
        convert_fc (bool): Whether to convert fully connected layers
    """
    # Union of all exception module names from original functions
    exception_module_names = [
        'view', 'pool', 'avgpool', 'dropout', 'add', 'concat', 'transition',
        'AvgPool', 'hardsigmoid', 'hardswish', 'mul', 'cat', 'Aux', 'sigmoid', 'swish', 'SiLU'
    ]
    
    # Dictionary mapping network names to their model constructors
    network_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'densenet121': models.densenet121,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'squeezenet': models.squeezenet1_1,
        'googlenet': lambda: models.googlenet(aux_logits=False),
        'efficientnet_b0': models.efficientnet_b0
    }
    
    for name in network_names:
        if name not in network_models:
            print(f"Warning: Network {name} not supported. Skipping...")
            continue
            
        print(f"Converting {name} with batch size {batch_size}...")
        try:
            # Initialize model
            model = network_models[name]()
            
            # Include batch size in model name
            model_name = f'{name}_{batch_size}'
            
            # Convert model
            pytorch2timeloop.convert_model(
                model=model,
                input_size=input_shape,
                batch_size=batch_size,
                model_name=model_name,
                save_dir=top_dir,
                fuse=False,
                convert_fc=convert_fc,
                exception_module_names=exception_module_names
            )
        except Exception as e:
            print(f"Error converting {name}_{batch_size}: {str(e)}")
            
if __name__ == "__main__":
    # Example usage
    batch_sizes = [1,32]
    networks_to_convert = ['vgg16', 'resnet18', 'resnet50', 'mobilenet_v3_small', 'efficientnet_b0']
    for b in batch_sizes:
        convert_networks(batch_size=b, network_names=networks_to_convert)
    print("Conversions completed!")