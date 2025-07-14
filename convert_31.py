import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch2timeloop
import torchvision.ops as ops

# Import the Converter class
from pytorch2timeloop.utils.interpreter import Converter

# Add module types to bypass
Converter.DEFAULT_BYPASSED_MODULES = (
    nn.BatchNorm2d,
    nn.SyncBatchNorm,    
    nn.Dropout,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.ReLU,
    nn.ReLU6,
    nn.AvgPool2d,         
    nn.AdaptiveAvgPool2d, 
    nn.MaxPool2d,
    nn.SiLU,              
    nn.Sigmoid,           
    nn.LayerNorm,         
    nn.GELU,              
    nn.Identity           
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
    ops.stochastic_depth,
    F.gelu,              
    F.layer_norm,        
    torch.permute,       
    F.pad                
])

def convert_replknet(batch_size: int):
    """
    Convert RepLKNet to Timeloop format using the implementation from replknet.py.
    Uses large depthwise convolutions (up to 31x31) as described in the paper
    "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"
    """
    # Add the RepLKNet-pytorch directory to the path
    repo_dir = os.path.abspath("RepLKNet-pytorch")
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    
    try:
        # Import the RepLKNet implementation directly
        sys.path.append(".")  # Add current directory to path
        from replknet import create_RepLKNet31B
        
        # Disable sync_bn in the replknet module to use regular BatchNorm2d instead
        import replknet
        replknet.use_sync_bn = False
        
        # Create RepLKNet-31B using the provided factory function
        # This uses the parameters from the paper: large kernel sizes [31,29,27,13], etc.
        model = create_RepLKNet31B(
            drop_path_rate=0.3,                # Default drop path rate
            num_classes=1000,                  # ImageNet classification
            use_checkpoint=False,              # Disable checkpointing for conversion
            small_kernel_merged=False          # Keep small kernels separate for better analysis
        )
        
        # Set to evaluation mode
        model.eval()
        
        print("Successfully created RepLKNet-31B model")
        
    except ImportError as e:
        print(f"Error importing RepLKNet: {e}")
        print(f"Make sure the replknet.py file is accessible")
        return
    except Exception as e:
        print(f"Error creating RepLKNet model: {e}")
        print(f"Exception details: {str(e)}")
        return
    
    input_shape = (3, 224, 224)  # Standard ImageNet input size
    top_dir = 'workloads'
    model_name = f'replknet31b_{batch_size}'
    convert_fc = False  # RepLKNet has a final linear layer
    
    # Define exception module names specific to RepLKNet architecture
    # Based on the actual implementation in replknet.py
    exception_module_names = [
        'view', 'pool', 'avgpool', 'cat', 'concat', 'add', 'mul',
        'relu', 'gelu', 'norm', 'drop', 'Identity', 'DropPath',
        'RepLK', 'IRFFN', 'ConvFFN', 'RepLKBlock', 'RepLKNetStage', 'RepLKNet',
        'ReparamLargeKernelConv', 'ModuleList', 'Sequential',
        'checkpoint', 'flatten', 'stem', 'transitions', 'SyncBatchNorm'
    ]
    
    # Create the workloads directory if it doesn't exist
    os.makedirs(top_dir, exist_ok=True)
    
    print(f"Converting RepLKNet-31B with input shape {input_shape}")
    
    # Convert the model
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
    
    print(f"RepLKNet-31B conversion completed and saved to {top_dir}/{model_name}")

# Main execution
if __name__ == "__main__":
    print("Converting RepLKNet-31B...")
    convert_replknet(batch_size=1)
    print("Conversion completed!")
    print("Converting RepLKNet-31B...")
    convert_replknet(batch_size=32)
    print("Conversion completed!")