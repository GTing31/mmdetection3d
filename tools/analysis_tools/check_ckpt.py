import torch
from collections import OrderedDict


def load_ckpt_and_print_architecture(ckpt_path, model_class=None):
    """
    Load a PyTorch checkpoint and print the model architecture.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        model_class (torch.nn.Module, optional): The class of the model.
            If provided, the checkpoint weights will be loaded into this model.
            If not provided, only the weight keys will be printed.
    """
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Check if it contains model state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    else:
        raise ValueError("The checkpoint format is not supported. Check its contents.")

    # Print the keys in the state dict (architecture components)
    print("\n[Architecture Components in Checkpoint]:")
    for key in state_dict.keys():
        print(key)

    # If a model class is provided, load the weights and print the architecture
    if model_class is not None:
        model = model_class()
        model.load_state_dict(state_dict, strict=False)
        print("\n[Model Architecture]:")
        print(model)
    else:
        print("\nModel class not provided. Skipping model loading.")


# Example usage:
# Replace with your own checkpoint path and model class
ckpt_path = '/app/checkpoint/pillarnest_tiny.pth'

# If you have the model class, pass it as the second argument
# from your_model_module import YourModelClass
# load_ckpt_and_print_architecture(ckpt_path, YourModelClass)

# If you don't have the model class
load_ckpt_and_print_architecture(ckpt_path)
