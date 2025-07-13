import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    """
    A simple two-linear-layer neural network for demonstration purposes.
    Takes a tensor of shape (batch_size, input_size) and outputs
    a tensor of shape (batch_size, output_size).
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 1):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

if __name__ == '__main__':
    # Example usage for direct testing
    print("Testing model instantiation...")
    model = SimpleModel()
    print("Model instantiated successfully:")
    print(model)

    # Test forward pass
    batch_s, input_s = 5, 10
    test_input = torch.randn(batch_s, input_s)
    output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    assert output.shape == (batch_s, model.layer2.out_features)
    print(output)
    print("Forward pass test successful.")