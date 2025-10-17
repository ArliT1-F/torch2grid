"""Test gradient visualization."""
import torch
import torch.nn as nn
from torch2grid.gradient_visualizer import (
    visualize_gradient_flow,
    analyze_gradient_health,
    print_gradient_health_report,
    compare_weights_and_gradients
)
from torch2grid.inspector import inspect_torch_object

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)

# Dummy forward/backward pass to generate gradients
x = torch.randn(8, 10)
y = torch.randint(0, 5, (8,))
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()

output = model(x)
loss = criterion(output, y)
loss.backward()

# Collect gradients
gradients = {}
weights = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        gradients[name] = param.grad.detach().cpu().numpy()
        weights[name] = param.detach().cpu().numpy()

print("Collected gradients for", len(gradients), "layers")

# Analyze gradient health
analysis = analyze_gradient_health(gradients)
print_gradient_health_report(analysis)

# Visualize gradient flow
visualize_gradient_flow(gradients)

# Compare weights and gradients
compare_weights_and_gradients(weights, gradients)

print("\n✓ Gradient visualization test complete!")
