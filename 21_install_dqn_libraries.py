# Q21. Install required libraries for DQN.
# Terminal command:
# pip install gymnasium torch numpy matplotlib
# Optional rendering support:
# pip install "gymnasium[classic-control]"
import sys
print("Python:",sys.version)
try:
    import torch
    print("PyTorch:",torch.__version__)
except ImportError:
    print("PyTorch is not installed. Run: pip install torch")
try:
    import gymnasium
    print("Gymnasium:",gymnasium.__version__)
except ImportError:
    print("Gymnasium is not installed. Run: pip install gymnasium")
