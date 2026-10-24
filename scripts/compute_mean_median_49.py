import re
from pathlib import Path
import numpy as np

# Path to the JSONL file
jsonl_path = Path('outputs/compute_complexity_measure/20250528_053749_model=resnet50_weights=weights/cifar10/pt_imagenet_ft_CIFAR10_resnet50_epoch90.pth/complexity_measure.jsonl')

# Storage for complexity measure values
complexity_values = []

# Read and process the file
with open(jsonl_path, 'r') as f:
    buffer = ''
    brace_level = 0
    
    while True:
        c = f.read(1)
        if not c:
            break
            
        if c == '{':
            if brace_level == 0:
                buffer = ''  # start new object
            brace_level += 1
            
        if brace_level > 0:
            buffer += c
            
        if c == '}':
            brace_level -= 1
            if brace_level == 0:
                # Completed a JSON object in buffer
                # Extract the complexity_measure.value
                cm_match = re.search(r'"complexity_measure"\s*:\s*{[^}]*"value"\s*:\s*([-+]?\d*\.\d+|\d+)', buffer)
                if cm_match:
                    value = float(cm_match.group(1))
                    complexity_values.append(value)
                buffer = ''  # reset

# Calculate statistics
if complexity_values:
    mean_val = np.mean(complexity_values)
    median_val = np.median(complexity_values)
    
    print(f"Number of complexity measure values: {len(complexity_values)}")
    print(f"Mean value: {mean_val:.6f}")
    print(f"Median value: {median_val:.6f}")
else:
    print("No complexity measure values found in the file.")