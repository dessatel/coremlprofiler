# CoreML (Python) Profiler 🐍

Quick and easy profiling of CoreML models in Python!

## Quick Start ⚡️

```python
# Convert model using guide: https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html
# Or download one from the 🤗 Hub 
from coremlprofiler import CoreMLProfiler
from huggingface_hub import snapshot_download

repo_path = snapshot_download(repo_id="FL33TW00D-HF/test-st", local_dir="./")
model_path = repo_path + "/sentence_transformer_all-MiniLM-L6-v2.mlpackage"

# Create the Profiler
profiler = CoreMLProfiler(model_path)

# Print your device usage
print(profiler.device_usage_summary_chart())
# Enjoy your nice plot!
```
![Python Profiling](https://github.com/user-attachments/assets/c9c7eb77-6a21-4ae7-a3a5-4f80ce8a1d8f)

## CLI

Profile a local model:

```bash
coremlprofile path/to/my/model.mlpackage
```

Download from Hugging Face and profile:

```bash
coremlprofile --hf_repo="FL33TW00D-HF/test-st" sentence_transformer_all-MiniLM-L6-v2.mlpackage
```

## Experimental Multi-Function Support

For models with multiple functions, you can specify which function to analyze:

```bash
# List available functions
coremlprofile path/to/model.mlpackage -l

# Analyze specific function
coremlprofile path/to/model.mlpackage -f function_name -r all
```

Available report types:
- `usage`: Device usage summary
- `chart`: Visual device usage chart
- `operators`: Operator compatibility table
- `specs`: Model specifications
- `all`: All of the above

