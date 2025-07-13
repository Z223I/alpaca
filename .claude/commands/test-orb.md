---
description: "Run all ORB-related tests with conda environment activation"
allowed-tools: ["bash"]
---

# Run ORB Tests

This command runs all ORB-related tests in the proper conda environment.

```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && ./test.sh orb
```