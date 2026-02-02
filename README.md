# CGI_py - Python Implementation of Causality Graphical Inference

This is a Python port of the CGI (Causality Graphical Inference) MATLAB toolbox.

## Overview

CGI is a causal discovery algorithm that uses conditional independence tests based on Gaussian processes and kernel methods to identify causal relationships between variables.

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy >= 1.18.0
- scipy >= 1.5.0

## Usage

```python
import numpy as np
from CGI_py import find_genes_gci, load_data

# Load data (from .mat file)
data = load_data('normalized_Leukemia.mat')

# Run causal gene discovery
results = find_genes_gci(data, alpha=0.05)

# Get causal genes
causal_genes = results['found_genes']
print(f"Found {len(causal_genes)} causal genes")
```

## API

### Core Functions

- `kernel(x, xKern, theta)`: Compute RBF kernel matrix
- `dist2(x, c)`: Compute squared Euclidean distance
- `paco_test(x, y, Z, alpha)`: Partial correlation test
- `kcit(X, Y, Z, ...)`: Kernel conditional independence test
- `fit_gpr(X, Y, cov, hyp, Ncg)`: Fit GP regression model

### Main Algorithm

- `find_genes_gci(data, alpha, cov, Ncg, hyp)`: Find causal genes

## References

- Original MATLAB implementation: [CGI](https://github.com/Causality-Inference/CGI.git)

- Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
  Kernel-based conditional independence test and application in causal discovery.
  arXiv:1202.2775

- Causal Gene Identification Using Non-Linear Regression-Based Independence Tests

## License

See the original CGI repository for license information.
