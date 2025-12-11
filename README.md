# EMG-NN Controlled Simulation

Simulation environment for bionic arm control using EMG neural network predictions and myosuite.

## Overview

This repository integrates trained neural network models from the [EMG Classification NN](https://github.com/Red-Dog15/EMG_classification_Neural_Network) with myosuite simulation software to test and visualize bionic arm control in real-time.

## Features

- Integration with myosuite elbow pose environment
- NN-driven action prediction from EMG signals
- Real-time visualization of arm movements
- Performance testing and validation

## Quick Start

### Prerequisites

- Python 3.8+
- myosuite installed
- Trained model from EMG classification repository

### Installation

```bash
pip install myosuite
```

### Running the Simulation

```bash
python main.py
```

## Repository Structure

```
simulation/
├── main.py              # Main simulation entry point
├── README.md            # This file
├── DOCUMENTATION.md     # Detailed technical documentation
└── test.md             # Testing procedures and results
```

## Related Projects

- [EMG Classification Neural Network](https://github.com/Red-Dog15/EMG_classification_Neural_Network) - Source for trained models

## License

TBD

## Authors

Red-Dog15
