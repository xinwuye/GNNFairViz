# GNNFairViz

![Build Status](https://github.com/xinwuye/GNNFairViz/actions/workflows/python-publish.yml/badge.svg) ![License](https://img.shields.io/github/license/xinwuye/GNNFairViz)

## Overview

GNNFairViz is a visualization tool designed to provide insights into the fairness of Graph Neural Networks from the perspective of data. 

## Installation

You can install GNNFairViz using pip or from source.

### Using pip

```bash
pip install gnnfairviz
```

### From Source

```bash
git clone https://github.com/xinwuye/GNNFairViz.git
cd GNNFairViz
pip install .
```

## Usage

Examples of how to use the package can be found in the `evaluation/cases` folder.

## Features

- Support customizing and inspecting fairness through various
viewpoints.
- Provide clues and interactions for node selection to analyze how
they affect model bias.
- Allow diagnosing GNN fairness issues in an interactive manner.

## Contributing

We welcome contributions! Follow these steps to set up your development environment and contribute to the project.

### Setting Up the Development Environment

```bash
git clone https://github.com/xinwuye/GNNFairViz.git
cd GNNFairViz
poetry install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:

- **Email**: xwye23@m.fudan.edu.cn
- **GitHub Issues**: [GitHub Issues](https://github.com/xinwuye/GNNFairViz/issues)

## Credits

This project uses and adapts code from the following repositories:

- [CSC591_Community_Detection by imabhishekl](https://github.com/imabhishekl/CSC591_Community_Detection)
- [NoLiES by leitte](https://github.com/leitte/NoLiES)
- [PyGDebias by yushundong](https://github.com/yushundong/PyGDebias)