# Deep Frequency Filtering for Domain Generalization

This repository contains an implementation of the Deep Frequency Filtering (DFF) method for domain generalization. DFF improves the robustness of deep neural networks across different domains by selectively filtering frequency components in the latent space.

## Overview

Domain generalization is essential when models are trained on data from certain domains but must perform well on unseen ones. The DFF method leverages the frequency domain to filter out domain-specific details while preserving general, transferable features. This is achieved through:
- **Frequency Transformation:** Converting spatial features to the frequency domain via FFT.
- **Selective Attention:** Learning instance-adaptive masks to modulate frequency components.
- **Two-Branch Architecture:** Combining original spatial features with frequency-filtered features.

## Original Paper

For more details on the method, please refer to the original paper:  
[Deep Frequency Filtering for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Deep_Frequency_Filtering_for_Domain_Generalization_CVPR_2023_paper.pdf)

## Features

- **Global Frequency Filtering:** Uses FFT to obtain a global view of frequency components.
- **Selective Attention:** Applies a 1×1 convolution, batch normalization, and ReLU to create an adaptive spatial mask.
- **Efficient Architecture:** The latent space is smaller than the raw input, reducing computational cost while retaining essential information.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

(Additional dependencies are listed in `requirements.txt`.)

## Dataset
https://www.kaggle.com/api/v1/datasets/download/pengcw1/market-1501

## Installation

Clone the repository and install the required packages:
```bash
git clone https://github.com/RisithPerera/dff-implementation.git
cd dff-implementation
pip install -r requirements.txt
```

## Usage

### Training
Run the training script to train the DFF model:
```bash
python train.py
```

### Evaluation and Visualization
After training, run the evaluation script to visualize the feature maps and learned masks:
```bash
python evaluate.py
```

## Code Structure

- **train.py:** Training script.
- **evaluate.py:** Evaluation and visualization script.
- **models/:** Contains implementations of the ResNet-based models (with and without DFF).
- **datasets/:** Custom dataset loader for the Market1501 dataset.
- **utils/:** Utility functions for data processing and visualization.

## Citation

If you use this code in your research, please cite the original paper:

```
@inproceedings{lin2023deep,
  title={Deep frequency filtering for domain generalization},
  author={Lin, Shiqi and Zhang, Zhizheng and Huang, Zhipeng and Lu, Yan and Lan, Cuiling and Chu, Peng and You, Quanzeng and Wang, Jiang and Liu, Zicheng and Parulkar, Amey and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11797--11807},
  year={2023}
}
```

## Acknowledgements

Thanks to the original authors for their work, which inspired this implementation.

---