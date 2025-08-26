# TimesNet with WESAD Dataset

This project sets up TimesNet for training on the WESAD (Wearable Stress and Affect Detection) dataset.

## Setup Instructions

### 1. Data Formatting
First, format your WESAD data for TimesNet:
```bash
python wesad_timesnet_formatter.py
```

This will:
- Load WESAD data from the `WESAD/` directory
- Extract and align physiological signals
- Normalize the data
- Split into train/val/test sets
- Save formatted data to `timesnet_data/`
- Create TimesNet configuration files

### 2. Environment Setup
The setup script will:
- Clone the TimesNet repository
- Install dependencies
- Create custom data loaders
- Set up VS Code configurations

Run the setup:
```bash
python setup_timesnet_vscode.py
```

### 3. Training
Train TimesNet on your WESAD data:
```bash
python train_timesnet_wesad.py
```

Or use VS Code debugger with the provided launch configurations.

## Data Structure

After formatting, your data will be organized as:
```
timesnet_data/
├── train/
│   ├── all_subjects_train_data.npy
│   └── all_subjects_train_labels.npy
├── val/
│   ├── all_subjects_val_data.npy
│   └── all_subjects_val_labels.npy
├── test/
│   ├── all_subjects_test_data.npy
│   └── all_subjects_test_labels.npy
└── all_subjects_metadata.json
```

## Model Configuration

The model is configured for:
- **Task**: Classification (stress/emotion detection)
- **Input**: Multivariate time series (7 physiological signals)
- **Sequence Length**: 1000 time steps
- **Features**: 7 (ACC, BVP, EDA, EMG, ECG, Resp, Temp)
- **Classes**: 4 (baseline, amusement, stress, meditation)

## Troubleshooting

### Memory Issues
- Reduce batch size in `train_timesnet_wesad.py`
- Use CPU if GPU memory is insufficient
- Reduce sequence length in data formatting

### CUDA Errors
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Use CPU: Change device in training script

### Data Shape Errors
- Verify data format: `python -c "import numpy as np; data=np.load('timesnet_data/train/all_subjects_train_data.npy'); print(data.shape)"`
- Expected shape: `(samples, 1000, 7)`

## Performance Tips

1. **GPU Usage**: Enable GPU in training script
2. **Batch Size**: Start with 32, adjust based on memory
3. **Sequence Length**: 1000 is good for WESAD, can be reduced for speed
4. **Data Augmentation**: Consider adding noise or time shifts

## Results

Training will save:
- Best model: `best_model_classification.pth`
- Training curves: `training_curves_classification.png`
- Logs: Console output with loss/accuracy metrics
