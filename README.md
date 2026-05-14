
# CNN-BiLSTM-SE for Water-Flooded Layer Identification

A hybrid deep learning model combining CNN, BiLSTM, and SE attention mechanism for water-flooded layer identification from well logging data.

## Requirements

```bash
pip install numpy pandas scikit-learn tensorflow imbalanced-learn matplotlib openpyxl
```

Or use the full `requirements.txt`:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
tensorflow==2.20.0
imbalanced-learn==0.12.4
matplotlib==3.8.4
openpyxl==3.1.5
```

## Synthetic Data

The `/data` directory contains synthetic datasets for code testing:

| File | Description | Samples |
|------|-------------|---------|
| `synthetic_sample.xlsx` | Training data | 1500 |
| `synthetic_sample_example.xlsx` | Validation data | 300 |

**Note:** Original well logging data are proprietary and cannot be shared due to confidentiality agreements.

