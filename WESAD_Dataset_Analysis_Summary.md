# WESAD Dataset Analysis Summary

## Dataset Overview

The WESAD (Wearable Stress and Affect Detection) dataset contains physiological signals from 15 subjects (S2-S17) with stress and affect detection labels. This analysis focuses on Subject S2 as the primary example.

## Data Structure

### Main Components
- **signal**: Contains physiological signals from chest and wrist sensors
- **label**: Stress/affect labels for each time point
- **subject**: Subject identifier

### Signal Types Available

#### Chest Signals (4,255,300 samples each)
- **ACC**: 3-axis accelerometer data
- **ECG**: Electrocardiogram (heart rate)
- **EMG**: Electromyography (muscle activity)
- **EDA**: Electrodermal activity (skin conductance)
- **Temp**: Temperature
- **Resp**: Respiration

#### Wrist Signals (varying lengths)
- **ACC**: 3-axis accelerometer (194,528 samples)
- **BVP**: Blood volume pulse (389,056 samples)
- **EDA**: Electrodermal activity (24,316 samples)
- **TEMP**: Temperature (24,316 samples)

## Label Distribution

| Label | Name | Count | Percentage |
|-------|------|-------|------------|
| 0 | Baseline | 2,142,701 | 50.4% |
| 1 | Stress | 800,800 | 18.8% |
| 2 | Amusement | 430,500 | 10.1% |
| 3 | Meditation | 253,400 | 6.0% |
| 4 | Label 4 | 537,599 | 12.6% |
| 6 | Label 6 | 45,500 | 1.1% |
| 7 | Label 7 | 44,800 | 1.1% |

## Signal Quality Analysis

### Data Quality
- **No Missing Values**: All signals have 0% NaN values
- **No Infinite Values**: All signals have 0% Inf values
- **Clean Data**: The dataset is well-preprocessed and ready for analysis

### Signal Characteristics

#### Chest Signals (4,255,300 samples each)
- **ECG**: Heart rate signal, range [-1.5, 1.5], mean ≈ 0
- **Resp**: Respiration signal, range [-28, 27], mean ≈ 0
- **EDA**: Skin conductance, range [0.26, 7.58], mean ≈ 1.53
- **Temp**: Body temperature, range [28.0, 34.4]°C, mean ≈ 30.9°C
- **EMG**: Muscle activity, range [-0.42, 0.30], mean ≈ -0.002
- **ACC**: 3-axis acceleration, normalized values

#### Wrist Signals
- **BVP**: Blood volume pulse, range [-874, 988], mean ≈ 0
- **TEMP**: Wrist temperature, range [32.3, 36.0]°C, mean ≈ 34.4°C
- **EDA**: Wrist skin conductance, range [0.05, 1.72], mean ≈ 0.39
- **ACC**: 3-axis acceleration, raw values [-128, 127]

## Key Findings

### 1. Signal Length Differences
- **Chest signals**: All have 4,255,300 samples (full duration)
- **Wrist signals**: Have different lengths due to different sampling rates
  - ACC: 194,528 samples
  - BVP: 389,056 samples  
  - EDA/TEMP: 24,316 samples

### 2. Signal Correlations (Top 10)
1. **chest_EDA ↔ wrist_TEMP**: 0.650 (Strong positive)
2. **chest_EDA ↔ chest_Temp**: 0.556 (Strong positive)
3. **chest_Temp ↔ wrist_TEMP**: 0.532 (Strong positive)
4. **wrist_ACC ↔ wrist_EDA**: -0.371 (Moderate negative)
5. **chest_Temp ↔ wrist_EDA**: 0.353 (Moderate positive)
6. **chest_ACC ↔ wrist_TEMP**: -0.337 (Moderate negative)
7. **chest_Temp ↔ wrist_ACC**: -0.292 (Weak negative)
8. **wrist_EDA ↔ wrist_TEMP**: -0.216 (Weak negative)
9. **chest_ACC ↔ chest_EDA**: -0.195 (Weak negative)
10. **chest_ACC ↔ wrist_EDA**: 0.149 (Weak positive)

### 3. Important Insights

#### Temperature Correlation
- Strong correlation between chest and wrist temperature (0.532)
- Both temperature signals correlate with EDA (skin conductance)
- This suggests physiological coherence between different body locations

#### Heart Rate Signals
- **ECG (chest)** and **BVP (wrist)** are complementary heart rate measures
- ECG is more reliable but requires chest placement
- BVP is more convenient but may be noisier
- Both signals have similar characteristics (mean ≈ 0, similar ranges)

#### Respiration and Heart Rate
- Respiration signal shows natural breathing patterns
- Can be used to validate heart rate signal quality
- Respiration and heart rate are physiologically linked

## Recommendations for Analysis

### 1. Signal Selection
- **Primary heart rate**: Use ECG (chest) as primary, BVP (wrist) as secondary
- **Temperature**: Use chest temperature as primary (more stable)
- **EDA**: Use chest EDA as primary (longer duration)
- **Movement**: Use chest ACC for activity detection

### 2. Data Preprocessing
- **No missing value handling needed** - data is clean
- **Normalization**: Consider normalizing signals to [0,1] range
- **Segmentation**: Use label boundaries for segmenting different states

### 3. Feature Engineering
- **Time-domain features**: Mean, std, min, max for each signal
- **Frequency-domain features**: FFT coefficients for periodic signals
- **Cross-signal features**: Correlation-based features between signals

### 4. Model Considerations
- **Imbalanced classes**: Baseline (50.4%) vs Stress (18.8%)
- **Temporal nature**: Consider LSTM/CNN for sequence modeling
- **Multi-modal**: Combine chest and wrist signals for better performance

## Generated Visualizations

1. **wesad_full_dataset_S2.png**: Complete signal plots showing all physiological signals
2. **wesad_full_correlations_S2.png**: Correlation heatmap between all signals
3. **wesad_basic_plots_S2.png**: Basic signal overview plots

## Next Steps

1. **Multi-subject analysis**: Extend analysis to all 15 subjects
2. **Temporal analysis**: Analyze signal patterns over time
3. **Feature extraction**: Extract meaningful features from signals
4. **Model development**: Build stress/affect detection models
5. **Cross-validation**: Validate findings across subjects

## Data Quality Assessment

✅ **Excellent**: No missing values, no infinite values, clean signals
✅ **Well-structured**: Clear organization with chest/wrist separation
✅ **Rich information**: Multiple complementary signals
✅ **Balanced**: Good representation of different affective states
⚠️ **Length differences**: Wrist signals have shorter durations
⚠️ **Class imbalance**: Baseline class dominates the dataset 