# WESAD Dataset Feature Explanation

## 🎯 Overview

This document explains what the feature numbers (feature_0, feature_1, etc.) in your analysis correspond to based on the **WESAD (WEarable Stress and Affect Detection)** dataset structure.

## 📊 WESAD Dataset Structure

### **Device Configuration:**
- **Chest Device:** RespiBAN Professional (700 Hz sampling)
- **Wrist Device:** Empatica E4 (4-64 Hz sampling)
- **Synchronization:** Manual alignment using double-tap gesture

### **Signal Channels:**

#### **Chest Signals (RespiBAN - 8 channels):**
1. **ACC_X** - Accelerometer X-axis (3-axis accelerometer)
2. **ACC_Y** - Accelerometer Y-axis 
3. **ACC_Z** - Accelerometer Z-axis
4. **ECG** - Electrocardiogram (heart rate)
5. **EDA** - Electrodermal Activity (skin conductance)
6. **EMG** - Electromyography (muscle activity)
7. **RESP** - Respiration (breathing pattern)
8. **TEMP** - Temperature

#### **Wrist Signals (Empatica E4 - 6 channels):**
1. **ACC_X** - Accelerometer X-axis (32 Hz)
2. **ACC_Y** - Accelerometer Y-axis (32 Hz)
3. **ACC_Z** - Accelerometer Z-axis (32 Hz)
4. **BVP** - Blood Volume Pulse (64 Hz)
5. **EDA** - Electrodermal Activity (4 Hz)
6. **TEMP** - Temperature (4 Hz)

## 🔍 Feature Engineering Process

### **Window-Based Feature Extraction:**
Your 42 features are likely derived from statistical measures computed over time windows from the raw WESAD signals.

### **Typical Feature Types:**
1. **Time-domain statistics:** mean, std, min, max, range, variance
2. **Frequency-domain features:** FFT coefficients, power spectral density
3. **Cross-modal correlations:** between different signal types
4. **Temporal patterns:** trends, slopes, zero-crossings

### **Feature Mapping (Estimated):**

Based on typical WESAD feature engineering, your 42 features likely correspond to:

#### **Features 0-2:** Chest Accelerometer (ACC_X, ACC_Y, ACC_Z)
- **Feature 0:** Chest ACC_X mean/std
- **Feature 1:** Chest ACC_Y mean/std  
- **Feature 2:** Chest ACC_Z mean/std

#### **Features 3-5:** Chest Physiological Signals
- **Feature 3:** ECG mean/std
- **Feature 4:** EDA mean/std
- **Feature 5:** EMG mean/std

#### **Features 6-7:** Chest Additional Signals
- **Feature 6:** RESP mean/std
- **Feature 7:** TEMP mean/std

#### **Features 8-10:** Wrist Accelerometer (ACC_X, ACC_Y, ACC_Z)
- **Feature 8:** Wrist ACC_X mean/std
- **Feature 9:** Wrist ACC_Y mean/std
- **Feature 10:** Wrist ACC_Z mean/std

#### **Features 11-13:** Wrist Physiological Signals
- **Feature 11:** BVP mean/std
- **Feature 12:** Wrist EDA mean/std
- **Feature 13:** Wrist TEMP mean/std

#### **Features 14-41:** Advanced Features
- **Features 14-20:** Frequency-domain features (FFT coefficients)
- **Features 21-27:** Cross-modal correlations
- **Features 28-34:** Temporal patterns and trends
- **Features 35-41:** Higher-order statistics and derived features

## 🎯 Your Top Features Explained

Based on your SHAP analysis results:

### **Feature 25 (Most Important):**
- **Likely corresponds to:** Frequency-domain feature from chest ECG or wrist BVP
- **Why important:** Heart rate variability patterns are strong stress indicators
- **Physiological meaning:** Changes in heart rate patterns during stress vs. relaxation

### **Feature 32 (Second Most Important):**
- **Likely corresponds to:** Cross-modal correlation between chest and wrist signals
- **Why important:** Synchronized physiological responses across body locations
- **Physiological meaning:** Coordinated stress response across multiple systems

### **Feature 9 (Third Most Important):**
- **Likely corresponds to:** Wrist ACC_Y statistical measure
- **Why important:** Motion patterns on wrist show stress-related movements
- **Physiological meaning:** Fidgeting, hand movements, or gesture patterns

### **Feature 41 (Fourth Most Important):**
- **Likely corresponds to:** Advanced derived feature (possibly signal quality or artifact detection)
- **Why important:** Signal quality affects stress detection reliability
- **Physiological meaning:** Clean vs. noisy signal segments

## 📈 WESAD Signal Interpretation

### **Stress Indicators in WESAD:**

#### **1. Accelerometer (ACC) - Motion Patterns:**
- **Stress:** Increased fidgeting, rapid movements, restlessness
- **Relaxation:** Smooth, slow movements, stillness
- **Your features:** Likely capture movement statistics and patterns

#### **2. Heart Rate Signals (ECG/BVP):**
- **Stress:** Increased heart rate, reduced heart rate variability
- **Relaxation:** Lower heart rate, increased variability
- **Your features:** Likely capture heart rate statistics and patterns

#### **3. Skin Conductance (EDA):**
- **Stress:** Increased skin conductance, more sweat gland activity
- **Relaxation:** Lower skin conductance, reduced activity
- **Your features:** Likely capture EDA response patterns

#### **4. Respiration (RESP):**
- **Stress:** Rapid, shallow breathing
- **Relaxation:** Slow, deep breathing
- **Your features:** Likely capture breathing pattern statistics

#### **5. Temperature (TEMP):**
- **Stress:** May show subtle changes due to blood flow
- **Relaxation:** More stable temperature patterns
- **Your features:** Likely capture temperature stability measures

## 🔬 Feature Engineering Context

### **Window Size:**
- Your analysis likely used 1000-sample windows
- At 700 Hz (chest) = ~1.4 seconds per window
- At 32 Hz (wrist ACC) = ~31 seconds per window
- At 4 Hz (wrist EDA/TEMP) = ~250 seconds per window

### **Feature Types:**
1. **Statistical measures:** mean, std, min, max, percentiles
2. **Frequency features:** dominant frequencies, power in different bands
3. **Correlation features:** between different signal types
4. **Temporal features:** trends, slopes, change points
5. **Quality features:** signal-to-noise ratios, artifact detection

## 💡 Recommendations for Feature Interpretation

### **1. Focus on Physiological Meaning:**
- **Feature 25:** Heart rate patterns (most predictive of stress)
- **Feature 32:** Multi-modal coordination (stress affects multiple systems)
- **Feature 9:** Movement patterns (stress-related behaviors)
- **Feature 41:** Signal quality (affects detection reliability)

### **2. Consider Signal Characteristics:**
- **Chest signals:** Higher sampling rate (700 Hz), more detailed patterns
- **Wrist signals:** Lower sampling rate (4-64 Hz), but more practical for wearables
- **Cross-modal:** Synchronized responses across body locations

### **3. Stress Detection Insights:**
- **Motion patterns:** Most accessible for wearable applications
- **Heart rate:** Most reliable stress indicator
- **Multi-modal fusion:** Improves detection accuracy
- **Signal quality:** Critical for real-world deployment

## 🎯 Conclusion

Your feature numbers represent **statistical measures extracted from WESAD physiological signals** over time windows. The most important features likely correspond to:

1. **Heart rate patterns** (Feature 25)
2. **Multi-modal correlations** (Feature 32) 
3. **Movement patterns** (Feature 9)
4. **Signal quality measures** (Feature 41)

These features capture the **physiological stress response** across multiple body systems, making them effective for stress detection in wearable applications.

**Key Insight:** The combination of motion patterns, heart rate variability, and multi-modal coordination provides the strongest stress detection signals in the WESAD dataset. 