To start the `app.py` file, you should use the **Streamlit** command:

```bash
streamlit run app.py

```

---

# Prognos AI 

### NASA C-MAPSS Turbofan Engine RUL Prediction

This repository contains a machine learning project focused on predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the NASA C-MAPSS dataset. The project includes a comprehensive research phase where multiple deep learning architectures were evaluated to find the best-performing model for real-time health monitoring.

## 📊 About the Dataset

The **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset is a standard benchmark for predictive maintenance. It consists of four sub-datasets (**FD001, FD002, FD003, FD004**) reflecting different operating conditions and fault modes:

* **Data:** Multivariate time-series data from 21 sensors and 3 operational settings.
* **Target:** Remaining Useful Life (RUL) in flight cycles.
* **Complexity:** Ranges from a single operating condition/fault mode (FD001) to six operating conditions and two fault modes (FD004).

## 🚀 The Multi-Model Approach

My approach involved an extensive research phase where I trained and compared several deep learning architectures to capture the temporal dependencies in sensor data:

1. **CNN (1D Convolutional Neural Networks):** Used for feature extraction from the sensor sequences.
2. **LSTM (Long Short-Term Memory):** To capture long-term dependencies in the time-series data.
3. **Stacked LSTM:** Multiple LSTM layers for deeper temporal feature learning.
4. **Bi-LSTM (Bidirectional LSTM):** To process sequences in both forward and backward directions.
5. **GRU (Gated Recurrent Units):** A computationally efficient alternative to LSTM.
6. **CNN + LSTM Hybrid:** A combination model using CNN for local feature extraction followed by LSTM for temporal modeling.

### 📈 Evaluation Metrics

The models were evaluated using the following metrics to ensure high precision in safety-critical aviation environments:

* **RMSE (Root Mean Squared Error):** To measure the average prediction error magnitude.
* **MAE (Mean Absolute Error):** To understand the average absolute deviation from true RUL.
* **R² Score (Coefficient of Determination):** To assess how well the models explain the variance in the engine's degradation.

## 🛠️ Local Installation & Setup

To run the **AeroHealth Manager** dashboard on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/prognos-ai-cmapss.git
cd to the folder

```

### 2. Install Dependencies

Ensure you have `tensorflow`, `streamlit`, and `pandas` installed:

```bash
pip install streamlit pandas numpy tensorflow joblib plotly scikit-learn

```

### 3. Run the Dashboard

Launch the Streamlit application to visualize individual engine inspections and fleet health heatmaps:

```bash
streamlit run app.py

```

## 🖥️ Project Structure

* `research.ipynb`: Initial data exploration and preprocessing.
* `FD001(1).ipynb` to `FD004model.ipynb`: Comprehensive training scripts for the various model architectures.
* `app.py`: Streamlit-based dashboard for real-time RUL prediction and fleet management.
* `*.h5`: Saved trained models for different fleets.
* `scaler_*.pkl`: Pre-trained scalers for data normalization.
