# NSSP - Project 2

## Electromyography (EMG) Signal Analysis for Intention Decoding

### Description

This project focuses on processing and analyzing Electromyography (EMG) signals to decode the intended actions of individuals. This technology plays a vital role in creating effective intention decoders, which are crucial for controlling robotic prostheses and other assistive devices. By working through this project, students will apply knowledge of signal processing and machine learning concepts to a real-world problem involving EMG data.

---

### Project Structure

The repository contains the following directories:

- **`data/`**  
  Contains the dataset for the project.  
  - `s1.zip`: The EMG data for subject 1, provided in compressed `.zip` format.

- **`output/`**  
  Stores the results of the analysis, including visualizations.  
  - Contains matplotlib-generated plots related to data exploration and analysis.

- **`scripts/`**  
  Contains the code for the project.  
  - `main.ipynb`: The primary Jupyter notebook to run the analysis pipeline.  
  - `functions.py`: Includes helper functions for preprocessing and other operations.  
  - `features.py`: Contains functions for feature extraction from EMG signals.

---

### Dependencies

The project uses the following Python libraries:

- **File Handling and Data Manipulation**  
  - `os`  
  - `zipfile`  
  - `pandas`  
  - `numpy`  

- **Signal Processing**  
  - `scipy` (`scipy.io`, `scipy.signal`, `scipy.stats`)  
  - `pywt` (Wavelet Transform for signal analysis)  

- **Machine Learning**  
  - `sklearn` (specifically `train_test_split`)  

- **Visualization**  
  - `matplotlib.pyplot`  

---

### Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install the required libraries**  
   Ensure Python 3.7+ is installed. Install the dependencies using `pip`:
   ```bash
   pip install numpy pandas scipy matplotlib scikit-learn pywavelets
   ```

3. **Run the analysis**  
   Open and execute the Jupyter notebook `main.ipynb` in the `scripts/` folder to process the data and generate outputs.  
   Note: The `main.ipynb` notebook automatically handles unzipping the dataset from `data/s1.zip`.

   ```bash
   jupyter notebook scripts/main.ipynb
   ```

---

### Key Functionalities

1. **Data Preprocessing**  
   - EMG data extraction from `.mat` files.  
   - Signal filtering using Butterworth filters for noise reduction.

2. **Visualization**  
   - Plotting raw and processed signals to analyze trends and patterns. 

3. **Feature Extraction**  
   - Application of wavelet transforms (`pywt`) for feature derivation.  
   - Statistical feature computation (e.g., mean, variance, mode).

4. **Train-Test Split**  
   - Dividing data into training and testing subsets for downstream tasks.

---

### License

This project is licensed under the MIT License. See the LICENSE file for more details.  