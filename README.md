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
  - For part 3 you should download datasets S1_E1_A1, S1_E1_A2, S1_E1_A3 from this page (https://ninapro.hevs.ch/instructions/DB8.html), and place them into the folder

- **`output/`**  
  Stores the results of the analysis, including visualizations.  
  - Contains matplotlib-generated plots related to data exploration and analysis.

- **`scripts/`**  
  Contains the code for the project.  
  - `part1.ipynb`: The Jupyter notebook to run for the first part of the project.
  - `part2.ipynb`: The Jupyter notebook to run for the second part of the project.
  - `part3.ipynb`: The Jupyter notebook to run for the third part of the project.
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
   pip install numpy pandas scipy matplotlib scikit-learn pywavelets statsmodels
   ```

### License

This project is licensed under the MIT License. See the LICENSE file for more details.  