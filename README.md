# Predicting Response to Paclitaxel and Carboplatin

This repository contains the code and in-silico data files associated with the research paper titled "Predicting tumor response to paclitaxel and carboplatin using quantitative MRI and mathematical modeling in patient-derived xenografts." The goal of the study is to develop a mathematical model that predicts the response of patient-derived xenografts (PDXs) to paclitaxel and carboplatin using longitudinal tumor volume and baseline imaging biomarkers. 

## Code Structure

The repository is organized as follows:

- `calibration/`: Contains the Python code implementing the mathematical model. The model is based on the exponential function and factors in tumor decay due to treatment. Contains scripts to analyze the data obtained, initial calibration, 1-week treatment calibration, full treatment calibration, biomarkers calibration and figure generation. 

- `data/`: Contains sample in-silico data files to be used for calibration and analysis example. True data was obtained from eight PDX models with tumors, each one including control (n = X) and 4-week treatment groups (n = X). This directory includes all relevant data. 

## Calibration Framework

The calibration framework employed in this study is implemented in Python using the following available libraries: numpy, matplotlib, tqdm, and corner. The model parameters are estimated using a parallel, adaptive, multilevel Markov Chain Monte Carlo (MCMC) sampling method.

## Getting Started

To use the code and reproduce the results:

1. Clone this repository to your local machine.

2. Follow the instructions in the respective directories to compile and run the code.

3. Refer to the README files within each directory for more detailed instructions on how to use the code, preprocess data, and interpret the results.

## Citation

If you use the code or findings from this work, please consider citing the original paper.

## License

GNU General Public License, Version 3, 29 June 2007

## Contact

For any questions or inquiries regarding the code or the research, please contact: rahulrdm13 AT utexas.edu and/or ernesto.lima AT utexas.edu.
