# MELI Repository: Used or New Classification Model

Repository with the solution to the "ML New or Used" test for Data Science.

## Presented by: Edwin Jair López López

## Developed in: February 2025

## Contents

The repository contains 3 main folders for the development of the test:

- **notebooks**: contains three notebooks:
    - Data and variable cleaning
    - EDA (Exploratory Data Analysis)
    - Model Calibration.

- **python_scripts**: contains two scripts:
    - Script requested as output for the exercise *new_or_used.py*. This script consolidates the steps documented in the notebooks, giving as a result the model saved in the resources/pkl/ folder.
        - You can run it directly, previously installing the requirements needed and consolidated in the *requirements.txt* file.
        - The output of running this *.py* is in the file *python_scripts/new_or_used_output.txt* 
    - Utility scripts developed for the exercise *ml_utils.py*.

- **resources**: contains the `.pkl` file of the model output.

In the **data** folder, you will find:
- The input file: there's where I put the *MLA_100k_checked_v3.jsonlines* file in the *inputs* folder.
- The output file of the data review *df_final.parquet* in the *staging* folder.

**Notes:**
    - I did not add to .gitignore the .parquet, .jsonlines files so you can replicate all the exercise.
    - In the .pdf file you can find the explanatory document. It contains a really short explanation for the steps taken in the exercise. They are detailed better in each notebook-

