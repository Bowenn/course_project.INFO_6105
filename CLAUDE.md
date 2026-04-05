# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a coursework repository for **INFO 6105** (Spring 2026) at Northeastern University. It contains problem sets and projects as Jupyter notebooks and PDF documents — there is no buildable application or test suite.

## Repository Structure

- `problem_set1/` — Problem Set 1: Jupyter notebook (`PS-1-code.ipynb`) + PDF question/answer sheets
- `problem_set2/` — Problem Set 2 (placeholder, empty)
- `midterm/` — Midterm project: house price prediction using Ridge Regression (`midterm_code&presentation.ipynb` + `.docx` report)
- `final/` — Final project (placeholder, empty)

## Tech Stack

- **Python** with Jupyter notebooks (`.ipynb`)
- Key libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, mlxtend
- Datasets are loaded from Google Drive URLs within notebooks

## Working with Notebooks

Run notebooks with Jupyter:
```
jupyter notebook
```

## Key Patterns in Existing Code

- The midterm project follows a standard ML pipeline: data loading, EDA with seaborn pairplots, preprocessing (log transforms, outlier removal, feature engineering), model training (Ridge with cross-validation via `RidgeCV`), bias-variance analysis via `mlxtend.evaluate.bias_variance_decomp`, and prediction export to CSV.
- `StandardScaler` is used before Ridge regression. Test data is transformed with the same scaler fitted on training data.
- Target variable (`price`) is log-transformed; predictions are exponentiated back before export.
