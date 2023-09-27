# Three Voice Clustering

Welcome to the "Three Voice Clustering" project! This project aims to automatically cluster audio files based on their corresponding voices. 

This README provides an overview of the project structure and how to use it.

## Project Structure

The project is organized into the following structure:


- `audio/`: This folder contains all the audio files that you want to cluster based on their voices.
You can place your audio files here.

- `main.py`: This is the main script of the project.
It accepts a command-line argument `--folder_path` to specify the folder containing audio files.
Running this script initiates the clustering process.

- `utils.py`: This file contains utility functions used for audio processing,
feature extraction, and clustering. 

- `report.ipynb`: This Jupyter Notebook report documents the project's steps,
including visualizations, audio feature extraction, and clustering.
It provides a comprehensive overview of the project's workflow and results.

- `requirements.txt`: This file lists the Python dependencies required to run the project.
You can use it to create a virtual environment and install the necessary packages.

## Getting Started

To get started with the "Three Voice Clustering" project, follow these steps:

1. Install the project dependencies using `pip`:
```
pip install -r requirements.txt
```
2. Run the main script with the `--folder_path` argument to specify
the folder containing audio files for clustering:
```
python3 main.py --folder_path audio
```
3. The script will perform audio clustering and generate
a JSON file with the clustering results.

## Report
For a detailed explanation of the project, feature extraction techniques,
clustering methods, and visualization, please refer to the `report.ipynb`
Jupyter Notebook in this repository.
