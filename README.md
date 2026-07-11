# House Value Prediction Pipeline


This project implements a machine learning pipeline to predict median house values based on housing data. It uses a `RandomForestRegressor` and automated preprocessing to ensure data consistency and accuracy.


## Project Structure
- `main.py`: The core script that handles data preprocessing, training, and inference.
- `data/`: Folder containing your datasets (`housing.csv`, `input.csv`).
- `models/`: Folder where the trained model (`model.pkl`) and pipeline (`pipeline.pkl`) are stored.
- `requirements.txt`: List of dependencies required to run the project.


## Prerequisites
Ensure you have Python installed on your machine. It is recommended to use a virtual environment. Install the required libraries using:


```bash
pip install -r requirements.txt
```

##How to Run
1. Training
If you have housing.csv in your data/ folder, run the script. It will detect that no model exists, train a new RandomForestRegressor, and save the trained model and pipeline to the models/ folder:

```bash
python main.py
```

2. Inference
Once the model is saved, you can run the script again. It will automatically load the existing model, process the data in data/input.csv, and save the predictions to data/output.csv:


```bash
python main.py
```

Key Features
Automation: The script automatically toggles between training and inference modes based on the presence of the model file.

Preprocessing Pipeline: Uses scikit-learn's Pipeline and ColumnTransformer to handle numerical imputation, scaling, and categorical one-hot encoding efficiently.

Cross-Platform Compatibility: Uses os.path.join for file handling to ensure the code works seamlessly on different operating systems.



Security & Version Control
This project utilizes a .gitignore file to ensure that sensitive data (data/) and heavy binary files (models/) are kept locally and not uploaded to GitHub, maintaining a clean and secure repository.



Author
Snehanshi Chaudhury
