## notes for TA
    to save your time just look at the project_tf_V3 notebook its the main one backend has the same logic but slightly difference in memory mangment and returns for gui 

# Neural Network Optimization with Evolutionary Algorithms

This repository contains experiments on optimizing neural networks through Differential Evolution (DE) and other evolutionary methods.

## Project Structure

- `code/`
  - `notebooks/` - Contains all Jupyter notebooks
    - `project_tf_V3.ipynb` - Main code for MNIST data experiments
    - `gradient_model/` - Traditional gradient-based models
    - `regression.ipynb` - Regression task experiments using Boston dataset
    - `keras_model.ipynb` - Dynamic models implementation (computationally intensive)
  - `front_react/` - Frontend GUI implementation
  - `backend/` - Python backend services
  - `model.png` and `newplot.png` - Visualization outputs

- `papers/` - Reference papers used in this research

- `documemntation+expirments/` - Documentation, test results, comparisons and our research paper

## Running the GUI

1. For the frontend:
   cd code/front_react
   npm run dev

2. For the backend:
   cd code/backend
   python main.py

3. Access the GUI in your browser

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- Plotly
- Keras
- Node.js (for GUI)
```

