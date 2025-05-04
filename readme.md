````markdown
# Loan Approval Classifier

This project includes a machine learning pipeline for classifying loan approvals using the k-Nearest Neighbors (k-NN) algorithm from OpenCV's ML module. The training process is parallelized using OpenMP to speed up preprocessing.

## Files

- `model_training.cpp`  
  Trains a k-NN classifier using preprocessed loan data (`processed_data.csv`) and saves the model as `trained_knn.yml`.

- `prediction.cpp`  
  Loads the trained model and makes predictions based on user input or hardcoded sample data.

---

## Requirements

- OpenCV 4.x
- GCC with OpenMP support
- `processed_data.csv` (CSV file with loan data: 5 features + 1 label)

---

## Compilation

### Compile the training module (parallelized with OpenMP)

```bash
g++ model_training.cpp -o classifier -fopenmp `pkg-config --cflags --libs opencv4`
````

### Compile the prediction module

```bash
g++ prediction.cpp -o predictor `pkg-config --cflags --libs opencv4`
```

---

## Usage

### Step 1: Prepare your data

Ensure `processed_data.csv` is in the working directory. The CSV should contain:

* 5 float-type features
* 1 integer label (loan approval status) in the 6th column

### Step 2: Train the model

```bash
./classifier
```

This will output the predicted label for a sample input and save the trained model to `trained_knn.yml`.

### Step 3: Make predictions

```bash
./predictor
```

This will load `trained_knn.yml` and predict based on the input vector (which you can modify in `prediction.cpp`).

---

## Notes

* The training code uses OpenMP to parallelize data conversion to OpenCV matrices.
* You can modify `prediction.cpp` to accept user input via command line or file if desired.

```

