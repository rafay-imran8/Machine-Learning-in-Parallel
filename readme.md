# 1. Build all targets as defined in the default Makefile
make all

 2. Run the data preprocessor:
    - takes raw CSV (loan_data.csv)
    - outputs cleaned/feature-engineered CSV (processed_data.csv)
./loan_preprocessor loan_data.csv processed_data.csv

 3. Train your hybrid model (MPI-parallelized):
    - “--oversubscribe” lets MPI spawn more processes than physical cores
    - “-np 3” launches 3 parallel ranks
    - trains on processed_data.csv
mpirun --oversubscribe -np 3 ./hybrid_ml_trainer processed_data.csv

 ── OR ──
 If you prefer CLI flags instead of positional args:
    --data            : path to processed dataset
    --trees           : number of trees for the random forest component
    --hidden          : comma-separated list of hidden layer sizes for MLP
    --epochs          : number of training epochs
    --learning-rate   : learning rate for optimizer
    --output          : directory to write models
mpirun --oversubscribe -np 3 ./hybrid_ml_trainer \
  --data processed_data.csv \
  --trees 100 \
  --hidden "10,5" \
  --epochs 150 \
  --learning-rate 0.01 \
  --output ./

 4. Run the prediction CLI against your freshly trained models
./ml_predictor

 5. Build and run your evaluation pipeline:
    a) compile evaluate targets from makefile_evaluate
make -f makefile_evaluate

   b) launch model evaluator under MPI:
       - “-np 2” uses two processes (e.g. one per model)
       - arguments: processed data + paths to each serialized model
mpirun --oversubscribe -np 2 \
  ./model_evaluator \
    processed_data.csv \
    ./mlp_model.bin \
    ./logistic_regression_model.bin
