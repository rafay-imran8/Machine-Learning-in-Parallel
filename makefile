# Makefile for Hybrid Parallel Machine Learning System
# This makefile builds both the training and prediction executables

# Compiler and flags
CXX = mpicxx
CXXFLAGS = -std=c++17 -fopenmp -Wall -O3 -DOMP_NUM_THREADS=5

# Source files
MAIN_SRC = main_model.cpp
MODEL_SRCS = logistic_regression.cpp mlp.cpp random_forest.cpp
HEADERS = logistic_regression.h mlp.h random_forest.h omp_config.h

# Object files
MODEL_OBJS = $(MODEL_SRCS:.cpp=.o)
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

# Training executable
TRAIN_EXEC = hybrid_ml_trainer

# Prediction executable
PRED_SRC = prediction.cpp
PRED_OBJ = $(PRED_SRC:.cpp=.o)
PRED_EXEC = ml_predictor

# Default target
all: $(TRAIN_EXEC) $(PRED_EXEC)

# Linking the training executable
$(TRAIN_EXEC): $(MAIN_OBJ) $(MODEL_OBJS)
	$(CXX) $(CXXFLAGS) $(MAIN_OBJ) $(MODEL_OBJS) -o $@

# Linking the prediction executable
$(PRED_EXEC): $(PRED_OBJ) $(MODEL_OBJS)
	$(CXX) $(CXXFLAGS) $(PRED_OBJ) $(MODEL_OBJS) -o $@

# Compiling source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the omp_config header if it doesn't exist
omp_config.h:
	echo '#ifndef OMP_CONFIG_H' > $@
	echo '#define OMP_CONFIG_H' >> $@
	echo '#include <omp.h>' >> $@
	echo '' >> $@
	echo '#ifndef OMP_NUM_THREADS' >> $@
	echo '#define OMP_NUM_THREADS 5' >> $@
	echo '#endif' >> $@
	echo '' >> $@
	echo 'inline void setup_openmp_threads() {' >> $@
	echo '    omp_set_num_threads(OMP_NUM_THREADS);' >> $@
	echo '}' >> $@
	echo '' >> $@
	echo '#endif // OMP_CONFIG_H' >> $@

# Clean target
clean:
	rm -f $(MAIN_OBJ) $(MODEL_OBJS) $(PRED_OBJ) $(TRAIN_EXEC) $(PRED_EXEC) *.bin

# Run the training with proper environment settings
run: $(TRAIN_EXEC)
    OMP_NUM_THREADS=5 mpirun --oversubscribe --use-hwthread-cpus -np 3 ./$(TRAIN_EXEC) processed_data.csv

# Run the prediction
predict: $(PRED_EXEC)
	OMP_NUM_THREADS=5 ./$(PRED_EXEC)

# Generate test data if needed
test_data:
	@echo "Generating test data file..."
	@printf "feature1,feature2,feature3,feature4,feature5,label\n" > processed_data.csv
	@for i in $$(seq 1 1000); do \
		f1=$$(echo "scale=2; $$(( RANDOM % 1000 )) / 100" | bc); \
		f2=$$(echo "scale=2; $$(( RANDOM % 1000 )) / 100" | bc); \
		f3=$$(echo "scale=2; $$(( RANDOM % 1000 )) / 100" | bc); \
		f4=$$(echo "scale=2; $$(( RANDOM % 1000 )) / 100" | bc); \
		f5=$$(( RANDOM % 2 )); \
		label=$$(( RANDOM % 2 )); \
		printf "%.2f,%.2f,%.2f,%.2f,%d,%d\n" $$f1 $$f2 $$f3 $$f4 $$f5 $$label >> processed_data.csv; \
	done
	@echo "Generated 1000 random samples in processed_data.csv"

.PHONY: all clean run predict test_data