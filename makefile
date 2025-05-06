# Makefile for Hybrid Parallel Machine Learning System
# This makefile builds the preprocessing, training and prediction executables

# Compiler and flags
CXX = mpicxx
CXXFLAGS = -std=c++17 -fopenmp -Wall -O3 -DOMP_NUM_THREADS=5

# Include directory
INCDIR = include

# Source directories
SRCDIR = src

# Source files
MAIN_SRC = main.cpp
MAIN_MODEL_SRC = main_model.cpp
PREPROCESSOR_SRC = loan_data_preprocessor.cpp
MODEL_SRCS = logistic_regression.cpp mlp.cpp random_forest.cpp
PRED_SRC = prediction.cpp

# Object files with their paths
MAIN_OBJ = main.o
MAIN_MODEL_OBJ = main_model.o
PREPROCESSOR_OBJ = loan_data_preprocessor.o
MODEL_OBJS = logistic_regression.o mlp.o random_forest.o
PRED_OBJ = prediction.o

# Executables
TRAIN_EXEC = hybrid_ml_trainer
PREPROCESSOR_EXEC = loan_preprocessor
PRED_EXEC = ml_predictor

# Default target
all: $(PREPROCESSOR_EXEC) $(TRAIN_EXEC) $(PRED_EXEC)

# Rule for creating the include directory
$(INCDIR)/omp_config.h:
	mkdir -p $(INCDIR)
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

# Linking the preprocessing executable
$(PREPROCESSOR_EXEC): $(MAIN_OBJ) $(PREPROCESSOR_OBJ)
	$(CXX) $(CXXFLAGS) -I. $^ -o $@

# Linking the training executable
$(TRAIN_EXEC): $(MAIN_MODEL_OBJ) $(MODEL_OBJS)
	$(CXX) $(CXXFLAGS) -I. $^ -o $@

# Linking the prediction executable
$(PRED_EXEC): $(PRED_OBJ) $(MODEL_OBJS)
	$(CXX) $(CXXFLAGS) -I. $^ -o $@

# Compiling source files with correct include paths
main.o: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

main_model.o: $(SRCDIR)/main_model.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

loan_data_preprocessor.o: $(SRCDIR)/loan_data_preprocessor.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

logistic_regression.o: $(SRCDIR)/logistic_regression.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

mlp.o: $(SRCDIR)/mlp.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

random_forest.o: $(SRCDIR)/random_forest.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

prediction.o: $(SRCDIR)/prediction.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

# Clean target
clean:
	rm -f $(MAIN_OBJ) $(MAIN_MODEL_OBJ) $(PREPROCESSOR_OBJ) $(MODEL_OBJS) $(PRED_OBJ) $(TRAIN_EXEC) $(PREPROCESSOR_EXEC) $(PRED_EXEC) *.bin

# Process raw loan data
preprocess: $(PREPROCESSOR_EXEC)
	./$(PREPROCESSOR_EXEC) loan_data.csv processed_data.csv

# Run the training with proper environment settings (with oversubscribe for GitHub Codespaces)
train: $(TRAIN_EXEC)
	OMP_NUM_THREADS=5 mpirun --oversubscribe -np 3 ./$(TRAIN_EXEC) processed_data.csv

# Run the prediction
predict: $(PRED_EXEC)
	OMP_NUM_THREADS=5 ./$(PRED_EXEC)

# Full workflow
workflow: preprocess train predict

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

.PHONY: all clean preprocess train predict workflow test_data