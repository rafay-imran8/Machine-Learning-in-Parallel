# Makefile for Parallelized Machine Learning Pipeline

# Compiler settings
CC = mpicc
NVCC = nvcc
CFLAGS = -fopenmp -O2 -Wall -I./include -lm
CUDA_FLAGS = -lcudart

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include
RESULTS_DIR = results

# Files
COMMON_SRC = $(SRC_DIR)/common.c
EVAL_SRC = $(SRC_DIR)/evaluate.c
DEMO_SRC = $(SRC_DIR)/evaluation_demo.c

COMMON_OBJ = $(OBJ_DIR)/common.o
EVAL_OBJ = $(OBJ_DIR)/evaluate.o
DEMO_OBJ = $(OBJ_DIR)/evaluation_demo.o

# Targets
DEMO_TARGET = $(BIN_DIR)/evaluation_demo
EVAL_LIB = $(BIN_DIR)/libevaluate.a

# Default target
all: directories $(EVAL_LIB) $(DEMO_TARGET)

# Create necessary directories
directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(RESULTS_DIR)

# Compile common utility functions
$(COMMON_OBJ): $(COMMON_SRC) $(INCLUDE_DIR)/common.h
	$(CC) $(CFLAGS) -c $< -o $@

# Compile evaluation module
$(EVAL_OBJ): $(EVAL_SRC) $(INCLUDE_DIR)/evaluate.h $(INCLUDE_DIR)/common.h
	$(CC) $(CFLAGS) -c $< -o $@

# Create evaluation static library
$(EVAL_LIB): $(COMMON_OBJ) $(EVAL_OBJ)
	ar rcs $@ $^

# Compile demo application
$(DEMO_OBJ): $(DEMO_SRC) $(INCLUDE_DIR)/evaluate.h $(INCLUDE_DIR)/common.h
	$(CC) $(CFLAGS) -c $< -o $@

# Link demo application
$(DEMO_TARGET): $(DEMO_OBJ) $(EVAL_LIB)
	$(CC) $(CFLAGS) $< -L$(BIN_DIR) -levaluate -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run demo
run: all
	mpirun -np 4 $(DEMO_TARGET)

# Install (copy library and headers to system directories)
install: $(EVAL_LIB)
	mkdir -p /usr/local/include/ml_pipeline
	cp $(INCLUDE_DIR)/*.h /usr/local/include/ml_pipeline/
	cp $(EVAL_LIB) /usr/local/lib/

.PHONY: all clean run directories install