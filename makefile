CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp
LDFLAGS = -fopenmp

# Target executable
TARGET = hybrid_train

# Source files
SOURCES = 
          random_forest.cpp \
          mlp.cpp \
          logistic_regression.cpp \
          model_training.cpp \
          prediction.cpp \
          main_model.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Header files
HEADERS = logistic_regression.h \
          mlp.h \
          random_forest.h

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS) *.bin

# Run with 3 processes (as required by the code)
run: $(TARGET)
	mpirun -np 3 ./$(TARGET) processed_data.csv

# Phony targets
.PHONY: all clean run