Compiling model trainer

g++ -std=c++11 your_program.cpp -o model_trainer `pkg-config --cflags --libs opencv4`