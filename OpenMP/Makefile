# Compiler
CXX = g++

# Compilation flags
CXXFLAGS = -fopenmp `pkg-config --cflags --libs sdl2`

# Target executable
TARGET = parallelexe

# Source file
SRC = 22i0977_Q2_parallel.cpp

# Default target: compile and run the parallel program
all: $(TARGET) run

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the compiled executable
run: $(TARGET)
	./$(TARGET)

# Clean compiled files
clean:
	rm -f $(TARGET)

