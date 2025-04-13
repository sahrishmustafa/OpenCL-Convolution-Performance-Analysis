# Compiler
CXX = g++

# Compilation flags
CXXFLAGS = -fopenmp `pkg-config --cflags --libs sdl2`

# Target executable
TARGET = serialexe

# Source file
SRC = 22i0977_Q2_serial.cpp

# Default target: compile and run the serial program
all: $(TARGET) run

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the compiled executable
run: $(TARGET)
	./$(TARGET)

# Clean compiled files
clean:
	rm -f $(TARGET)

