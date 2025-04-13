CC = g++
CFLAGS = -g -lOpenCL `pkg-config --cflags --libs opencv4`

# Executables
PARA_EXE = paraexe
SHARED_EXE = sharedexe


# Source files
PARA_SRC = 22i0977_Q1_host_global.cpp
SHARED_SRC = 22i0977_Q1_host_shared.cpp


all: $(PARA_EXE) $(SHARED_EXE) 

$(PARA_EXE): $(PARA_SRC)
	$(CC) $(PARA_SRC) -o $(PARA_EXE) $(CFLAGS)

$(SHARED_EXE): $(SHARED_SRC)
	$(CC) $(SHARED_SRC) -o $(SHARED_EXE) $(CFLAGS)



run_para: $(PARA_EXE)
	./$(PARA_EXE)

run_shared: $(SHARED_EXE)
	./$(SHARED_EXE)


clean:
	rm -f $(PARA_EXE) $(SHARED_EXE) $(SERIAL_EXE)

