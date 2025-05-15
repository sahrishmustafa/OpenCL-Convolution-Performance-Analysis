# README

## Project Overview
This project implements both serial and parallel versions of a convolution operation using OpenCL. It compares the performance of different implementations and verifies results against OpenCV and SciPy.

## File Structure
```
Project_Folder/  
│── Report.pdf  
│── README.md  
├── Scalar/  
│   ├── Q1/  
│   │   ├── 22i0977_Q1_serial.cpp  
│   │   ├── Makefile  
├── OpenCL/  
│   ├── 22i0977_Q1_host_global.cpp  # Parallel (Global Memory) implementation  
│   ├── 22i0977_Q1_host_shared.cpp  # Parallel (Shared Memory) implementation  
│   ├── conv.cl  # OpenCL kernel
│   ├── check_results.py  # results verify (one file each)
│   ├── Makefile  

```

## Dataset
```
├── dataset/
│   ├── grayscale/
│   │   ├── 512/      # Contains 512x512 grayscale images
│   │   ├── 1024/     # Contains 1024x1024 grayscale images
```


## Compilation and Execution
### Question 1 (OpenCL Implementations)
To compile and execute the different implementations:

**Serial Version:**
```bash
make serial
./serialexe
```

**Parallel (Global Memory) Version:**
```bash
make global
./paraexe
```

**Parallel (Shared Memory) Version:**
```bash
make shared
./sharedexe
```

## Verification
To verify the convolution results, run:
```bash
python check_results.py
```
