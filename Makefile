# Compilateur CUDA
NVCC ?= nvcc

# Architecture GPU : sm_75 = T4. Surcharger : make ARCH=sm_86  (RTX 30xx), sm_89, sm_90...
# -arch=sm_XX embarque aussi le PTX : un GPU plus récent peut recompiler à la volée.
ARCH ?= sm_75

NVCC_FLAGS = -O3 -std=c++14 -arch=$(ARCH) -lineinfo -I./include

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

TARGET = $(BIN_DIR)/heat_solver

SRCS_CU  = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/heat_gpu_naive.cu $(SRC_DIR)/heat_gpu_ellpack.cu \
           $(SRC_DIR)/heat_gpu_stencil.cu $(SRC_DIR)/heat_gpu_shared.cu
SRCS_CPP = $(SRC_DIR)/heat_cpu.cpp

OBJS_CU  = $(patsubst $(SRC_DIR)/%.cu,  $(OBJ_DIR)/%.o, $(SRCS_CU))
OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))
OBJS     = $(OBJS_CU) $(OBJS_CPP)

# Tout header modifié => recompilation de tous les objets
HEADERS = $(wildcard $(INC_DIR)/*.cuh)

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Compilation réussie : $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

rebuild: clean all

.PHONY: all clean rebuild
