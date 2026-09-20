# Compilateur CUDA
NVCC = nvcc

# Options de compilation
NVCC_FLAGS = -O3 -std=c++14 -I./include

# Dossiers du projet
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Nom de l'exécutable final
TARGET = $(BIN_DIR)/heat_solver

# Fichiers sources basés sur ton arborescence
SRCS_CU = $(SRC_DIR)/benchmark.cu $(SRC_DIR)/heat_gpu_naive.cu $(SRC_DIR)/heat_gpu_shared.cu
SRCS_CPP = $(SRC_DIR)/heat_cpu.cpp

# Fichiers objets correspondants
OBJS_CU = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS_CU))
OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))
OBJS = $(OBJS_CU) $(OBJS_CPP)

# Règle par défaut
all: directories $(TARGET)

# Création des dossiers
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Édition de liens (Création de l'exécutable)
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Compilation réussie ! L'exécutable est dans $(TARGET)"

# Règles de compilation pour les fichiers .cu
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Règles de compilation pour les fichiers .cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Nettoyage
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Fichiers objets et exécutable supprimés."

rebuild: clean all

.PHONY: all directories clean rebuild