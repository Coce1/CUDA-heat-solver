# Compilateur CUDA
NVCC = nvcc

# Options de compilation (-O3 pour l'optimisation maximale des performances)
NVCC_FLAGS = -O3 -std=c++14 -I./include

# Dossiers du projet
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Nom de l'exécutable final
TARGET = $(BIN_DIR)/heat_solver_spmv

# Liste des fichiers sources et objets
SRCS = $(SRC_DIR)/heat_matrix_solver.cu $(SRC_DIR)/benchmark_matrix.cu
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

# Règle par défaut (ce qui se lance quand on tape juste "make")
all: directories $(TARGET)

# Règle pour créer les dossiers bin/ et obj/ s'ils n'existent pas
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Règle d'édition de liens (création de l'exécutable à partir des objets)
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Compilation réussie ! L'exécutable est dans $(TARGET)"

# Règle de compilation (création des objets à partir des sources)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/heat_matrix_solver.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Règle de nettoyage (supprime les dossiers générés)
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Fichiers objets et exécutable supprimés."

# Règle pour tout recompiler de zéro
rebuild: clean all

# Indique à Make que ces mots ne sont pas des noms de fichiers
.PHONY: all directories clean rebuild