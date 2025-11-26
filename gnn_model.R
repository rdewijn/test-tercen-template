# ===============================================================================
# GNN Model Definition: Substrate-Kinase-Pathway-Response Architecture
# TRUE GRAPH CONVOLUTION IMPLEMENTATION ONLY
# ===============================================================================

# Load required libraries for model building
library(keras)
library(tensorflow)
library(Matrix)
library(dplyr)  # For %>% operator

# ===============================================================================
# GNN MODEL CONFIGURATION
# ===============================================================================

# Default model hyperparameters
default_config <- list(
  # Layer dimensions
  substrate_dim = 100,    # Input layer: number of substrate features
  kinase_dim = 64,        # Hidden layer 1: kinase representation
  pathway_dim = 32,       # Hidden layer 2: pathway representation
  response_dim = 1,       # Output layer: binary classification
  
  # Training parameters
  learning_rate = 0.001,
  
  # Regularization
  dropout_rate = 0.3,
  l2_reg = 0.01
)

# ===============================================================================
# TRUE GRAPH CONVOLUTION IMPLEMENTATION
# ===============================================================================

# Function to create true graph convolution layer with adjacency matrix
create_graph_conv_layer <- function(adjacency_matrix, units, 
                                   activation = 'relu',
                                   use_bias = TRUE,
                                   kernel_regularizer = NULL,
                                   name = NULL) {
  
  # This implements: A * X * W
  # Where A = normalized adjacency matrix, X = input features, W = learnable weights
  
  layer_lambda(
    function(x) {
      # Convert adjacency matrix to tensor
      adj_tensor <- k_constant(adjacency_matrix)
      
      # Apply graph convolution: A * X
      graph_conv_output <- k_dot(adj_tensor, x)
      
      return(graph_conv_output)
    },
    name = paste0(name, "_graph_conv")
  ) %>%
  layer_dense(
    units = units,
    activation = activation,
    use_bias = use_bias,
    kernel_regularizer = kernel_regularizer,
    name = name
  )
}

# ===============================================================================
# ADJACENCY MATRIX FUNCTIONS
# ===============================================================================

# Function to normalize adjacency matrix (Laplacian normalization)
normalize_adjacency <- function(adj_matrix) {
  # Calculate degree matrix
  degree <- Matrix::rowSums(adj_matrix)
  degree[degree == 0] <- 1  # Avoid division by zero
  
  # D^(-1/2) * A * D^(-1/2)
  degree_sqrt_inv <- 1 / sqrt(degree)
  degree_matrix <- Matrix::Diagonal(x = degree_sqrt_inv)
  
  normalized_adj <- degree_matrix %*% adj_matrix %*% degree_matrix
  return(as.matrix(normalized_adj))
}

# Function to create weighted adjacency matrix from edge list
create_weighted_adjacency_matrix <- function(edge_list_with_weights, n_nodes) {
  adj_matrix <- Matrix::sparseMatrix(
    i = edge_list_with_weights[,1],     # Source nodes
    j = edge_list_with_weights[,2],     # Target nodes  
    x = edge_list_with_weights[,3],     # Edge weights
    dims = c(n_nodes, n_nodes)
  )
  
  # Make symmetric and add self-loops
  adj_matrix <- adj_matrix + Matrix::t(adj_matrix)
  Matrix::diag(adj_matrix) <- 1
  
  return(adj_matrix)
}

# ===============================================================================
# GNN MODEL ARCHITECTURE
# ===============================================================================

# Main function to build the 4-layer GNN model with TRUE graph convolution
build_gnn_model <- function(config = default_config, adjacency_matrices) {
  
  # Check if adjacency matrices are provided (required for true GNN)
  if (is.null(adjacency_matrices)) {
    stop("Adjacency matrices are required for true GNN implementation!")
  }
  
  # Extract and normalize adjacency matrices
  substrate_kinase_adj <- normalize_adjacency(adjacency_matrices$substrate_kinase)
  kinase_pathway_adj <- normalize_adjacency(adjacency_matrices$kinase_pathway)
  
  # Input layer: Substrate features
  substrate_input <- layer_input(shape = c(config$substrate_dim), name = "substrate_features")
  
  # Layer 1: Substrates -> Kinases (TRUE Graph Convolution)
  kinase_layer <- substrate_input %>%
    create_graph_conv_layer(
      adjacency_matrix = substrate_kinase_adj,
      units = config$kinase_dim,
      activation = 'relu',
      kernel_regularizer = regularizer_l2(config$l2_reg),
      name = "kinase_layer"
    ) %>%
    layer_dropout(rate = config$dropout_rate, name = "kinase_dropout") %>%
    layer_batch_normalization(name = "kinase_bn")
  
  # Layer 2: Kinases -> Pathways (TRUE Graph Convolution)
  pathway_layer <- kinase_layer %>%
    create_graph_conv_layer(
      adjacency_matrix = kinase_pathway_adj,
      units = config$pathway_dim,
      activation = 'relu',
      kernel_regularizer = regularizer_l2(config$l2_reg),
      name = "pathway_layer"
    ) %>%
    layer_dropout(rate = config$dropout_rate, name = "pathway_dropout") %>%
    layer_batch_normalization(name = "pathway_bn")
  
  # Layer 3: Pathways -> Response (Output Layer)
  response_output <- pathway_layer %>%
    layer_dense(
      units = config$response_dim,
      activation = 'sigmoid',  # For binary classification (0/1)
      name = "response_output"
    )
  
  # Create the complete model
  model <- keras_model(
    inputs = substrate_input,
    outputs = response_output,
    name = "substrate_kinase_pathway_gnn"
  )
  
  return(model)
}

# ===============================================================================
# MODEL COMPILATION
# ===============================================================================

# Function to compile the GNN model with appropriate optimizer and loss
compile_gnn_model <- function(model, config = default_config) {
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = config$learning_rate),
    loss = 'binary_crossentropy',  # For binary classification
    metrics = c('accuracy', 'precision', 'recall')
  )
  
  return(model)
}

# ===============================================================================
# CONVENIENCE FUNCTIONS
# ===============================================================================

# Function to create and compile a complete GNN model in one step
create_compiled_gnn <- function(config = default_config, adjacency_matrices) {
  
  # Build model (adjacency matrices are required)
  model <- build_gnn_model(
    config = config, 
    adjacency_matrices = adjacency_matrices
  )
  
  # Compile model
  compiled_model <- compile_gnn_model(model, config)
  
  # Print model summary
  cat("GNN Model Architecture Summary:\n")
  cat("==============================\n")
  summary(compiled_model)
  
  return(compiled_model)
}

# Function to print model configuration
print_model_config <- function(config = default_config) {
  cat("GNN Model Configuration:\n")
  cat("========================\n")
  cat(sprintf("Substrate dimensions: %d\n", config$substrate_dim))
  cat(sprintf("Kinase dimensions: %d\n", config$kinase_dim))
  cat(sprintf("Pathway dimensions: %d\n", config$pathway_dim))
  cat(sprintf("Response dimensions: %d\n", config$response_dim))
  cat(sprintf("Learning rate: %.4f\n", config$learning_rate))
  cat(sprintf("Dropout rate: %.2f\n", config$dropout_rate))
  cat(sprintf("L2 regularization: %.4f\n", config$l2_reg))
  cat("\nModel Architecture:\n")
  cat("Substrates [100] -> Kinases [64] -> Pathways [32] -> Response [1]\n")
}

# ===============================================================================
# SAMPLE ADJACENCY MATRIX GENERATION (for testing)
# ===============================================================================

# Function to generate sample adjacency matrices for testing
generate_sample_adjacency_matrices <- function(config) {
  
  # Sample Substrate-Kinase connections
  # Each substrate can connect to multiple kinases
  substrate_kinase_edges <- data.frame(
    substrate = sample(1:config$substrate_dim, 200, replace = TRUE),
    kinase = sample(1:config$kinase_dim, 200, replace = TRUE),
    weight = runif(200, 0.1, 1.0)  # Random weights between 0.1 and 1.0
  )
  
  # Sample Kinase-Pathway connections  
  # Each kinase can participate in multiple pathways
  kinase_pathway_edges <- data.frame(
    kinase = sample(1:config$kinase_dim, 150, replace = TRUE),
    pathway = sample(1:config$pathway_dim, 150, replace = TRUE),
    weight = runif(150, 0.1, 1.0)
  )
  
  # Create adjacency matrices
  substrate_kinase_adj <- create_weighted_adjacency_matrix(
    as.matrix(substrate_kinase_edges), 
    max(config$substrate_dim, config$kinase_dim)
  )
  
  kinase_pathway_adj <- create_weighted_adjacency_matrix(
    as.matrix(kinase_pathway_edges),
    max(config$kinase_dim, config$pathway_dim)
  )
  
  return(list(
    substrate_kinase = substrate_kinase_adj,
    kinase_pathway = kinase_pathway_adj
  ))
}

# ===============================================================================
# USAGE EXAMPLES (commented out)
# ===============================================================================

# Example 1: Create GNN with sample adjacency matrices
# adjacency_matrices <- generate_sample_adjacency_matrices(default_config)
# model <- create_compiled_gnn(adjacency_matrices = adjacency_matrices)

# Example 2: Create GNN with custom configuration
# custom_config <- list(
#   substrate_dim = 150,
#   kinase_dim = 80, 
#   pathway_dim = 40,
#   response_dim = 1,
#   learning_rate = 0.0005,
#   dropout_rate = 0.25,
#   l2_reg = 0.005
# )
# adjacency_matrices <- generate_sample_adjacency_matrices(custom_config)
# model <- create_compiled_gnn(config = custom_config, adjacency_matrices = adjacency_matrices)

# Example 3: Create GNN with your own adjacency matrices
# adjacency_matrices <- list(
#   substrate_kinase = your_substrate_kinase_matrix,
#   kinase_pathway = your_kinase_pathway_matrix
# )
# model <- create_compiled_gnn(
#   config = custom_config,
#   adjacency_matrices = adjacency_matrices
# )

cat("GNN Model functions loaded successfully!\n")
cat("Use create_compiled_gnn() to build and compile your model.\n")
cat("Use print_model_config() to see the current configuration.\n")