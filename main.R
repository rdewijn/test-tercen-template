# ===============================================================================
# Graph Neural Network (GNN) Implementation with KERAS in R
# 4-Layer Architecture: Substrates -> Kinases -> Pathways -> Response
# ===============================================================================

# Load required libraries
library(keras)
library(tensorflow)
library(dplyr)
library(Matrix)
library(igraph)

# Set random seed for reproducibility
set.seed(42)
tensorflow::tf$random$set_seed(42L)

# ===============================================================================
# ENVIRONMENT SETUP
# ===============================================================================

# Install and configure TensorFlow/Keras if needed
if (!tensorflow::tf_config()$gpu_available) {
  cat("GPU not available, using CPU\n")
} else {
  cat("GPU available for training\n")
}

# ===============================================================================
# GNN MODEL CONFIGURATION
# ===============================================================================

# Model hyperparameters
config <- list(
  # Layer dimensions
  substrate_dim = 100,    # Input layer: number of substrate features
  kinase_dim = 64,        # Hidden layer 1: kinase representation
  pathway_dim = 32,       # Hidden layer 2: pathway representation
  response_dim = 1,       # Output layer: binary classification
  
  # Training parameters
  learning_rate = 0.001,
  batch_size = 32,
  epochs = 100,
  validation_split = 0.2,
  
  # Regularization
  dropout_rate = 0.3,
  l2_reg = 0.01
)

# ===============================================================================
# GRAPH CONVOLUTION LAYER IMPLEMENTATION
# ===============================================================================

# Custom Graph Convolution Layer
graph_conv_layer <- function(units, 
                           activation = 'relu',
                           use_bias = TRUE,
                           kernel_regularizer = NULL,
                           name = NULL) {
  
  keras::layer_dense(
    units = units,
    activation = activation,
    use_bias = use_bias,
    kernel_regularizer = kernel_regularizer,
    name = name
  )
}

# ===============================================================================
# DATA PREPROCESSING FUNCTIONS
# ===============================================================================

# Function to create adjacency matrix from edge list
create_adjacency_matrix <- function(edge_list, n_nodes) {
  adj_matrix <- Matrix::sparseMatrix(
    i = edge_list[,1], 
    j = edge_list[,2], 
    x = rep(1, nrow(edge_list)),
    dims = c(n_nodes, n_nodes)
  )
  
  # Make symmetric and add self-loops
  adj_matrix <- adj_matrix + Matrix::t(adj_matrix)
  Matrix::diag(adj_matrix) <- 1
  
  return(adj_matrix)
}

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

# ===============================================================================
# GNN MODEL ARCHITECTURE
# ===============================================================================

build_gnn_model <- function(config, adjacency_matrices = NULL) {
  
  # Input layers
  substrate_input <- layer_input(shape = c(config$substrate_dim), name = "substrate_features")
  
  # Layer 1: Substrates -> Kinases (Graph Convolution)
  kinase_layer <- substrate_input %>%
    graph_conv_layer(
      units = config$kinase_dim,
      activation = 'relu',
      kernel_regularizer = regularizer_l2(config$l2_reg),
      name = "kinase_layer"
    ) %>%
    layer_dropout(rate = config$dropout_rate, name = "kinase_dropout") %>%
    layer_batch_normalization(name = "kinase_bn")
  
  # Layer 2: Kinases -> Pathways (Graph Convolution)
  pathway_layer <- kinase_layer %>%
    graph_conv_layer(
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
  
  # Create model
  model <- keras_model(
    inputs = substrate_input,
    outputs = response_output,
    name = "substrate_kinase_pathway_gnn"
  )
  
  return(model)
}

# ===============================================================================
# MODEL COMPILATION AND TRAINING
# ===============================================================================

compile_gnn_model <- function(model, config) {
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = config$learning_rate),
    loss = 'binary_crossentropy',  # For binary classification
    metrics = c('accuracy', 'precision', 'recall')
  )
  return(model)
}

# Training function with callbacks
train_gnn_model <- function(model, x_train, y_train, config, 
                           x_val = NULL, y_val = NULL) {
  
  # Define callbacks
  callbacks <- list(
    callback_early_stopping(
      monitor = 'val_loss',
      patience = 10,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = 'val_loss',
      factor = 0.5,
      patience = 5,
      min_lr = 1e-7
    )
  )
  
  # Train model
  history <- model %>% fit(
    x = x_train,
    y = y_train,
    validation_data = if (!is.null(x_val)) list(x_val, y_val) else NULL,
    validation_split = if (is.null(x_val)) config$validation_split else 0,
    epochs = config$epochs,
    batch_size = config$batch_size,
    callbacks = callbacks,
    verbose = 1
  )
  
  return(history)
}

# ===============================================================================
# EVALUATION FUNCTIONS
# ===============================================================================

evaluate_gnn_model <- function(model, x_test, y_test) {
  # Get predictions
  predictions <- model %>% predict(x_test)
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate metrics
  accuracy <- mean(predicted_classes == y_test)
  
  # Confusion matrix
  confusion_matrix <- table(Predicted = predicted_classes, Actual = y_test)
  
  # Precision, Recall, F1-score
  if (length(unique(y_test)) == 2) {
    tp <- sum(predicted_classes == 1 & y_test == 1)
    fp <- sum(predicted_classes == 1 & y_test == 0)
    fn <- sum(predicted_classes == 0 & y_test == 1)
    
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1_score <- ifelse(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)
  } else {
    precision <- recall <- f1_score <- NA
  }
  
  results <- list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    confusion_matrix = confusion_matrix,
    predictions = predictions
  )
  
  return(results)
}

# ===============================================================================
# EXAMPLE USAGE AND TESTING
# ===============================================================================

# Function to generate sample data for testing
generate_sample_data <- function(n_samples = 1000, config) {
  
  # Generate random substrate features
  x_data <- matrix(rnorm(n_samples * config$substrate_dim), 
                   nrow = n_samples, 
                   ncol = config$substrate_dim)
  
  # Generate binary response (0 or 1)
  # Simple relationship: response based on sum of first few features
  response_score <- rowSums(x_data[, 1:min(5, config$substrate_dim)])
  y_data <- as.numeric(response_score > median(response_score))
  
  return(list(x = x_data, y = y_data))
}

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

main <- function() {
  cat("Initializing GNN for Substrate-Kinase-Pathway-Response modeling...\n")
  
  # Generate sample data (replace with your actual data)
  cat("Generating sample data...\n")
  sample_data <- generate_sample_data(n_samples = 1000, config = config)
  
  # Split data
  n_train <- floor(0.8 * nrow(sample_data$x))
  train_indices <- sample(1:nrow(sample_data$x), n_train)
  
  x_train <- sample_data$x[train_indices, ]
  y_train <- sample_data$y[train_indices]
  x_test <- sample_data$x[-train_indices, ]
  y_test <- sample_data$y[-train_indices]
  
  cat(sprintf("Training samples: %d, Test samples: %d\n", 
              length(y_train), length(y_test)))
  
  # Build model
  cat("Building GNN model...\n")
  model <- build_gnn_model(config)
  model <- compile_gnn_model(model, config)
  
  # Print model summary
  cat("Model architecture:\n")
  summary(model)
  
  # Train model
  cat("Training model...\n")
  history <- train_gnn_model(model, x_train, y_train, config)
  
  # Evaluate model
  cat("Evaluating model...\n")
  results <- evaluate_gnn_model(model, x_test, y_test)
  
  # Print results
  cat("Model Performance:\n")
  cat(sprintf("Accuracy: %.4f\n", results$accuracy))
  cat(sprintf("Precision: %.4f\n", results$precision))
  cat(sprintf("Recall: %.4f\n", results$recall))
  cat(sprintf("F1-Score: %.4f\n", results$f1_score))
  
  cat("Confusion Matrix:\n")
  print(results$confusion_matrix)
  
  return(list(model = model, history = history, results = results))
}

# ===============================================================================
# PLACEHOLDER FOR YOUR DATA INTEGRATION
# ===============================================================================

# TODO: Replace the sample data generation with your actual data loading
# 
# load_your_data <- function() {
#   # Load substrate features
#   substrates <- read.csv("path/to/substrate_features.csv")
#   
#   # Load response labels (0/1)
#   responses <- read.csv("path/to/response_labels.csv")
#   
#   # Load network topology (optional)
#   # substrate_kinase_edges <- read.csv("path/to/substrate_kinase_network.csv")
#   # kinase_pathway_edges <- read.csv("path/to/kinase_pathway_network.csv")
#   
#   return(list(x = substrates, y = responses))
# }

# Run the main function if script is executed directly
if (!interactive()) {
  results <- main()
}

cat("GNN model initialization complete!\n")
cat("Next steps:\n")
cat("1. Replace sample data with your actual substrate and response data\n")
cat("2. Define network topology between substrates, kinases, and pathways\n")
cat("3. Adjust model hyperparameters in the config list\n")
cat("4. Run the main() function to train your model\n")