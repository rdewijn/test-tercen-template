# Visualize the substrate-kinase and kinase-pathway bipartite graphs using igraph
visualize_gnn_graphs <- function(substrate_kinase_edges, kinase_pathway_edges, file_prefix = "gnn_graph") {
  # Substrate-Kinase graph
  g_sk <- igraph::graph_from_data_frame(substrate_kinase_edges, directed = TRUE)
  png(paste0(file_prefix, "_substrate_kinase.png"), width = 800, height = 600)
  plot(g_sk, main = "Substrate-Kinase Bipartite Graph", vertex.size = 20, vertex.label.cex = 0.8)
  dev.off()

  # Kinase-Pathway graph
  g_kp <- igraph::graph_from_data_frame(kinase_pathway_edges, directed = TRUE)
  png(paste0(file_prefix, "_kinase_pathway.png"), width = 800, height = 600)
  plot(g_kp, main = "Kinase-Pathway Bipartite Graph", vertex.size = 20, vertex.label.cex = 0.8)
  dev.off()
  cat("Graph visualizations saved as PNG files.\n")
}

# Visualize the sample graphs

# GNN model visualization utility
plot_gnn_model <- function(model, filename = "gnn_model.png") {
  keras3::plot_model(model, to_file = filename, show_shapes = TRUE, show_layer_names = TRUE)
  cat(paste0("Model plot saved to ", filename, "\n"))
}

# ===============================================================================
# GNN Model Definition: Substrate -> Kinase -> Pathway -> Categorical Response
# Rectangular (bipartite) graph convolution between successive biological entity sets
# ===============================================================================

# Load required libraries for model building
library(keras3)

# tensorflow is loaded automatically by keras3 if needed
library(Matrix)
library(dplyr)  # For %>% operator
library(igraph)  # For network/graph operations

# ===============================================================================
# GNN MODEL CONFIGURATION
# ===============================================================================

# Default model hyperparameters for 3-layer substrate-kinase-pathway GNN
default_config <- list(
  # Biological layer sizes (node counts)
  n_substrates = 100,
  n_kinases = 64,
  n_pathways = 32,
  n_classes = 2,        # Categorical response (softmax)
  
  # Hidden feature dimensions after each graph conv + dense transform
  kinase_feature_dim = 64,
  pathway_feature_dim = 32,
  
  # Training parameters
  learning_rate = 0.001,
  
  # Regularization
  dropout_rate = 0.3,
  l2_reg = 0.01
)

# ===============================================================================
# TRUE GRAPH CONVOLUTION IMPLEMENTATION
# ===============================================================================

# Function to create rectangular bipartite graph convolution layer
# adjacency_matrix: shape (target_nodes, source_nodes)
# x: (batch, source_nodes)
create_graph_conv_layer <- function(adjacency_matrix,
                                    units,
                                    activation = 'relu',
                                    use_bias = TRUE,
                                    kernel_regularizer = NULL,
                                    name = NULL) {
  function(x) {
    n_targets <- dim(adjacency_matrix)[1]
    n_sources <- dim(adjacency_matrix)[2]
    x |> 
      keras3::layer_lambda(
        f = function(x) {
          adj_tensor <- keras3::k_constant(adjacency_matrix)
          keras3::k_dot(adj_tensor, x)
        },
        output_shape = list(n_targets, n_sources),
        name = paste0(name, "_graph_conv")
      ) |> 
      keras3::layer_dense(
        units = units,
        activation = activation,
        use_bias = use_bias,
        kernel_regularizer = kernel_regularizer,
        name = name
      )
  }
}

# ===============================================================================
# ADJACENCY MATRIX FUNCTIONS
# ===============================================================================

# Build rectangular (target x source) adjacency from bipartite edge list
# edge_list: data.frame with columns from_col, to_col
create_rectangular_adjacency <- function(edge_list,
                                         from_col,
                                         to_col,
                                         from_nodes = NULL,
                                         to_nodes = NULL) {
  if (!from_col %in% names(edge_list) || !to_col %in% names(edge_list)) {
    stop("Specified columns not found in edge_list")
  }
  if (is.null(from_nodes)) from_nodes <- sort(unique(edge_list[[from_col]]))
  if (is.null(to_nodes)) to_nodes <- sort(unique(edge_list[[to_col]]))
  from_index <- match(edge_list[[from_col]], from_nodes)
  to_index <- match(edge_list[[to_col]], to_nodes)
  x_vals <- rep(1, nrow(edge_list))
  adj <- Matrix::sparseMatrix(i = to_index, j = from_index,
                              x = x_vals,
                              dims = c(length(to_nodes), length(from_nodes)),
                              dimnames = list(to_nodes, from_nodes))
  adj
}

# Row-normalize rectangular adjacency (each target node aggregates equally from sources)
normalize_rectangular_adjacency <- function(adj) {
  rs <- Matrix::rowSums(adj)
  rs[rs == 0] <- 1
  Dinv <- Matrix::Diagonal(x = 1/rs)
  Dinv %*% adj
}

normalize_adjacency <- function(adj_matrix) {
  degree <- Matrix::rowSums(adj_matrix)
  degree[degree == 0] <- 1
  degree_sqrt_inv <- 1 / sqrt(degree)
  degree_matrix <- Matrix::Diagonal(x = degree_sqrt_inv)
  as.matrix(degree_matrix %*% adj_matrix %*% degree_matrix)
}


# ===============================================================================
# GNN MODEL ARCHITECTURE
# ===============================================================================

# Build 3-layer substrate -> kinase -> pathway GNN with categorical output
build_substrate_kinase_pathway_gnn <- function(config,
                                               substrate_kinase_adj,
                                               kinase_pathway_adj) {
  if (is.null(substrate_kinase_adj) || is.null(kinase_pathway_adj)) {
    stop("Both substrate->kinase and kinase->pathway adjacency matrices are required")
  }
  # Normalize rectangular adjacencies
  A_sk <- normalize_rectangular_adjacency(substrate_kinase_adj)
  A_kp <- normalize_rectangular_adjacency(kinase_pathway_adj)
  
  # Input: substrate feature vector (one feature per substrate node)
  substrate_input <- keras3::layer_input(shape = c(config$n_substrates), name = "substrate_features")
  
  # Substrate -> Kinase convolution
  kinase_rep <- substrate_input |>
    create_graph_conv_layer(
      adjacency_matrix = A_sk,
      units = config$kinase_feature_dim,
      activation = 'relu',
      name = "kinase_conv"
    )() |>
    keras3::layer_dropout(rate = config$dropout_rate, name = "kinase_dropout") |>
    keras3::layer_batch_normalization(name = "kinase_bn")
  
  # Kinase -> Pathway convolution
  pathway_rep <- kinase_rep |>
    create_graph_conv_layer(
      adjacency_matrix = A_kp,
      units = config$pathway_feature_dim,
      activation = 'relu',
      name = "pathway_conv"
    )() |>
    keras3::layer_dropout(rate = config$dropout_rate, name = "pathway_dropout") |>
    keras3::layer_batch_normalization(name = "pathway_bn")
  
  # Global mean pooling over pathway nodes (convert pathway node features to single vector)
  pooled <- pathway_rep |> keras3::layer_global_average_pooling_1d(name = "pathway_global_pool")
  
  # Categorical output (2 classes)
  response <- pooled |> keras3::layer_dense(units = config$n_classes,
                                            activation = 'softmax',
                                            name = "response")
  
  keras3::keras_model(inputs = substrate_input, outputs = response, name = "substrate_kinase_pathway_gnn")
}

# Backward compatible wrapper name
build_gnn_model <- function(config = default_config, adjacency_matrices) {
  build_substrate_kinase_pathway_gnn(config,
                                     substrate_kinase_adj = adjacency_matrices$substrate_kinase,
                                     kinase_pathway_adj   = adjacency_matrices$kinase_pathway)
}

# ===============================================================================
# MODEL COMPILATION
# ===============================================================================

# Function to compile the GNN model with appropriate optimizer and loss
compile_gnn_model <- function(model, config = default_config, use_sparse_labels = FALSE) {
  loss_fn <- if (use_sparse_labels) 'sparse_categorical_crossentropy' else 'categorical_crossentropy'
  keras3::compile(
    model,
    optimizer = keras3::optimizer_adam(learning_rate = config$learning_rate),
    loss = loss_fn,
    metrics = c('accuracy')
  )
  model
}

# ===============================================================================
# CONVENIENCE FUNCTIONS
# ===============================================================================

# Function to build GNN directly from substrate-kinase edge list
build_gnn_from_edge_lists <- function(substrate_kinase_edges,
                                      kinase_pathway_edges,
                                      config = default_config) {
  # Create rectangular adjacencies
  A_sk <- create_rectangular_adjacency(substrate_kinase_edges, from_col = 'substrate', to_col = 'kinase')
  A_kp <- create_rectangular_adjacency(kinase_pathway_edges, from_col = 'kinase', to_col = 'pathway')
  
  # Update config node counts from adjacency dimensions
  config$n_substrates <- ncol(A_sk)
  config$n_kinases    <- nrow(A_sk)
  config$n_pathways   <- nrow(A_kp)
  
  build_substrate_kinase_pathway_gnn(config, substrate_kinase_adj = A_sk, kinase_pathway_adj = A_kp)
}

# Function to create and compile a complete GNN model in one step
create_compiled_gnn <- function(config = default_config,
                                adjacency_matrices = NULL,
                                substrate_kinase_edges = NULL,
                                kinase_pathway_edges = NULL,
                                use_sparse_labels = FALSE) {
  if (!is.null(substrate_kinase_edges) && !is.null(kinase_pathway_edges)) {
    model <- build_gnn_from_edge_lists(substrate_kinase_edges, kinase_pathway_edges, config)
  } else if (!is.null(adjacency_matrices)) {
    model <- build_gnn_model(config = config, adjacency_matrices = adjacency_matrices)
  } else {
    stop("Provide either adjacency_matrices or both substrate_kinase_edges and kinase_pathway_edges")
  }
  compiled <- compile_gnn_model(model, config, use_sparse_labels = use_sparse_labels)
  cat("GNN Model Architecture Summary:\n")
  cat("==============================\n")
  summary(compiled)
  compiled
}

# Function to print model configuration
print_model_config <- function(config = default_config) {
  cat("Substrate-Kinase-Pathway GNN Configuration:\n")
  cat("============================================\n")
  cat(sprintf("# Substrates: %d\n", config$n_substrates))
  cat(sprintf("# Kinases: %d\n", config$n_kinases))
  cat(sprintf("# Pathways: %d\n", config$n_pathways))
  cat(sprintf("# Classes: %d\n", config$n_classes))
  cat(sprintf("Kinase feature dim: %d\n", config$kinase_feature_dim))
  cat(sprintf("Pathway feature dim: %d\n", config$pathway_feature_dim))
  cat(sprintf("Learning rate: %.4f\n", config$learning_rate))
  cat(sprintf("Dropout rate: %.2f\n", config$dropout_rate))
  cat(sprintf("L2 regularization: %.4f\n", config$l2_reg))
  cat("\nFlow:\nSubstrates ->(A_sk conv)-> Kinases ->(A_kp conv)-> Pathways -> GlobalPool -> Softmax(2)\n")
}

# ===============================================================================
# SAMPLE ADJACENCY MATRIX GENERATION (for testing)
# ===============================================================================

generate_sample_edge_lists <- function(config = default_config,
                                       n_sk_edges = 10,
                                       n_kp_edges = 3) {
  substrate_kinase_edges <- data.frame(
    substrate = sample(paste0('S', 1:config$n_substrates), n_sk_edges, replace = TRUE),
    kinase    = sample(paste0('K', 1:config$n_kinases), n_sk_edges, replace = TRUE)
  )
  kinase_pathway_edges <- data.frame(
    kinase  = sample(paste0('K', 1:config$n_kinases), n_kp_edges, replace = TRUE),
    pathway = sample(paste0('P', 1:config$n_pathways), n_kp_edges, replace = TRUE)
  )
  list(substrate_kinase = substrate_kinase_edges,
       kinase_pathway   = kinase_pathway_edges)
}

# ===============================================================================
# USAGE EXAMPLES (commented out)
# ===============================================================================

# Example: Build and compile from synthetic edge lists
# edges <- generate_sample_edge_lists(default_config)
# model <- create_compiled_gnn(substrate_kinase_edges = edges$substrate_kinase,
#                              kinase_pathway_edges   = edges$kinase_pathway)
# print_model_config(default_config)

# Example: Using your real data frames substrate_kinase_df and kinase_pathway_df
# model <- create_compiled_gnn(substrate_kinase_edges = substrate_kinase_df,
#                              kinase_pathway_edges   = kinase_pathway_df,
#                              use_sparse_labels = TRUE)

edges = generate_sample_edge_lists()

A1 = edges[[1]] %>%  
  create_rectangular_adjacency(from_col = 'substrate', to_col = 'kinase')

agnn = build_gnn_from_edge_lists(edges[[1]], edges[[2]])

# Visualize the sample graphs (after 'edges' is defined)
visualize_gnn_graphs(edges$substrate_kinase, edges$kinase_pathway)
