Extended Conditional Density Estimation Network Creation and Evaluation"

Introduction
The developed R code implements an Extended Conditional Density Estimation Network Creation and Evaluation model designed for forecasting distribution parameters, with a generalized network architecture capable of accommodating any number of hidden layers, hidden neurons, and activation functions.

ecadence.initialize
This function initializes weights for the extended conditional density estimation neural network. 

ecadence.initialize <- function(x, hidden.neurons, init.range, distribution) {
  
  # Determine the number of parameters in the output layer
  n.parms <- length(distribution$parameters)  # For log-normal: mean and variance
  
  # Initialize an empty list to store weight matrices
  weights <- list()
  
  # The number of hidden layers is defined by the length of hidden.neurons
  n.hidden.layers <- length(hidden.neurons)
  
  # Initialize weights for the input layer -> first hidden layer
  weights[[1]] <- matrix(runif((ncol(x) + 1) * hidden.neurons[1], 
                               init.range[1], init.range[2]), 
                         ncol(x) + 1, hidden.neurons[1])
  
  # Initialize weights for hidden layers
  if (n.hidden.layers > 1) {
    for (i in 2:n.hidden.layers) {
      # Connecting previous hidden layer to the next hidden layer
      weights[[i]] <- matrix(runif((hidden.neurons[i-1] + 1) * hidden.neurons[i], 
                                   init.range[1], init.range[2]), 
                             hidden.neurons[i-1] + 1, hidden.neurons[i])
    }
  }
  
  # Initialize weights for the output layer (last hidden layer -> output)
  weights[[n.hidden.layers + 1]] <- matrix(runif((hidden.neurons[n.hidden.layers] + 1) * n.parms, 
                                                 init.range[1], init.range[2]), 
                                           hidden.neurons[n.hidden.layers] + 1, n.parms)
  
  # Return all weights as a single vector
  unlist(weights)
}

Explanation

1. Function Arguments
x: Matrix of input data.
hidden.neurons: Vector specifying neurons in each hidden layer.
init.range: Range for initializing weights (e.g., c(-0.1, 0.1)).
distribution: Contains information about the output distribution.

2. Weight Initialization Strategy
Input layer → first hidden layer: (features + 1) × neurons
Hidden layer → next hidden layer: (previous neurons + 1) × current neurons
Last hidden layer → output layer: (last neurons + 1) × output parameters

3. Output
The function returns a concatenated vector of initialized weights.

ecadence.reshape
This function reshapes a flattened weight vector into structured weight matrices for the extended conditional density estimation neural network. This function ensures proper connectivity between layers.

ecadence.reshape <- function(x, weights, hidden.neurons, distribution) {
  w <- list()  # Initialize a list to store reshaped weights
  start_idx <- 1  # Start index for slicing the weights vector
  
  # Reshape weights for the hidden layers
  for (i in seq_along(hidden.neurons)) {
    n_input <- ifelse(i == 1, ncol(x) + 1, hidden.neurons[i - 1] + 1)  # Input size depends on layer
    n_output <- hidden.neurons[i]  # Output size is the number of neurons in the current hidden layer
    end_idx <- start_idx + (n_input * n_output) - 1  # Calculate the end index
    w[[i]] <- matrix(weights[start_idx:end_idx], n_input, n_output)  # Reshape the weights into a matrix
    start_idx <- end_idx + 1  # Update the start index for the next layer
  }
  
  # Reshape weights for the output layer
  if (length(hidden.neurons) > 0) {  # Check if there are hidden layers
    n_input <- hidden.neurons[length(hidden.neurons)] + 1  # Input size for output layer
  } else {  # If no hidden layers, input is from input layer directly to output
    n_input <- ncol(x) + 1
  }
  
  n_output <- length(distribution$parameters)  # Number of output parameters
  end_idx <- start_idx + (n_input * n_output) - 1  # Calculate the end index for output layer weights
  w[[length(hidden.neurons) + 1]] <- matrix(weights[start_idx:end_idx], n_input, n_output)  # Reshape to output layer
  
  # Return the list of reshaped weight matrices
  return(w)
}

Explanation

1. Function Arguments
- `x`: Matrix of input data.
- `weights`: A single vector containing all neural network weights.
- `hidden.neurons`: Vector specifying the number of neurons in each hidden layer.
- `distribution`: Contains details about the output distribution.

2. Reshaping Process
- Hidden layers:
- If it’s the first hidden layer, it connects to the input layer (`ncol(x) + 1` for bias).
- If it’s a subsequent hidden layer, it connects to the previous hidden layer (`hidden.neurons[i-1] + 1` for bias).
- Extracts weights from `weights` vector and reshapes them into matrices.

- Output layer:
- Takes input from the last hidden layer or directly from the input layer (if no hidden layers exist).
- Maps to `n_output` neurons, where `n_output = length(distribution$parameters)`.

3. Output
- A list of reshaped weight matrices, where each matrix represents the weights connecting layers.

ecadence.evaluate
This function is responsible for evaluating the neural network model by performing a forward pass through its layers. It takes the input data, applies weight transformations, passes activations through hidden layers using specified activation functions, and finally transforms the output using distribution-based functions.

ecadence.evaluate <- function(x, weights, hidden.fcn, distribution) {
  # Handling fixed output parameters (if any)
  if (!is.null(distribution$parameters.fixed)) {
    colnames(weights[[length(weights)]]) <- distribution$parameters
    weights[[length(weights)]][1:(nrow(weights[[length(weights)]]) - 1), distribution$parameters.fixed] <- 0
  }
  
  # Forward pass: Input layer to the first hidden layer
  x <- cbind(x, 1)  # Add bias term to input layer
  current_activation <- x  # Initialize current activation
  
  # Loop through all hidden layers
  for (i in seq_along(weights[-length(weights)])) {
    h <- current_activation %*% weights[[i]]  # Calculate activations
    y <- hidden.fcn[[i]](h)  # Apply activation function
    current_activation <- cbind(y, 1)  # Add bias term for the next layer
  }
  
  # Forward pass to the output layer
  output <- current_activation %*% weights[[length(weights)]]
  
  # Apply output functions to the raw output values
  output <- mapply(do.call, distribution$output.fcns, lapply(data.frame(output), list))
  
  # Ensure the output is in matrix form
  if (!is.matrix(output)) output <- matrix(output, nrow = 1)
  
  # Assign column names to match the output parameters
  colnames(output) <- distribution$parameters
  
  # Return the final output matrix
  return(output)
}


Explanation

1. Handling Fixed Output Parameters
- The function first checks if there are any fixed output parameters in the `distribution$parameters.fixed` attribute.
- If such parameters exist, the corresponding weight values in the last layer are set to zero to ensure that these parameters remain unchanged.

2. Forward Pass from Input Layer to First Hidden Layer
- The input matrix `x` is augmented by adding a bias term (column of ones) to facilitate weight multiplication.
- The variable `current_activation` is initialized with this modified input data.

3. Propagation Through Hidden Layers
- The function iterates over all hidden layers, performing the following steps:
  - Compute the weighted sum of inputs (`h = current_activation %*% weights[[i]]`).
- Apply the corresponding activation function (`y = hidden.fcn[[i]](h)`).
- Add a bias term and update `current_activation` for the next layer.

4. Forward Pass to the Output Layer
- After propagating through all hidden layers, the function computes the final weighted sum (`output = current_activation %*% weights[[length(weights)]]`).
- The `mapply` function is used to apply specific transformations to the output based on the distribution's functions (`distribution$output.fcns`).

5. Formatting the Output
- If the result is not already in matrix form, it is converted to ensure consistency.
- Column names are assigned according to the parameters defined in `distribution$parameters`.
- Finally, the processed output matrix is returned.

ecadence.cost
This function computes the cost (negative log-likelihood) for the extended conditional density estimation neural network based on the distribution. It incorporates regularization to prevent overfitting.

ecadence.cost <- function(weights, x, y, hidden.neurons, hidden.fcn, distribution, sd.norm, valid) {
  
  # Initialize a vector for valid weights
  weights.valid <- valid * 0
  weights.valid[valid] <- weights
  
  # Reshape the weights into matrices
  w <- ecadence.reshape(x, weights, hidden.neurons, distribution)
  
  # Evaluate the neural network
  cdn <- ecadence.evaluate(x, w, hidden.fcn, distribution)
  
  # Prepare arguments for the density function
  args <- as.list(data.frame(cbind(y, cdn)))
  names(args) <- NULL
  
  # Calculate the negative log-likelihood
  L <- do.call(distribution$density.fcn, args)
  NLL <- -sum(log(L))
  
  # Initialize the penalty term
  penalty <- 0
  
  # Apply regularization if sd.norm is finite
  if (sd.norm != Inf) {
    penalty_list <- list()  # Initialize a list to store the penalty for each layer
    
    for (i in seq_along(w)) {  # Loop through the weight matrices (W1, W2, W3, ...)
      layer_weights <- w[[i]]  # Get the current weight matrix (e.g., W1, W2, W3, ...)
      
      if (i == length(w)) {  # For the last layer, apply regularization to all elements
        penalty_list[[i]] <- dnorm(as.vector(layer_weights), sd = sd.norm)
      } else {
        # Check if the activation function is the identity function for this layer
        if (identical(hidden.fcn[[i]], identity)) {
          # Regularization for identity activation function (exclude the last row for bias)
          penalty_list[[i]] <- dnorm(as.vector(layer_weights[1:(nrow(layer_weights) - 1), ]), sd = sd.norm)
        } else {
          # Regularization for non-identity activation function (apply to all elements)
          penalty_list[[i]] <- dnorm(as.vector(layer_weights), sd = sd.norm)
        }
      }
    }
    
    # Combine all penalties and take the negative log mean
    penalty <- -mean(log(unlist(penalty_list)))
  }
  
  # Handle NaN values in NLL
  if (is.nan(NLL)) NLL <- .Machine$double.xmax
  
  # Add penalty to the NLL
  NLL <- NLL + penalty
  
  # Store penalty as an attribute
  attr(NLL, "penalty") <- penalty
  
  return(NLL)
}

Explanation
1. Reshaping Weights:
   - The function reshapes the input weight vector into matrices corresponding to the neural network's structure using `ecadence.reshape`.

2. Evaluating the Neural Network:
  - The function calls `ecadence.evaluate` to obtain predictions from the model.

3. Computing Negative Log-Likelihood (NLL):
  - It calculates the likelihood of the observed data using the density function of the given distribution.
- The log-likelihood is summed, and its negative is taken as the cost function value.

4. Regularization Penalty:
  - If `sd.norm` is finite, a penalty is applied to regularize weights and prevent overfitting.
- Weights are adjusted based on the activation function used.

5. Final Cost Calculation:
  - The penalty term is added to NLL.
- If `NLL` becomes `NaN`, it is replaced with the maximum double-precision value to ensure numerical stability.

ecadence.fit

This function is designed to fit the extended conditional density estimation neural network based on a distribution. The function optimizes the weights of the neural network using various optimization methods, including Nelder-Mead, particle swarm optimization (PSO), and resilient backpropagation (Rprop). The goal is to minimize the negative log-likelihood (NLL) while optionally applying regularization.

ecadence.fit <- function(x, y, iter.max = 500, hidden.neurons = hidden.neurons, hidden.fcn = hidden.fcn, 
                              distribution = NULL, sd.norm = 0.1, init.range = c(-0.5, 0.5),
                              method = c("optim", "psoptim", "Rprop"), n.trials = 1,
                              trace = 1, maxit.Nelder = 2000, trace.Nelder = 0,
                              swarm.size = NULL, vectorize = TRUE,
                              delta.0 = 0.1, delta.min = 1e-06, delta.max = 50, epsilon = 1e-08,
                              range.mult = 2, step.tol = 1e-08, f.target = -Inf,
                              f.cost = ecadence.cost, max.exceptions = 500) {
  
  set.seed(123)  # Ensures reproducibility
  
  if (!is.matrix(x)) stop("x must be a matrix")
  if (!is.matrix(y)) stop("y must be a matrix")
  if (any(c(hidden.neurons, n.trials) <= 0)) stop("Invalid hidden.neurons or n.trials")
  
  if (is.null(distribution)) {
    distribution <- list(density.fcn = dnorm, parameters = c("mean", "sd"), output.fcns = c(identity, exp))
    warning("Unspecified distribution; defaulting to dnorm")
  }
  
  x <- scale(x)  # Normalize input data
  attr(x, "scaled:scale")[attr(x, "scaled:scale") == 0] <- 1
  x[is.nan(x)] <- 0
  method <- match.arg(method)
  
  fit <- list()
  NLL <- Inf
  
  for (i in seq(n.trials)) {
    
    exception <- TRUE
    n.exceptions <- 0
    while (exception) {
      
      weights <- ecadence.initialize(x, hidden.neurons, init.range, distribution)
      gradient <- fprime(weights, f.cost, epsilon)
      valid.cur <- gradient^2 > epsilon
      weights <- weights[valid.cur]
      
      if (method == "optim") {
        output.cdn.cur <- try(optim(weights, f.cost, method = "BFGS", control = list(maxit = iter.max, trace = trace),
                                    x = x, y = y, hidden.neurons = hidden.neurons, hidden.fcn = hidden.fcn,
                                    distribution = distribution, sd.norm = sd.norm, valid = valid.cur),
                              silent = trace == 0)
      } else if (method == "psoptim") {
        swarm.size <- ifelse(is.null(swarm.size), floor(10 + 2 * sqrt(length(weights))), swarm.size)
        output.cdn.cur <- try(pso::psoptim(weights, f.cost, lower = weights - range.mult * diff(range(weights)),
                                           upper = weights + range.mult * diff(range(weights)),
                                           control = list(maxit = iter.max, trace = trace),
                                           x = x, y = y, hidden.neurons = hidden.neurons, hidden.fcn = hidden.fcn,
                                           distribution = distribution, sd.norm = sd.norm, valid = valid.cur),
                              silent = trace == 0)
      }
      
      if (!inherits(output.cdn.cur, "try-error")) {
        exception <- FALSE
      } else {
        n.exceptions <- n.exceptions + 1
        if (n.exceptions > max.exceptions) stop("Max number of exceptions reached.")
      }
    }
    
    NLL.cur <- output.cdn.cur$value
    if (NLL.cur < NLL) {
      NLL <- NLL.cur
      output.cdn <- output.cdn.cur
      valid <- valid.cur
    }
  }
  
  weights <- output.cdn$par
  NLL <- ecadence.cost(weights, x, y, hidden.neurons, hidden.fcn, distribution, sd.norm, valid)
  penalty <- attr(NLL, "penalty")
  
  k <- length(weights)
  n <- nrow(y)
  BIC <- 2 * NLL + k * log(n)
  AIC <- 2 * NLL + 2 * k
  AICc <- AIC + (2 * k * (k + 1)) / (n - k - 1)
  
  w <- ecadence.reshape(x, weights, hidden.neurons, distribution)
  attr(w, "hidden.neurons") <- hidden.neurons
  attr(w, "BIC") <- BIC
  attr(w, "AICc") <- AICc
  attr(w, "AIC") <- AIC
  
  return(list(fit = w))

Explanation

1. Input Handling: Ensures `x` and `y` are matrices, scales `x`, and sets default values if `distribution` is not provided.
2. Gradient Approximation (`fprime`): Computes the numerical gradient for weight initialization.
3. Optimization Process:
  - `optim` (BFGS) for gradient-based optimization.
- `psoptim` (particle swarm optimization) for global search.
- `Rprop` (resilient backpropagation) for adaptive learning.
4. Model Selection: Runs multiple trials (`n.trials`) and selects the best model based on the lowest negative log-likelihood.
5. Penalty Regularization: Uses a Gaussian prior to prevent overfitting by penalizing large weight values.
6. Model Evaluation Metrics:
  - AIC (Akaike Information Criterion): Measures model quality with a trade-off between goodness-of-fit and complexity.
- BIC (Bayesian Information Criterion): Similar to AIC but penalizes complexity more strongly.
- AICc (Corrected AIC): Adjusts AIC for small sample sizes.

The function ultimately returns the best-fitted network model with optimized weights and evaluation metrics.

ecadence.predict

This function is used for making predictions using a trained conditional density estimation neural network. It takes a matrix of input values (`x`), a fitted model (`fit`), and model parameters such as `hidden.neurons` and `hidden.fcn`. The function processes the input data, applies the trained model parameters, and returns predictions based on the specified probability distribution.

ecadence.predict <- function(x, fit, hidden.neurons, hidden.fcn) {
  if (!is.matrix(x)) stop("\"x\" must be a matrix")
  if ("W1" %in% names(fit)) fit <- list(fit = fit) # Standardize the input format
  
  pred <- list() # Initialize an empty list to store predictions
  
  for (i in seq_along(fit)) {
    nh <- names(fit)[i] # Get the name of the current model configuration
    hidden.fcn1 <- attr(fit[[nh]], "hidden.fcn1") # Extract hidden function
    distribution <- attr(fit[[nh]], "distribution") # Extract distribution used
    x.center <- attr(fit[[nh]], "x.center") # Extract centering info
    x.scale <- attr(fit[[nh]], "x.scale") # Extract scaling info
    
    # Center and scale the input data based on model attributes
    x.pred <- sweep(x, 2, x.center, "-")
    x.pred <- sweep(x.pred, 2, x.scale, "/")
    
    # Extract weights from the model
    weight_list <- fit[[nh]]
    
    # Reshape the weights for the given number of hidden neurons
    reshaped_weights <- ecadence.reshape(
      x = x.pred,
      weights = unlist(weight_list), # Combine all weights into a vector
      hidden.neurons = hidden.neurons, # Dynamic hidden neurons from weights
      distribution = distribution
    )
    
    # Perform evaluation using the reshaped weights, hidden activation functions, and distribution
    pred[[nh]] <- ecadence.evaluate(
      x = x.pred,
      weights = reshaped_weights,
      hidden.fcn = hidden.fcn, # Dynamic activation function
      distribution = distribution
    )
  }
  
  if (length(pred) == 1) pred <- pred[[1]] # Simplify if there's only one prediction set
  
  return(pred)
}

Explanation

Input Parameters
- `x`: A matrix of input features for prediction.
- `fit`: The trained model object containing weights and attributes.
- `hidden.neurons`: Number of neurons in the hidden layers.
- `hidden.fcn`: Activation function used in the hidden layers.

Step-by-Step Execution
1. Validation of Input
- The function first checks whether `x` is a matrix. If not, it stops execution with an error message.
- It also ensures `fit` is structured as a list.

2. Loop Through Models
- If multiple models exist in `fit`, the function iterates over each model.
- Extracts important attributes such as:
- `hidden.fcn1`: Activation function for hidden layers.
- `distribution`: The assumed probability distribution.
- `x.center` and `x.scale`: Used for scaling input data.

3. Preprocessing Input Data
- The input matrix `x` is centered and scaled using the stored mean (`x.center`) and standard deviation (`x.scale`).

4. Weight Reshaping
- Extracts the weight parameters from the model and reshapes them using `ecadence.reshape`.
- This ensures the correct structure is maintained for different hidden layer configurations.

5. Prediction Calculation
- The function calls `ecadence.evaluate` to compute predictions based on:
- The processed input `x.pred`
- The reshaped model weights
- The chosen activation function (`hidden.fcn`)
- The selected probability distribution

6. Return Predictions
- If multiple models exist, predictions are stored in a list.
- If only one model exists, the function simplifies the output to return a single prediction object.


  Defining the Density Function and Activation Functions
  
  For instance, in this framework, we define the density function for the log-normal distribution as follows:
  
lnorm.distribution <- list(
  density.fcn = dlnorm,  # Log-normal density function
  parameters = c("meanlog", "sdlog"),  # Parameters of the log-normal distribution
  output.fcns = c(identity, exp)  # Output transformations for mean and standard deviation
)
lnorm.distribution

This setup allows the model to estimate log-normal parameters based on the given input data.

Activation Functions

Similarly, activation functions for different hidden layer configurations can be defined as follows:
  
Three Hidden Layers:
hidden.fcn3 <- list(
  function(h) { tanh(h) },  # Activation function for the first hidden layer
  function(h) { tanh(h) },  # Activation function for the second hidden layer
  function(h) { tanh(h) }   # Activation function for the third hidden layer
)

Two Hidden Layers:
hidden.fcn2 <- list(
  function(h) { tanh(h) },  # Activation function for the first hidden layer
  function(h) { tanh(h) }   # Activation function for the second hidden layer
)

These activation functions introduce non-linearity into the network, allowing it to capture complex relationships in the data. The hyperbolic tangent (`tanh`) function is chosen due to its properties of outputting values between -1 and 1, which helps stabilize the learning process.
