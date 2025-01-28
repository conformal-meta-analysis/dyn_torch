#' Modified entropy balancing implementation with autodiff via torch
#' @param X0 donor units matrix
#' @param X1 target moments
#' @param kappa regularization parameter
#' @return list with optimized weights
#' @import torch
#' @export
ebal_torch = function(X0, X1, kappa, maxit = 200) {
  n <- nrow(X0)
  d <- ncol(X0)

  inp = list(
    x0 = torch_tensor(X0),
    x1 = torch_tensor(X1),
    n = n
  )

  # Define loss function
  ebal_loss_torch = function(theta, eta) {
    Weights <- torch_softmax(theta, dim = 1) # Reparameterize Weights
    lambda <- torch_log1p(eta)         # Reparameterize lambda

    # Compute moment imbalance
    moment_imbalance <- torch_max(
      torch_abs(torch_matmul(Weights$t(), inp$x0) - torch_mean(inp$x0, dim = 1))
    )

    # Compute divergence term
    uniform_dist <- torch_full_like(Weights, 1 / n)
    divergence <- torch_sum(Weights * torch_log(Weights / uniform_dist))

    # Full loss
    loss <- moment_imbalance + lambda * divergence + kappa / (lambda * torch_sqrt(torch_tensor(n)))
    loss
  }

  # Gradient computation with autograd
  loss_grad = function(params) {
    theta = torch_tensor(params[1:n], requires_grad = TRUE)
    eta = torch_tensor(params[n + 1], requires_grad = TRUE)

    loss = ebal_loss_torch(theta, eta)
    grad = autograd_grad(loss, list(theta, eta))

    c(as.numeric(grad[[1]]), as.numeric(grad[[2]]))
  }

  # Optimization using BFGS
  init_params <- c(rep(0, n), 0.1) # Initialize theta and eta

  opt_result = optim(
    fn = function(x) as.numeric(ebal_loss_torch(torch_tensor(x[1:n]), torch_tensor(x[n + 1]))),
    gr = loss_grad, par = init_params,
    method = "BFGS",
    control = list(maxit = maxit)
  )

  # Recover W from optimized theta
  theta_opt <- torch_tensor(opt_result$par[1:n])
  W_opt <- torch_softmax(theta_opt, dim = 1)$numpy()

  list(Weights = W_opt)
}
