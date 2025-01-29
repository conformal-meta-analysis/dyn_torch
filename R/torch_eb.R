#' Modified entropy balancing implementation with autodiff via torch
#' @param X0 donor units matrix
#' @param X1 target moments
#' @param kappa regularization parameter
#' @return list with optimized weights
#' @import torch
#' @export
ebal_torch = function (X0, X1, kappa, maxit = 200) 
{
    n <- nrow(X0)
    d <- ncol(X0)
    inp = list(x0 = torch_tensor(X0), x1 = torch_tensor(X1), 
        n = n)
    ebal_loss_torch = function(theta, eta) {
        Weights <- nnf_softmax(theta, dim = 1)
        lambda <- torch_log1p(eta)
        moment_imbalance <- torch_max(torch_abs(torch_matmul(Weights$t(), 
            inp$x0) - inp$x1))
        uniform_dist <- torch_full_like(Weights, 1/n)
        divergence <- torch_sum(Weights * torch_log(Weights/uniform_dist))
        loss <- moment_imbalance + lambda * divergence + kappa/(lambda * 
            torch_sqrt(torch_tensor(n)))
        return(loss)
    }
    loss_grad = function(params) {
        theta = torch_tensor(params[1:n], requires_grad = TRUE)
        eta = torch_tensor(params[n + 1], requires_grad = TRUE)
        loss = ebal_loss_torch(theta, eta)
        grad = autograd_grad(loss, list(theta, eta))
        c(as.numeric(grad[[1]]), as.numeric(grad[[2]]))
    }
    init_params <- c(rep(1/n, n), 0.1)
    objective_function <- function(x) as.numeric(ebal_loss_torch(torch_tensor(x[1:n]), 
        torch_tensor(x[n + 1])))
    opt_result = optim(fn = objective_function, gr = loss_grad, 
        par = init_params, method = "BFGS", control = list(maxit = maxit))
    theta_opt <- torch_tensor(opt_result$par[1:n])
    W_opt <- nnf_softmax(theta_opt, dim = 1)
    list(Weights = W_opt)
}
