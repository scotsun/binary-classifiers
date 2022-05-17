local_log_likelihood <- function(a, x_train, y_train, x, h) {
  K <- function(x) {dnorm(x, mean = 0, sd = 1)}
  
  a0 <- a[1]
  a1 <- a[2]
  l <- sum(
    K((x - x_train)/h) * (y_train*(a0 + a1*(x_train - x)) - log(1 + exp(a0 + a1*(x_train - x))))
  )
  return(l)
}

get_optimal_a0 <- function(init_a, x_train, y_train, x, h) {
  
  a_hat <- optim(par = init_a, local_log_likelihood, 
                 x_train = x_train, y_train = y_train, x = x, h = h,
                 control = list(fnscale = -1))
  return(a_hat$par[1])
}

local_linear_logistic_regressor <- function(x_train, y_train, x, h) {
  r_hat <- c()
  for (elem in x) {
    a0_hat <- get_optimal_a0(init_a = c(0, 0), x_train, y_train, elem, h)
    r_hat_x <- exp(a0_hat) / (1 + exp(a0_hat))
    r_hat <- append(r_hat, r_hat_x)
  }
  return(r_hat)
}

loocv_loglike <- function(x_train, y_train, h) {
  a0_hat <- c() # list of leave-one-out eta_hat
  for (i in 1:length(x_train)) {
    a0_hat_i <- get_optimal_a0(init_a = c(0, 0), x_train[-i], y_train[-i], x_train[i], h)
    a0_hat <- append(a0_hat, a0_hat_i)
  }
  score <- sum(y_train*a0_hat - log(1 + exp(a0_hat)))
  return(score)
}

loocv_loglike.best_bandwidth <- function(x_train, y_train, from, to, step_size) {
  bandwidths <- seq(from, to, by = step_size)
  score <- c()
  for (h in bandwidths) {
    score <- append(score, loocv_loglike(x_train, y_train, h))
  }
  return(list(score = score,
              bandwidths = bandwidths,
              optim_bandwidth = bandwidths[which.max(score)])) 
  # since CV is log-likelihood, we need to pick max CV
}
