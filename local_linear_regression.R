local_linear_effective_kernal <- function(x, x_train, h) {
  K <- function(x) {dnorm(x, mean = 0, sd = 1)}
  # S_1
  S_1 <- sum(K((x_train - x)/h) * (x_train - x))
  S_2 <- sum(K((x_train - x)/h) * (x_train - x)^2)
  # b_i in vector
  b <- K((x_train - x)/h) * (S_2 - (x_train - x) * S_1)
  l <- b/sum(b)
  return(l)
}

local_linear_smoother <- function(x, x_train, h) { # the L matrix
  L <- c()
  for (elem in x) {
    l <- local_linear_effective_kernal(elem, x_train, h)
    L <- append(L, l)
  }
  n <- length(x_train)
  m <- length(x)
  L <- matrix(L, nrow = m, ncol = n, byrow = TRUE)
  return(L)
}

local_linear_regressor <- function(y_train, x, x_train, h) {
  L <- local_linear_smoother(x, x_train, h)
  r_hat <- L %*% y_train %>%
    as.vector()
  return(r_hat)
}

loocv_risk <- function(y_train, x, x_train, h) {
  L_ii <- diag(local_linear_smoother(x, x_train, h))
  r_hat <- local_linear_regressor(y_train, x, x_train, h)
  estimated_risk <- mean(((y_train - r_hat)/(1 - L_ii))^2)
  return(estimated_risk)
}

loocv_risk.best_bandwidth <- function(y_train, x_train, from, to, step_size) {
  bandwidths <- seq(from, to, by = step_size)
  score <- c()
  for (h in bandwidths) {
    score <- append(score, loocv_risk(y_train, x = x_train, x_train = x_train, h = h))
  }
  return(list(score = score,
              bandwidths = bandwidths,
              optim_bandwidth = bandwidths[which.min(score)]))
}

get_opt_plugin_bandwidth <- function(df, C) {
  pilot.fit <- lm(y ~ x + I(x^2) + I(x^3) + I(x^4), data = df)
  pilot.coef <- coef(pilot.fit)
  x <- df$x
  y <- df$x
  pilot.est <- fitted(pilot.fit) %>% unname()
  pilot.est_2nd <- 2*pilot.coef[3] + 6*pilot.coef[4]*x + 12*pilot.coef[5]*x^2
  
  sigma_tildaSq <- mean((y - pilot.est)^2)
  range <- diff(range(x))
  return(
    ((C*sigma_tildaSq*range)/sum(pilot.est_2nd^2))^(1/5) 
  )
}

empirical_risk <- function(y_hat, y_true) {
  (y_hat - y_true)^2 %>%
    mean()
}
