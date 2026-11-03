#-----------------------------------------------------------------------------------
#   Implementation of the loss functions for robust XGBoost methods for regression
#-----------------------------------------------------------------------------------

#------------------------------------------
# 1.  M-loss (with Huber)
#------------------------------------------

# loss function in function of a fixed scale and tuning constant b

m_loss_huber <- function(scale, b){
  function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    errors <- labels - preds
    n <- length(errors)
    
    errors.s <- errors/scale
    
    grad <- - psi_huber(errors.s, b)*scale 
    
    hess <- psi_prime_huber(errors.s, b)
    
    return(list(grad = grad, hess = hess))
  }}



#------------------------------------------
# 2.  S-loss (with Tukey's bisquare)
#------------------------------------------

# loss function in function of tuning constant c and breakdownvalue delta

s_loss_tukey <- function(c, delta){
  function(preds, dtrain) {
    labels <- getinfo(dtrain,"label") 
    errors <- labels - preds
    
    n <- length(errors)
    scale <- RobStatTM::scaleM(errors, delta = delta, tuning.chi = c) 
    
    errors.s <- errors/scale
    
    psi <- psi_tukey(errors.s, c)
    psi_prime <- psi_prime_tukey(errors.s, c)
    
    A <- sum(errors.s * psi)
    B <- sum(errors.s * (psi + errors.s * psi_prime))
    
    grad.1 <- - psi / A
    
    hess.1 <- (psi_prime / (scale * A)) -
      (2 * psi_prime * psi * errors.s) / (scale * A^2) +
      (psi^2 / (scale * A^2)) * (B / A - 1)
    
    grad <- n*scale*grad.1
    
    hess <- n*scale*hess.1 + n*grad.1^2
    
    return(list(grad = grad, hess = hess))
  }
}



#---------------------------------------------
# 3.  Tau^2-loss (with Tukey's bisquare)
#---------------------------------------------

# rho1 = Tukey's bisquare with tuning constant c.tuk and value kappa.tuk
# rho = Tukey's bisquare with tuning constant c.m.tuk

# loss function in function of tuning parameters: c.tuk, c.m.tuk, breakdownvalue: delta and value kappa.tuk

tau_loss_tukey <- function(c.tuk, c.m.tuk, delta, kappa.tuk){
  function(preds, dtrain) {
    labels <- getinfo(dtrain,"label") 
    errors <- labels - preds
    
    n <- length(errors)
    scale <- RobStatTM::scaleM(errors, delta = delta, tuning.chi = c.tuk) 
    
    errors.s <- errors/scale
    
    rho <- rho_tukey(errors.s, c.m.tuk)
    
    psi <- psi_tukey(errors.s, c.m.tuk)
    psi_prime <- psi_prime_tukey(errors.s, c.m.tuk)
    
    psi1 <- psi_tukey(errors.s, c.tuk)
    psi_prime1 <- psi_prime_tukey(errors.s, c.tuk)
    
    A <- sum(errors.s * psi)
    A1 <- sum(errors.s * psi1)
    
    B <- sum(errors.s * (psi + errors.s * psi_prime))
    B1 <- sum(errors.s * (psi1 + errors.s * psi_prime1))
    
    C <- sum(rho)
    
    grad.sigma <- - psi1 / A1
    
    hess.sigma <- (psi_prime1 / (scale * A1)) -
      (2 * psi_prime1 * psi1 * errors.s) / (scale * A1^2) +
      (psi1^2 / (scale * A1^2)) * (B1 / A1 - 1)
    
    grad.1 <- (- psi1*scale/n)*(2*C/A1 - A/A1) - (psi*scale)/n
    
    hess.1 <- 1/n*(grad.sigma^2*(2*C+B-3*A) + hess.sigma*(2*scale*C - scale*A) 
                   + grad.sigma*(2*psi_prime*errors.s - 2*psi) + psi_prime)
    
    grad <- (n/(2*kappa.tuk))*grad.1
    
    hess <- (n/(2*kappa.tuk))*hess.1
    
    return(list(grad = grad, hess = hess))
  }
}


#-----------------------------------------------
# 4.  MM-XGBoost (with Tukey's bisquare)
#-----------------------------------------------

# Step 1: fixed scale estimator: M-estimator of scale from predictions of S-XGBoost 

# Step 2: M-step

# loss function in function of fixed scale estimate (from stage 1) and tuning constant c

mm_loss_tukey <- function(scale, c){
  function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    errors <- labels - preds
    n <- length(errors)
    
    errors.s <- errors/scale
    
    grad <- - psi_tukey(errors.s, c)*scale 
    
    hess <- psi_prime_tukey(errors.s, c)
    
    return(list(grad = grad, hess = hess))
  }}


