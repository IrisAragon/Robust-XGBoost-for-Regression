#--------------------------------------------------------------------------------------------------
#   Example of implementation of robust loss functions for regression in XGBoost
#--------------------------------------------------------------------------------------------------

# Number of training and test samples, and number of predictors
n_train <- 100
n_test <- 100
p <- 10

# Simulate predictor and response data
X_train <- matrix(rnorm(n_train*p), nrow = n_train, ncol = p)
X_test <- matrix(rnorm(n_test*p), nrow = n_test, ncol = p)
y_train <- rnorm(n_train)
y_test <- rnorm(n_test)

# Create XGBoost matrices for training and testing
# `label` stores the target values for regression
 dtrain <- xgb.DMatrix(X_train, label = y_train)
 dtest <- xgb.DMatrix(X_test, label = y_test)

# Set a common initial prediction equal to the median response for all observations
initial_median_train <- rep(median(y_train), length(y_train))
initial_median_test  <- rep(median(y_train), nrow(X_test))

# Hyperparameters for the tree ensemble
pars <- list(tree_method = "exact", max_depth = 1)

# M-XGBoost 
scale.mad <- mad(y_train)
mxg.model <- xgb.train(data = dtrain, objective = m_loss_huber(scale = scale.mad, b = 1.345), params = pars, nrounds = 100)
mxg.predictions <- predict(mxg.model, X_test, base_margin = initial_median_test)

# S-XGBoost
sxg.model <- xgb.train(data = dtrain, objective = s_loss_tukey(c = 1.465, delta = 0.5), params = pars, nrounds = 100)
sxg.predictions <- predict(sxg.model, X_test, base_margin = initial_median_test)

# Tau^2-XGBoost
tauxg.model <- xgb.train(data = dtrain, objective = tau_loss_tukey(c.tuk = 1.465, c.m.tuk = 4.685, delta = 0.5, kappa.tuk = 0.2), params = pars, nrounds = 100)
tauxg.predictions <- predict(tauxg.model, X_test, base_margin = initial_median_test)

# MM-XGBoost: two-stage robust procedure
# First stage: fit a robust S-XGBoost model
sxg.stage1 <- xgb.train(data = dtrain, objective = s_loss_tukey(c = 1.465, delta = 0.5), params = pars, nrounds = 50)
pred_train <- predict(sxg.stage1, X_train, base_margin = initial_median_train)
res <- y_train - pred_train

# Estimate the robust scale from residuals for the second stage
scale.mm <- scaleM(res, delta = delta)

# Second stage: use MM-loss with the robust scale estimate
dtrain2 <- xgb.DMatrix(X_train, label = y_train, base_margin = pred_train)
mmxg.model <- xgb.train(data = dtrain2, objective = mm_loss_tukey(scale = scale.mm, c = 4.685), params = pars, nrounds = 50)
mm.predictions <- predict(mmxg.model, X_test, base_margin = predict(sxg.stage1, X_test, base_margin = initial_median_test))

# Compute RMSE for each robust XGBoost variant
rmse_mxg <- sqrt(mean((y_test - mxg.predictions)^2))
rmse_sxg <- sqrt(mean((y_test - sxg.predictions)^2))
rmse_tauxg <- sqrt(mean((y_test - tauxg.predictions)^2))
rmse_mmxg <- sqrt(mean((y_test - mm.predictions)^2))

# Summarize the results in a comparison table
 table_rmse <- data.frame(Method = c("M-XGBoost", "S-XGBoost", "Tau^2-XGBoost", "MM-XGBoost"),
                          RMSE = c(rmse_mxg, rmse_sxg, rmse_tauxg, rmse_mmxg))
