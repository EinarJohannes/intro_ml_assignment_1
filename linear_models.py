import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, x_train, y_train, x_test, y_test,  lr=0.001, n_iterations=1000): # lr is learning rate, n_iterations is number of iterations in gradient descent? Could just use convergence in interative difference in loss-function
        
        self.weights = None
        self.bias = None
        
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train) 
        
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        
        self.beta_0 = 0
        self.beta_1 = 0
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.training_loss_history = []
        self.test_loss_history = []
        

    def estimate_y_pred(self, beta_0, beta_1, x):
        """
        Estimates y_pred from x with parameter values beta_0 and beta_1
        
        Input
        -----
        - beta_0
            - float. constant term in linear model
        - beta_1 
            - float. growing factor in linear model 
        Output
        ------
        - y_pred
            - The estimated response from our input vector x and parameter values beta_0 and beta_1
        """
    
        y_pred = beta_0 + beta_1 * x
        
        return y_pred 
        
    def compute_training_mse(self, beta_0, beta_1):
        """
        Computes training mse based on y_pred and actual y
        
        Input 
        -----
        - beta_0
            - paramter in linear model used to estimate y_pred
        - beta_1 
            - parameter in linear model used to estimate y_pred
        
        Output 
        ------
        - mse
            - scalar value of mse for given values of y and x
        """
        
        y_pred = self.estimate_y_pred(beta_0, beta_1, self.x_train)
        difference = self.y_train - y_pred
        difference_squared = difference**2
        
        sum = np.sum(difference_squared)
        n = len(y_pred)
        
        mse = sum/n
        
        return mse 

    def compute_test_mse(self, beta_0, beta_1):
        """
        Compues test mse based on y_pred and actual y
        
        Input
        -----
        - beta_0
            - parameter in linear model used to estimate y_pred from self.x_test
        - beta_1
            - parameter in linear model used to estimate y_pred from self.x_test
            
        Outpu
        - mse
            - mse evaluated on test dataset, which has not been used for training
        """
        
        y_pred = self.estimate_y_pred(beta_0, beta_1, self.x_test)
        difference = self.y_test - y_pred
        difference_squared = difference**2
        
        sum = np.sum(difference_squared)
        n = len(y_pred)
        
        mse = sum/n
        
        return mse
        
    def compute_gradient(self, delta_beta_0 = 0.01, delta_beta_1 = 0.01):
        """
        Computes gradient of mse where beta_0 and beta_1 are as defined
        
        Input
        -----
        - delta_beta_0
            - used as the "infinitesimal" step to evaluate gradient for beta_0
        - delta_beta_1
            - used as the "infinitesimal" step to evaluate gradient for beta_1
            
        Output
        ------
        - grad_beta_0
            - beta_0 component of gradient
        - grad_beta_1
            - beta_1 component of gradient
        """
        grad_beta_0 = 0
        grad_beta_1 = 0
        
        grad_beta_0 = (self.compute_training_mse(self.beta_0 + delta_beta_0, self.beta_1) - self.compute_training_mse(self.beta_0, self.beta_1)) / delta_beta_0
        grad_beta_1 = (self.compute_training_mse(self.beta_0, self.beta_1 + delta_beta_1) - self.compute_training_mse(self.beta_0, self.beta_1)) / delta_beta_1
        
        return grad_beta_0, grad_beta_1
    
        
    def step_g_d(self):
        
        grad_beta_0, grad_beta_1 = self.compute_gradient()

        self.beta_0 = self.beta_0 - self.lr * grad_beta_0
        self.beta_1 = self.beta_1 - self.lr * grad_beta_1
        
    def fit(self):
        """
        Estimates parameters for the classifier. Estimates are stored as attributes for the class. 
        Loss history can be found in self.loss_history
        """
        # ====================================
        # YOUR CODE GOES HERE
        
        training_mse = self.compute_training_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
        self.training_loss_history.append(training_mse)
        
        test_mse = self.compute_test_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
        self.test_loss_history.append(test_mse)
        
        for _ in range(self.n_iterations):
            self.step_g_d()
            training_mse = self.compute_training_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
            self.training_loss_history.append(training_mse)
            
            test_mse = self.compute_test_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
            self.test_loss_history.append(test_mse)
            
            
        # ====================================
        # raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, x):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        
        self.fit()
        
        # Erasing history in case this is done several times for different input values of x
        self.training_loss_history = []
        self.test_loss_history = []
        
        prediction = self.beta_0 + self.beta_1 * x
        
        return prediction
        # raise NotImplementedError("LinearRegression.predict is not implemented yet.")
        
        
    def plot_training_loss(self, from_index = 4):
        """
        Scatterplot of training loss history
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 4))
        plt.scatter(range(from_index, len(self.training_loss_history)), self.training_loss_history[from_index:], label='Training Loss', color='blue')
        plt.title('Training Loss History from index ' + str(from_index), fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_test_loss(self):
        """
        Scatterplot of test loss history
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 4))
        plt.scatter(range(len(self.test_loss_history)), self.test_loss_history, label='Test Loss', color='red')
        plt.title('Test Loss History', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('MSE', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_fitted_line_on_training_data(self):
        """
        Plots the fitted regression line together with the training data points
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 4))
        plt.scatter(self.x_train, self.y_train, label='Data Points', color='blue')
        plt.plot(self.x_train, self.predict(self.x_train), label='Fitted Line', color='red')
        plt.title('Fitted Line vs Data Points', fontsize=16)
        plt.xlabel('Network Activity', fontsize=14)
        plt.ylabel('Energy', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()

class LogisticRegression():
    def __init__(self, x_train, y_train, x_test, y_test, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.beta_0 = 0
        self.beta_1 = 0
        self.beta_2 = 0
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.training_likelihood_history = []
        
    def calculate_prediction_accuracy(self): # Not implemented yet
        y_pred = self.predict()
        y_test = self.y_test
        
        # Vector with zeros or ones depending on whether the prediction is correct or not.
        evaluation_vector = np.array(y_pred == y_test)
        evaluation_score = np.sum(evaluation_vector) / len(evaluation_vector)
        return evaluation_score
            
    def plot_training_loss_history(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 4))
        plt.scatter(range(len(self.training_likelihood_history)), self.training_likelihood_history, label='Training Likelihood', color='blue')
        plt.title('Training Likelihood History', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Likelihood', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def calculate_bernoulli_density(self, p, k):
        """
        Calculating probability that X = k, k in {0,1} given that probability for X = 1 is p.
        Assuming X is bernoulli(p)
        """
        result = p**k * (1-p)**(1-k)
        return result
        
    def calculate_linear_predictor(self, beta_0, beta_1, beta_2, x0, x1):
        """
        Calculating linear predictor eta_i
        
        Input
        -----
        - All parameters are scalars or np arrays
        """
        
        return beta_0 + beta_1*x0 + beta_2*x1
    
    
    def calculate_likelihood(self, beta_0, beta_1, beta_2):
        """
        Calculates probability of observing Y = y, given probability with expectation equal to sigmoid(eta), eta = X*beta
        
        Input
        -----
        - beta_0
            - constant term in linear predictor 
        - beta_1 
            - first "linear factor" of linear predictor
        - beta_1 
            - second linear facotr of linear predictor 
        
        Output
        ------
        
        - likelihood
            - The probability of observing Y = y, given our parameter valules for beta_0, beta_1 and beta_2
        """
        
        likelihood = 1
        x = self.x_train
        y = self.y_train
        
        for i in range(len(x)):
            eta = self.calculate_linear_predictor(
                beta_0 = beta_0,
                beta_1 = beta_1,
                beta_2 = beta_2,
                x0 = x[i][0],
                x1 = x[i][1],
            )
            sigmoid = self.sigmoid(eta)
            bernoulli_probability = self.calculate_bernoulli_density(p = sigmoid, k = y[i])
            likelihood *= bernoulli_probability
        
        return likelihood

    def calculate_log_likelihood(self, beta_0, beta_1, beta_2):
        x = np.asarray(self.x_train)
        y = np.asarray(self.y_train)

        eta = self.calculate_linear_predictor(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            x0=x[:, 0],
            x1=x[:, 1],
        )
        probabilities = np.clip(self.sigmoid(eta), 1e-12, 1 - 1e-12)

        return np.sum(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )
    
    def gradient(self):
        """
        Calculates the gradient of the loss function where the loss function 
        is assumed to be a function of the beta vector. Have also assumed delta to be 0.01
        
        Input
        -----
        - beta_0, beta_1, beta_2
            - Should be the current state of the beta-vector
        """
        x = np.asarray(self.x_train)
        y = np.asarray(self.y_train)
        beta = np.array([self.beta_1, self.beta_2])
        probabilities = self.sigmoid(self.beta_0 + x @ beta)
        error = y - probabilities

        return (
            np.sum(error),
            np.sum(error * x[:, 0]),
            np.sum(error * x[:, 1]),
        )
    
    def step_g_d(self):
        """
        Takes no input-parameters. Performs one step of gradient-descent. 
        Uses attributes of this instance of the class logisticregression.
        """
        
        gradient_beta_0, gradient_beta_1, gradient_beta_2 = self.gradient()
        
        self.beta_0 = self.beta_0 + self.lr * gradient_beta_0
        self.beta_1 = self.beta_1 + self.lr * gradient_beta_1
        self.beta_2 = self.beta_2 + self.lr * gradient_beta_2

    def fit(self):
        """
        Should fit parameters beta_0, beta_1, beta_2 to the model
        
        Input
        -----
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        
        #Resetting loss history in case it has been used before. 
        self.training_likelihood_history = []
        
        self.training_likelihood_history.append(self.calculate_log_likelihood(
            beta_0 = self.beta_0,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2
        ))
        
        for _ in range(self.n_iterations):
            self.step_g_d()
            log_likelihood = self.calculate_log_likelihood(
                beta_0 = self.beta_0,
                beta_1 = self.beta_1,
                beta_2 = self.beta_2
            )
            self.training_likelihood_history.append(log_likelihood)
                
    def predict_proba(self):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        
        beta_vector = np.array([self.beta_1, self.beta_2])
        
        sigmoid_vector =self.sigmoid(self.beta_0 + self.x_test @ beta_vector)
        return sigmoid_vector
    
    
    def predict(self):
        probabilities = self.predict_proba()
        predictions_vector = probabilities > 0.5
        
        return predictions_vector
        
    def sigmoid(self, z):
        """
        Evaluates sigmoid function of z
        
        Input
        -----
        - z
        
        Output
        ------
        - sigmoid(z) = logit^{-1}(z)
        """
        
        z = np.asarray(z)
        result = np.empty_like(z, dtype=float)
        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1 + exp_z)
        return result
    
    
    
class LogisticRegression2():
    def __init__(self, x_train, y_train, x_test, y_test, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.beta_0 = 0
        self.beta_1 = 0
        self.beta_2 = 0
        self.beta_3 = 0
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.training_likelihood_history = []
        
    def calculate_prediction_accuracy(self): # Not implemented yet
        y_pred = self.predict()
        y_test = self.y_test
        
        # Vector with zeros or ones depending on whether the prediction is correct or not.
        evaluation_vector = np.array(y_pred == y_test)
        evaluation_score = np.sum(evaluation_vector) / len(evaluation_vector)
        return evaluation_score
            
    def plot_training_loss_history(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 4))
        plt.scatter(range(len(self.training_likelihood_history)), self.training_likelihood_history, label='Training Likelihood', color='blue')
        plt.title('Training Likelihood History', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Likelihood', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def calculate_bernoulli_density(self, p, k):
        """
        Calculating probability that X = k, k in {0,1} given that probability for X = 1 is p.
        Assuming X is bernoulli(p)
        """
        result = p**k * (1-p)**(1-k)
        return result
        
    def calculate_linear_predictor(self, beta_0, beta_1, beta_2, beta_3, x0, x1, x2):
        """
        Calculating linear predictor eta_i
        
        Input
        -----
        - All parameters are scalars or np arrays
        """
        
        return beta_0 + beta_1*x0 + beta_2*x1 + beta_3 * x2
    
    
    def calculate_likelihood(self, beta_0, beta_1, beta_2, beta_3):
        """
        Calculates probability of observing Y = y, given probability with expectation equal to sigmoid(eta), eta = X*beta
        
        Input
        -----
        - beta_0
            - constant term in linear predictor 
        - beta_1 
            - first "linear factor" of linear predictor
        - beta_2 
            - second linear facotr of linear predictor 
        - beta_3 
            - third linear factor of linear predictor 

        Output
        ------
        
        - likelihood
            - The probability of observing Y = y, given our parameter valules for beta_0, beta_1 and beta_2
        """
        
        likelihood = 1
        x = self.x_train
        y = self.y_train
        
        for i in range(len(x)):
            eta = self.calculate_linear_predictor(
                beta_0 = beta_0,
                beta_1 = beta_1,
                beta_2 = beta_2,
                beta_3 = beta_3,
                x0 = x[i][0],
                x1 = x[i][1],
                x2 = x[i][2]
            )
            sigmoid = self.sigmoid(eta)
            bernoulli_probability = self.calculate_bernoulli_density(p = sigmoid, k = y[i])
            likelihood *= bernoulli_probability
        
        return likelihood

    def calculate_log_likelihood(self, beta_0, beta_1, beta_2, beta_3):
        x = np.asarray(self.x_train)
        y = np.asarray(self.y_train)

        eta = self.calculate_linear_predictor(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            x0=x[:, 0],
            x1=x[:, 1],
            x2=x[:, 2],
        )
        probabilities = np.clip(self.sigmoid(eta), 1e-12, 1 - 1e-12)

        return np.sum(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )
    
    def gradient(self):
        """
        Calculates the gradient of the loss function where the loss function 
        is assumed to be a function of the beta vector. Have also assumed delta to be 0.01
        
        Input
        -----
        - beta_0, beta_1, beta_2, beta_3
            - Should be the current state of the beta-vector
        """
        x = np.asarray(self.x_train)
        y = np.asarray(self.y_train)
        beta = np.array([self.beta_1, self.beta_2, self.beta_3])
        probabilities = self.sigmoid(self.beta_0 + x @ beta)
        error = y - probabilities

        return (
            np.sum(error),
            np.sum(error * x[:, 0]),
            np.sum(error * x[:, 1]),
            np.sum(error * x[:, 2]),
        )

    def step_g_d(self):
        """
        Takes no input-parameters. Performs one step of gradient-descent. 
        Uses attributes of this instance of the class logisticregression.
        """
        
        gradient_beta_0, gradient_beta_1, gradient_beta_2, gradient_beta_3 = self.gradient()
        
        self.beta_0 = self.beta_0 + self.lr * gradient_beta_0
        self.beta_1 = self.beta_1 + self.lr * gradient_beta_1
        self.beta_2 = self.beta_2 + self.lr * gradient_beta_2
        self.beta_3 = self.beta_3 + self.lr * gradient_beta_3

    def fit(self):
        """
        Should fit parameters beta_0, beta_1, beta_2 to the model
        
        Input
        -----
        """
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        
        #Resetting loss history in case it has been used before. 
        self.training_likelihood_history = []
        
        self.training_likelihood_history.append(self.calculate_log_likelihood(
            beta_0 = self.beta_0,
            beta_1 = self.beta_1,
            beta_2 = self.beta_2,
            beta_3 = self.beta_3
        ))
        
        for _ in range(self.n_iterations):
            self.step_g_d()
            log_likelihood = self.calculate_log_likelihood(
                beta_0 = self.beta_0,
                beta_1 = self.beta_1,
                beta_2 = self.beta_2,
                beta_3 = self.beta_3
            )
            self.training_likelihood_history.append(log_likelihood)
                
    def predict_proba(self):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        
        beta_vector = np.array([self.beta_1, self.beta_2, self.beta_3])
        
        sigmoid_vector =self.sigmoid(self.beta_0 + self.x_test @ beta_vector)
        return sigmoid_vector
    
    
    def predict(self):
        probabilities = self.predict_proba()
        predictions_vector = probabilities > 0.5
        
        return predictions_vector
        
    def sigmoid(self, z):
        """
        Evaluates sigmoid function of z
        
        Input
        -----
        - z
        
        Output
        ------
        - sigmoid(z) = logit^{-1}(z)
        """
        
        z = np.asarray(z)
        result = np.empty_like(z, dtype=float)
        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1 + exp_z)
        return result 