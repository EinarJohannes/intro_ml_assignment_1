import numpy as np

class LinearRegression():
    def __init__(self, x_train, y_train, x_test, y_test,  lr=0.001, n_iterations=1000): # lr is learning rate, n_iterations is number of iterations in gradient descent? Could just use convergence in interative difference in loss-function
        
        self.weights = None
        self.bias = None
        
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train) 
        
        self.x_test = np.array(x_test) # Should not touch to begin with
        self.y_test = np.array(y_test)
        
        self.beta_0 = 0
        self.beta_1 = 0
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        

    def estimate_y_pred(self, beta_0, beta_1):
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
    
        y_pred = beta_0 + beta_1 * self.x_train
        
        return y_pred 
        
    def compute_mse(self, beta_0, beta_1):
        """
        Computes mse based on y_pred and actual y
        
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
        
        y_pred = self.estimate_y_pred(beta_0, beta_1)
        difference = self.y_train - y_pred
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
        
        grad_beta_0 = (self.compute_mse(self.beta_0 + delta_beta_0, self.beta_1) - self.compute_mse(self.beta_0, self.beta_1)) / delta_beta_0
        grad_beta_1 = (self.compute_mse(self.beta_0, self.beta_1 + delta_beta_1) - self.compute_mse(self.beta_0, self.beta_1)) / delta_beta_1
        
        return grad_beta_0, grad_beta_1
    
        
    def step_g_d(self):
        
        # Don't really need these two under here, because they are calculated in self.compute_gradient()
        # y_pred = self.estimate_y_pred(beta_0 = self.beta_0, beta_1 = self.beta_1)
        # mse = self.compute_mse(self.beta_0, self.beta_1)
        grad_beta_0, grad_beta_1 = self.compute_gradient()

        self.beta_0 = self.beta_0 - self.lr * grad_beta_0
        self.beta_1 = self.beta_1 - self.lr * grad_beta_1
        
    def fit(self):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        
        mse = self.compute_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
        self.loss_history.append(mse)
        
        for _ in range(self.n_iterations):
            self.step_g_d()
            mse = self.compute_mse(beta_0 = self.beta_0, beta_1 = self.beta_1)
        # Calculating estiamte for y_pred
        
        return self.beta_0, self.beta_1
        # ====================================
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X):
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
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")