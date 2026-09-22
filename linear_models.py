import numpy as np

class LinearRegression():
    def __init__(self, x_train, y_train, x_test, y_test,  lr=0.001, n_iterations=1000): # lr is learning rate, n_iterations is number of iterations in gradient descent? Could jus u
        self.weights = None
        self.bias = None
        
        self.x_train = x_train
        self.y_train = y_train 
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.beta_0 = 0
        self.beta_1 = 0
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        
    # def get_parameters(self):
        
    def compute_mse(self, y, y_hat):
        """
        Computes mse based on x and 
        
        Input 
        -----
        - x
            - np array or list of x-values (assumed 1-dimensional since this is linear regression case)
        - y 
            - np array or list of x-values (assumed 1-dimensional since this is linear regression case)
            
        Output 
        ------
        - mse
            - scalar value of mse for given values of y and x
        """
        y = np.array(y)
        y_hat = np.array(y_hat)
        
        n = len(y)
        
        difference = y-y_hat
        difference_squared = difference**2
        
        mse = difference_squared/n
        
        return mse 
    
    def estimate_y_hat(self):
        """
        Estimates y_hat from x with parameter values beta_0 and beta_1
        
        Input
        -----
        - x
            - vector of input_valaues
        Output
        ------
        - y_hat
            - The estimated response from our input vector x and parameter values beta_0 and beta_1
        
        """
        
        y_hat = self.beta_0 +self.beta_1 * self.x_train
        
        return y_hat 
        
    def step_g_d(self): # Not yet implemented
        """
        Performs one step of the gradient-descent-algorithm 
        """
        return "hello"
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        
        beta_0 = 0
        beta_1 = 0
        
        return beta_0, beta_1
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