# COMP5329 - Deep Learning
# Assignment 2
# Authors: King Tao Ng and Anjali Upadhyay
# glm.py: Generalised Linear Models are used as baselines. 

from patsy import dmatrices
import statsmodels.api as sm

class GLM:
    
    def __init__(self, formula):
        self.formula = formula
        
    def poisson(self, training_data):
        response, predictors = dmatrices(self.formula, training_data, return_type='dataframe')
        self.results = sm.GLM(response, predictors, family=sm.families.Poisson(link=sm.families.links.identity)).fit()

    def negative_binomial(self, training_data, alpha):
        response, predictors = dmatrices(self.formula, training_data, return_type='dataframe')
        self.results = sm.GLM(response, predictors, family=sm.families.NegativeBinomial(link=sm.families.links.identity, alpha=alpha)).fit()
    
    # Predict
    def predict(self, test_data):
        return self.results.summary(), self.results.predict(test_data)
    
