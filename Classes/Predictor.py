from .Filter import Filter
from .Covid import Covid

"""# Class Predictor which is the parent class of Past and Future"""

# Class Predictor should be used as a template only 
# WARNING: If you are inheriting from Predictor make sure to overwrite 
# all methods that are used (or assertion error will thrown)

class Predictor:
    # Predictor class variables: target, exog_targets, and include_covid which could be changed manually here
    target = "Total Pax" # target variable to be predicted
    max_iter = 1000 # integer representing the maximum number of models to be trained
    exog_targets = ["Total Seats", "GDP Percent Change", "time"] #can be changed depending on what exogenous variables should be included
    monthly = True # bool that is True if data points are given on a monthly basis and False when given 
                    # on a yearly basis
    xlabel = "Months since Jan 2016" # label for x-axis of graphs 
    

    #models = set of models to be used in predictions
    #weights = array of numbers that determine how each model should be weighted in final average
    #matplot = boolean that is True if matplotlib graphs should be exported and False otherwise
    #params = boolean that is True if model parameters should be exported for every model trained
    #path = string that contains the filepath the report should be saved into
    #filter = object of class Filter that decides what keys(datasets) should be included in the report
    #covid = object of class Covid that decides if and how an Covid exogenous dummy variable
            # should be incorperated into the model

    # Class defintion
    def __init__(self, models, weights, matplot, params, path, filter, covid):
        assert len(models) == len(weights), "NICK SAYS: The number of models must equal the number of weights"
        assert isinstance(matplot, bool), "NICK SAYS: matplot must be a boolean (True/False)"
        assert isinstance(params, bool), "NICK SAYS: params must be a boolean (True/False)"
        assert isinstance(path, str), "NICK SAYS: path must be of type string (surrounded by quotes)"
        assert isinstance(filter, Filter), "NICK SAYS: filter must be of class Filter"
        assert isinstance(covid, Covid), "NICK SAYS: covid input must be of class Covid"
        self.models = models
        self.weights = weights
        self.matplot = matplot
        self.params = params
        self.path = path
        self.filter = filter
        self.covid = covid

    def predict(self):
        raise AssertionError("NICK SAYS: Method .predict() not overwritten in Predictor Parent Class")

    def get_model_params(self):
        raise AssertionError("NICK SAYS: Method .get_model_params() not overwritten in Predictor Parent Class")

    def save_graphs(self):
        raise AssertionError("NICK SAYS: Method .save_graphs() not overwritten in Predictor Parent Class")
    
    def save_numbers(self):
        raise AssertionError("NICK SAYS: Method .save_numbers() not overwritten in Predictor Parent Class")

    def save_numbers2(self):
        raise AssertionError("NICK SAYS: Method .save_numbers2() not overwritten in Predictor Parent Class")
    
    def get_date(self):
        raise AssertionError("NICK SAYS: Method .get_date() not overwritten in Predictor Parent Class")