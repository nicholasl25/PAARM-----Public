"""### Covid Class that determines if and how to incorperate Covid dummy variables"""

class Covid:

    # Class definition
    # include: boolean that is True if Covid should be included as a dummy variable
    # start: integer that determines start time of Covid dummy variable
    # end: integer that determines start time of Covid dummy variable
    # Covid dummy variable will be set to 1 on the interval [start, end] and o 0 elsewhere
    def __init__(self, include, start, end):
        assert isinstance(include, bool), "NICK SAYS: include parameter must be a boolean in Covid object."
        if include:
            assert isinstance(start, int), "NICK SAYS: start parameter must be a integer in Covid object."
            assert isinstance(end, int), "NICK SAYS: end parameter must be a integer in Covid object."
            assert end >= start, "NICK SAYS: start (month/year) of Covid dummy variable interval must be no larger \n \
                                    than end of Covid dummy variable interval."
        # set Covid fields 
        self.include = include
        self.start = start
        self.end = end

    # Returns a boolean that determines in a Covid dummy variable should be included in the model
    # simple helper method which is called in all Predictor subclasses .predict() methods
    def get_include(self):
        return self.include
    
    # Return as a lambda expression that returns 1 if input is within 
    # the interval [start, end] and 0 otherwise
    # simple helper method which is called in all Predictor subclasses .predict() methods
    def get_function(self):
        return lambda x: 1 if 18 <= x <= 19 else 0
