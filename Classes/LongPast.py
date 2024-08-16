from .Past import Past
from .Helper import check_count, start_year, current_year
from statsmodels.base.model import LikelihoodModel
import pandas as pd
import os

"""# Defines Class LongPast which inherits from Past"""

# Class that functions are used when "Backtest" and "Long Term" is selected in Settings.html
class LongPast(Past):
    # class variables 
    exog_targets = ["% Change GDP", "% Change Fuel", "time"] # Override Predictor definition
    xlabel = "Years since " + str(start_year) # Override Predictor definition
    monthly = False # predictions are made on a yearly basis

    # class defintion - see Predictor's __init__() method
    def __init__(self, models, weights, matplot, params, path, filter, covid):
        super().__init__(models, weights, matplot, params, path, filter, covid)
    
    def predict(self, time, data, exog, exog2, boole2, name, global_exo):
        dct = {}
        ind = -1 * LongPast.max_iter # Max number of models to be trained (default 1000 from Predictor)
        for key in data.keys():
            include = self.filter.filter_keys(key)
            if ind < 0 and data[key] is not None and check_count(data[key], current_year - start_year) and \
                    len(data[key]["time"]) > time and include:
                assert len(data[key]["time"]) <= 2*time, "NICK SAYS: A time value given was too low, times should be at least half of data length"
                end = len(data[key]["time"])
                dct = {key: [] for key in range(time, int(end))}
                merged = pd.merge(data[key], global_exo, on = "time", how = "left")[LongPast.exog_targets + [LongPast.target]] \
                    if global_exo is not None else data[key]
                # include Covid as an exogenous variable that is 1 between March 2020 and Dec 2020 and 0 elsewhere
                if self.covid.get_include():
                    merged["Covid"] = merged["time"].apply(self.covid.get_function())
                merged = merged.drop(columns = ["time"])
                exo_df = merged.drop(columns = [LongPast.target])
                count = 1
                for model in self.models:
                    ind += 1
                    mod = model(merged[:time][LongPast.target], exo_df[:time] if not exo_df.empty else None)
                    model_fit = mod.fit()
                    # Export model parameters if recquired
                    if self.params:
                        self.get_model_params(model_fit, self.path, name, time, list(key), count)

                    # Make prediction (accounting for several model types)
                    if isinstance(mod, LikelihoodModel):
                        pred = model_fit.predict(time, end-1, exog = (exo_df[time:] if not exo_df.empty else None))
                    else:
                        forecast = model_fit.forecast(start=time, horizon=end-1, x=None)
                        pred = forecast.mean.stack().reset_index(drop=True)

                    for index, value in enumerate(pred):
                        dct[index+time].append(value)
                    count += 1
                
                # Make report directory
                if self.path is not None:
                    if not os.path.exists(self.path + name + "/"):
                        os.makedirs(self.path + name + "/")

                diff = (current_year - start_year) + 1

                # Creates matplotlib graphs and saves them to report directory
                if self.matplot:
                    self.save_graphs(data[key], dct, time, list(key), self.path, self.weights, name, boole2)

                # Generates the excel file containing all the date (boole2 is True if range of values should be reported)
                if boole2:
                    self.save_numbers(dct, data[key], time, list(key), self.path, pd.Series([None] * diff), name, self.weights)
                else:
                    self.save_numbers2(dct, data[key], time, list(key), self.path, pd.Series([None] * diff), name, self.weights)

    def get_model_params(self, model, path, name, time, keys, number):
        super().get_model_params(model, path, name, time, keys, number)

    def save_graphs(cls, x, dct, time, key, path, weights, name, boole2):
        super().save_graphs(x, dct, time, key, path, weights, name, boole2)
    
    def save_numbers(cls, dct, x, time, key, path, seats, name, weights):
        super().save_numbers(dct, x, time, key, path, seats, name, weights)

    def save_numbers2(cls, dct, x, time, key, path, seats, name, weights):
        super().save_numbers2(dct, x, time, key, path, seats, name, weights)


    # Returns a str that gives the date in YYYY format given a time inputted by the user
    # time = integer inputted by the user in PAARM.py which represents the number of months since start_year
    
    def get_date(self, time):
       return str(time + start_year)