from .Future import Future
from statsmodels.base.model import LikelihoodModel
from .Helper import check_count, start_year, current_year
import pandas as pd
import os

"""# Defines Class LongPast which inherits from Past"""

# Class that functions are used when "Extrapolate" and "Long Term" selected in Settings.html
class LongFuture(Future):
    # class variables 
    exog_targets = ["% Change GDP", "% Change Fuel", "time"] # Override Predictor definition
    xlabel = "Years since " + str(start_year)
    monthly = False # predictions are made on a yearly basis

    # class defintion - see Predictor's __init__() method
    def __init__(self, models, weights, matplot, params, path, filter, covid):
        super().__init__(models, weights, matplot, params, path, filter, covid)
    
    def predict(self, time, data, exog, exog2, boole2, name, global_exo):
        dct = {}
        ind = -1 * LongFuture.max_iter  # Max number of models to be trained (default 1000 from Predictor)
        for key in data.keys():
            include = include = self.filter.filter_keys(key)
            if ind < 0 and data[key] is not None and include and check_count(data[key], current_year - start_year):
                dct = {key: [] for key in range(len(data[key]["time"]), len(data[key]["time"]) +  time)}
                all_times = pd.Series(range(len(data[key]["time"]), len(data[key]["time"]) +  time))
                time_df = pd.DataFrame({"time": all_times})
                merged0 = pd.merge(time_df, data[key], on="time", how="outer")
                # Define dataframes so they have the appropriate features (exog/endog)
                merged = pd.merge(merged0, global_exo, on = "time", how = "outer")[LongFuture.exog_targets + \
                                                    [LongFuture.target]] if global_exo is not None else merged0
                # include covid dummy variable
                if self.covid.get_include():
                    merged["Covid"] = merged["time"].apply(self.covid.get_function())

                 # Sort merged DataFrame by 'time'
                merged = merged.sort_values(by='time').reset_index(drop=True)
                
                 # Determine split index based on time (current year as a time)
                split_index = current_year - start_year

                # Use .iloc for slicing (past/future)
                endog_past = merged.iloc[:split_index][LongFuture.target]
                exog_past = merged.iloc[:split_index].drop(columns=["Total Pax"])
                exog_future = merged.iloc[split_index:split_index+ time + 1].drop(columns=["Total Pax"])

                count = 1
                for model in self.models:
                    mod = model(endog_past, exog_past if not exog_past.empty else None) 
                    model_fit = mod.fit()

                    # Export model parameters if recquired
                    if self.params:
                        self.get_model_params(model_fit, self.path, name, time, list(key), count)

                    # Make prediction (accounting for several model types) 
                    if isinstance(mod, LikelihoodModel):   
                        pred = model_fit.predict(len(data[key]["time"]), len(data[key]["time"]) +  time-1,
                                                exog = (exog_future if not exog_future.empty else None))
                    else:
                        forecast = model_fit.forecast(start=len(data[key]["time"]), horizon=time-1 , x = None)
                        pred = forecast.mean.stack().reset_index(drop=True)
                    
                    for index, value in enumerate(pred):
                        dct[index+len(data[key]["time"])].append(value)
                    count += 1

                # Make report directory
                if self.path is not None:
                    if not os.path.exists(self.path + name + "/"):
                        os.makedirs(self.path + name + "/")
                
                # Creates matplotlib graphs and saves them to report directory
                if self.matplot:
                    self.save_graphs(data[key], dct, time, list(key), self.path, self.weights, name, boole2)

                # Generates the excel file containing all the date (boole2 is True if range of values should be reported)
                if boole2:
                    self.save_numbers(dct, data[key], time, list(key), self.path, pd.Series([None] * time), name, self.weights)

                else:
                    self.save_numbers2(dct, data[key], time, list(key), self.path, pd.Series([None] * time), name, self.weights)

                ind += 1

    def get_model_params(self, model, path, name, time, keys, number):
        super().get_model_params(model, path, name, time, keys, number)

    def save_graphs(self, x, dct, time, key, path, weights, name, boole2):
        super().save_graphs(x, dct, time, key, path, weights, name, boole2)
    
    def save_numbers(self, dct, x, time, key, path, seats, name, weights):
        super().save_numbers(dct, x, time, key, path, seats, name, weights)

    def save_numbers2(self, dct, x, time, key, path, seats, name, weights):
        super().save_numbers2(dct, x, time, key, path, seats, name, weights)

    # Returns a str that gives the date in YYYY format given a time inputted by the user
    # time = integer inputted by the user in PAARM.py which represents the number of months since current_year
    
    def get_date(self, time):
       return str(time + current_year)