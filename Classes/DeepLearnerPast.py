from .Past import Past
from .Helper import check_count, start_year, current_year
import pandas as pd
import tensorflow as tf
import os

"""# Defines Class DeepLearnerPast which inherits from Past"""

# Class that functions are used when "Backtest" and "Deep Learning" is selected in Settings.html
class DeepLearnerPast(Past):
    # class variables 
    
    # class defintion
    # LNE: Array of integers. LNE is an acronym that stands for Layers-Neurons-Epochs
    # this variable controls the hyperparameters of the neural network created in .predict()
    def __init__(self, models, weights, matplot, params, path, filter, covid, LNE):
        self.LNE = LNE
        super().__init__(models, weights, matplot, params, path, filter, covid)

    def predict(self, time, data, exog, exog2, boole2, name, global_exo):
        dct ={}
        for key in data.keys():
            include = self.filter.filter_keys(key)
            if data[key] is not None and key in exog.keys() and check_count(data[key]) and \
                check_count(exog[key]) and len(data[key]["time"]) > time and include:
                #assert len(data[key]["time"]) <= 2*time, "NICK SAYS: A time value given was too low, times should be at least half of data length"
                #assert len(exog[key]["time"]) >= time, "NICK SAYS: A time value given is out of range of exogenous variables"
                end = len(data[key]["time"])
                dct = {key: [] for key in range(time, int(end))}
                merged = pd.merge(data[key], exog[key], on='time', how='inner')
                merged2 = pd.merge(merged, global_exo, on = "time", how = "inner") \
                if global_exo is not None else merged
                
                merged2["month"] = (merged2["time"] % 12)
                merged2["year"] = merged2["time"] // 12

                if self.covid.get_include():
                    merged2["Covid"] = merged2["time"].apply(self.covid.get_function())

                merged2 = pd.get_dummies(merged2, columns=["month"], prefix="", prefix_sep="")
                merged2 = merged2.astype(int)

                 # Prepare data with lags
                x_data = merged2.drop([DeepLearnerPast.target], axis=1)
                y_data = merged2[DeepLearnerPast.target]
                # The Lag12 column represents the number of passengers 12 months prior
                x_data['Lag12'] = merged2[DeepLearnerPast.target].shift(12)

                
                # Divide the data according to training and testing data
                x_train = x_data[12:time]
                y_train = y_data[12:time]
                x_test = x_data[time:]
                y_test = y_data[time:]

                
                # Standardize the features
                #scaler = StandardScaler()
                #x_train = scaler.fit_transform(x_train)
                #x_test = scaler.transform(x_test)

                # Build the model
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Dense(self.LNE[1], input_dim=x_train.shape[1], activation='relu'))
                for i in range(self.LNE[0]):
                    model.add(tf.keras.layers.Dense(self.LNE[1], activation='relu'))
                model.add(tf.keras.layers.Dense(1))

                model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

                # Train the model
                model.fit(x_train, y_train, epochs=self.LNE[2], batch_size=32, validation_split=0.0, verbose=1)

                # Make predictions
                pred = model.predict(x_test).flatten()

                for index, value in enumerate(pred):
                        dct[index+time].append(value)

                 # Make report directory
                if self.path is not None:
                    if not os.path.exists(self.path + name + "/"):
                        os.makedirs(self.path + name + "/")               
                # Generates matplotlib graphs of all the data
                if self.matplot:
                    self.save_graphs(data[key], dct, time, list(key), self.path, self.weights, name, boole2)
                # Generates the excel file containing all the date (boole2 is True if range of values should be reported)
                if boole2:
                    self.save_numbers(dct, data[key], time, list(key), self.path, exog[key]["Total Seats"], name, self.weights)
                else:
                    self.save_numbers2(dct, data[key], time, list(key), self.path, exog[key]["Total Seats"], name, self.weights)

    def get_model_params(self, model, path, name, time, keys, number):
        super().get_model_params(model, path, name, time, keys, number)

    def save_graphs(self, x, dct, time, key, path, weights, name, boole2):
        super().save_graphs(x, dct, time, key, path, weights, name, boole2)
    
    def save_numbers(self, dct, x, time, key, path, seats, name, weights):
        super().save_numbers(dct, x, time, key, path, seats, name, weights)

    def save_numbers2(self, dct, x, time, key, path, seats, name, weights):
        super().save_numbers2(dct, x, time, key, path, seats, name, weights)

    def get_date(self, time):
        return super().get_date(time)