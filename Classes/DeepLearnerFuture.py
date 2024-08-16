from .Future import Future
from .Helper import check_count, current_month
import pandas as pd
import os
import tensorflow as tf

"""# Defines Class DeepLearnerFuture which inherits from Past"""

# Class that functions are used when "Extrapolate" and "Long Term" selected in Settings.html
class DeepLearnerFuture(Future):
    # class variables 

    # class defintion
    # LNE: Array of integers. LNE is an acronym that stands for Layers-Neurons-Epochs
    def __init__(self, models, weights, matplot, params, path, filter, covid, LNE):
        self.LNE = LNE
        super().__init__(models, weights, matplot, params, path, filter, covid)


    def predict(self, time, data, exog, exog2, boole2, name, global_exo):
        assert(time <= 12), "NICK SAYS: Deep Learning can only be used to extrapolate \
                            a maximum of 12 months into the future."
        ind = -1 * Future.max_iter  # Max number of models to be trained (default 1000 from Predictor)
        dct = {}
        for key in data.keys():
            include = include = self.filter.filter_keys(key)
            if ind < 0 and data[key] is not None and key in exog.keys() and key in exog2.keys() and include:
                if check_count(data[key]) and check_count(exog[key]) \
                     and check_count(exog2[key], current_month,time):
                    dct = {key: [] for key in range(len(data[key]["time"]), len(data[key]["time"]) +  time)}
                    # Define dataframes so they have the appropriate features (exog/endog)
                    merged = pd.merge(data[key], exog[key], on='time', how='inner')
                    merged_past = pd.merge(merged, global_exo, on='time', how='inner') \
                        if global_exo is not None else merged
                    merged_future = pd.merge(exog2[key], global_exo, on='time', how='inner') \
                        if global_exo is not None else exog2[key]
                    # Merge past and future values on time and combine columbs representing the same quantity
                    merged2 = pd.merge(merged_past, merged_future, on='time', how='outer')
                    merged2['Total Seats'] = merged2['Total Seats_x'].fillna(merged2['Total Seats_y'])

                    # Handle global exogenous GDP data
                    if global_exo is not None:
                        merged2['GDP Percent Change'] = merged2['GDP Percent Change_x'].fillna(merged2['GDP Percent Change_y'])
                        merged2 = merged2.drop(columns=['GDP Percent Change_x', 'GDP Percent Change_y', 'Total Seats_x', 'Total Seats_y'])
                    else:
                        merged2 = merged2.drop(columns=['Total Seats_x', 'Total Seats_y'])

                    # Fill missing "Total Pax" values with 0
                    merged2['Total Pax'] = merged2['Total Pax'].fillna(0)

                    # include covid dummy variable
                    if self.covid.get_include():
                        merged2["Covid"] = merged2["time"].apply(self.covid.get_function())


                    # Feature Engineering: month, year, lags
                    merged2["month"] = (merged2["time"] % 12)
                    merged2["year"] = merged2["time"] // 12
                    merged2 = pd.get_dummies(merged2, columns=["month"], prefix="", prefix_sep="")
                    merged2 = merged2.astype(int)
                     # Prepare data with lags
                    merged2['Lag12'] = merged2[DeepLearnerFuture.target].shift(12)
                    
                    x_data = merged2.drop([DeepLearnerFuture.target], axis=1)
                    y_data = merged2[DeepLearnerFuture.target]

                    # Divide the data according to training and testing data
                    x_train = x_data[12:current_month]
                    y_train = y_data[12:current_month]
                    x_test = x_data[current_month:current_month+time]

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

                    # Add predicted values to dictionary
                    for index, value in enumerate(pred):
                            dct[index+current_month+1].append(value)

                    # Make report directory
                    if self.path is not None:
                        if not os.path.exists(self.path + name + "/"):
                            os.makedirs(self.path + name + "/")   

                    # Generates matplotlib graphs of all the data
                    if self.matplot:
                        self.save_graphs(data[key], dct, time, list(key), self.path, self.weights, name, boole2)

                    # Generates the excel file containing all the date (boole2 is True if range of values should be reported)
                    if boole2:
                        self.save_numbers(dct, data[key], time, list(key), self.path, exog2[key]["Total Seats"], name, self.weights)

                    else:
                        self.save_numbers2(dct, data[key], time, list(key), self.path, exog2[key]["Total Seats"], name, self.weights)


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