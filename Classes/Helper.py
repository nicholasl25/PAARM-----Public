# Import relevant packages
import pandas as pd
import numpy as np
import subprocess
import pkg_resources
from itertools import product


"""## All Helper Functions"""

"""### Helper Function: get_everything"""

# Returns the entire report and exports report to filepath
# ensemble: object of parent class Predictor containing all models, weights, and report settings
# lst: list of ints that represent time being predicted into future
# boole: boolean that is True if the model is predicting into the future and
# False if the report is backtesting
# boole2: boolean that is True if optimistic/pessimistic values should be included
# name: string name of report to be saved to folder
# global_exo: all global exogenous variables to be included in the models
# long_term: boolean that is True if prediction is being made for the yearly/long term
# and False if it is for the monthly/short term

def get_everything(ensemble, lst, boole, boole2, name, global_exo, long_term):
  assert isinstance(boole, bool), "NICK SAYS: isFuture must be a boolean (True/False)"
  assert isinstance(boole2, bool),  "NICK SAYS: optpes must be a boolean (True/False)"
  assert isinstance(long_term, bool),  "NICK SAYS: long_term must be a boolean (True/False)"
  assert isinstance(name, str), "NICK SAYS: name must be a string: try surrounding name by quotes"
  assert isinstance(name, str), "NICK SAYS: reoprt_name must be a string: try surrounding name by quotes"
  if not isinstance(lst, list):
    assert isinstance(lst, int), "NICK SAYS: time_values must be a list of numbers (ints)"
    lst = [lst]
  for l in lst:
    assert isinstance(l, int), "NICK SAYS: times_values must be a list of numbers (ints)"
    if long_term:
      ensemble.predict(l, Yearly_pax_dict, None, None, boole2, name, global_exo)
    else:
      ensemble.predict(l, Pax_dict, Past_seats_dict, Future_seats_dict, boole2, name, global_exo)

    
"""### Ensemble Class Definition (Helper Functions: predict_past + predict_future)"""

target = "Total Pax" # target variable to be predicted

#models = set of models to be used in predictions
#weights = array of numbers that determine how each model should be weighted in final average
#matplot = boolean that is True if matplotlib graphs should be exported and False otherwise
#params = boolean that is True if model parameters should be exported for every model trained
#path = string that contains the filepath the report should be saved into
#filter = object of class Filter that decides what keys(datasets) should be included in the report

exog_targets = ["Total Seats", "GDP Percent Change", "time"] #can be changed depending on what exogenous variables should be included
include_covid = True # boolean that is True if Covid should be included as a dummy variable and False otherwise 
                      # currently not user inputted

"""### Helper Function: standard_dev"""

# Returns the weighted standard devaiation of an array as a float
# values: the array of values
# weights: the array of weights for each value

def standard_dev(values, weights):
    assert len(values) == len(weights), "NICK SAYS: The number of values must equal the number of weights"
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)

"""### Helper Function: get_title"""

# Returns a title as string given a list of keys denoting type/airport/airline/(exog)
# keys: Array that is 3-4 keys long that identifies what type, airport, airline, and exogenous
# variables are being exported

def get_title(keys):
  assert isinstance(keys, list) and len(keys) > 2, "NICK SAYS: The number of keys must be at least 3"

  title = ""
  if keys[0] != "Subtotal":
    title += "Type: " + keys[0] + " + "
  if keys[1] != "Subtotal":
    title += "Airport: " + keys[1] + " + "
  if keys[2] != "Subtotal":
    title += "Airline: " + keys[2] + " + "
  if len(keys) == 4:
    title += " " + keys[3] + " + "
  return title[:-2]

"### Helper Function: get_lf"
# Returns the ratio of pax to seats values as long as pax/seats are not missing values
# if pax and seats are missing values NaN/None get_lf returns None
# pax: series representing passenger volumes as an int
# seats: series representing seats as an int

def get_lf(pax, seats):
    if pax.isna().any() or seats.isna().any() or (seats == 0).any():
        return None
    return (pax / seats * 100).round(1).astype(str) + '%'

"""### Helper Function: get_dict"""

# Returns a dictionary which has keys of all possible combinations of types/airports/airlines
# and values of the corresponding dataframe or None
# df: the pandas dataframe that is being divided into sub dataframes
# target: the target variable in the dataframe (usally "Total Pax" or "Total Seats")

#valid_airlines = ["American Airlines", "Delta Air Lines", "JetBlue Airways",  "United Airlines"]

def get_dict(df, target):
  airlines = list(df['Airline'].unique()) + ["Subtotal"]
  types = list(df['Type'].unique()) + ["Subtotal"]
  airports = list(df['Airport'].unique()) + ["Subtotal"]

  combos = list(product(types, airports, airlines))
  table = {combination: None for combination in combos}


  #Division by type, airport, and airline
  df_type = df.groupby("Type")
  for typ, frame in df_type:
    df_type_airport = frame.groupby("Airport")
    for airport, frame2 in df_type_airport:
      df_type_airport_airline = frame2.groupby("Airline")
      for airline, frame3 in df_type_airport_airline:
        table[(typ, airport, airline)] = frame3.sort_values(by=['time'])

  #Division by airport and airline, subtotal over type
  df2 = df.groupby(['Airport', 'Airline', "time"])[target].sum().reset_index()
  df2_airport = df2.groupby("Airport")
  for airport, frame in df2_airport:
    df2_aiport_airline = frame.groupby("Airline")
    for airline, frame2 in df2_aiport_airline:
      table[("Subtotal", airport, airline)] = frame2.sort_values(by=['time'])

  #Division by type, airport, subtotal over all airlines
  df3 = df.groupby(['Airport', 'Type', "time"])[target].sum().reset_index()
  df3_type = df3.groupby("Type")
  for typ, frame in df3_type:
    df3_type_airport = frame.groupby("Airport")
    for airport, frame2 in df3_type_airport:
      table[(typ, airport, "Subtotal")] = frame2.sort_values(by=['time'])

  #Division by type, airline, subtotal over all airports
  df4 = df.groupby(['Airline', 'Type', "time"])[target].sum().reset_index()
  df4_type = df4.groupby("Type")
  for typ, frame in df4_type:
    df4_type_airline = frame.groupby("Airline")
    for airline, frame2 in df4_type_airline:
      table[(typ, "Subtotal", airline)] = frame2.sort_values(by=['time'])


  #Division by airport subtotal over type and airline
  df5 = df.groupby(['Airport', "time"])[target].sum().reset_index()
  df5_airport = df5.groupby("Airport")
  for airport, frame in df5_airport:
    table[("Subtotal", airport, "Subtotal")] = frame.sort_values(by=['time'])

  #Division by airline subtotal over type and airport
  df6 = df.groupby(['Airline', "time"])[target].sum().reset_index()
  df6_airline = df6.groupby("Airline")
  for airline, frame in df6_airline:
    table[("Subtotal", "Subtotal", airline)] = frame.sort_values(by=['time'])

  #Division by type subtotal over airline and airport
  df7 = df.groupby(['Type', "time"])[target].sum().reset_index()
  df7_type = df7.groupby("Type")
  for typ, frame in df7_type:
    table[(typ, "Subtotal", "Subtotal")] = frame.sort_values(by=['time'])

  for key in table.keys():
    if table[key] is not None and target in table[key].columns and "time" in table[key].columns:
      table[key] = table[key][[target, "time"]]

  return table

"""### Helper Function: check_count"""

# Returns a boolean that is True if df is a pandas dataframe
# that has no missing data and False otherwise (ensures Model will train properly)
# df = object that is being tested
# current = the time value as an int which the df should have data up until
# time = is None if df is part of training data, else is an int that
# represents the amount of months the model should predict

def check_count(df, current = None, time = None):
  if time is None and current is None:
    return (False if not isinstance(df, pd.DataFrame) else (len(df["time"]) == df["time"].max() + 1))
  elif time is None:
    return (False if not isinstance(df, pd.DataFrame) else (len(df["time"]) == current + 1))
  else:
    return (False if not isinstance(df, pd.DataFrame) else (len(df["time"]) >= time))

"""### Helper Functions: get_pax"""

# Helper function that replaces NA values with 0 in pandas dataframe row

# Returns an int (representing the target value for that row)
# row: row of pandas dataframe -- must have column [month]
# month: string representing the desired month in row
def get_pax(month, row):
    if pd.isna(row[month]):
        return 0
    else:
        return row[month]

"""## Data Preprocessing

### Pax Data Preproccesing
"""


# Dictionary to map month names to zero-indexed numbers
month_name = {
    'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5,
    'July': 6, 'August': 7, 'September': 8, 'October': 9, 'November': 10, 'December': 11 }

# Dictionary that converts the airline name in the pax csv to the airline name in the seats cvs
pax_to_seats_dict = {
    "UNITED": "United Airlines",
    "AMERICAN": "American Airlines",
    "DELTA": "Delta Air Lines"
}

# Isolate only Port Authority Airports
valid_airports = ["EWR", "JFK", "LGA", "SWF"]


try:
  all_pax = pd.read_csv("Datasets/Monthly Pax.csv")
  all_pax = all_pax.iloc[8:]
  new_column_names = all_pax.iloc[0]
  all_pax = all_pax.dropna(how='all')
  all_pax = all_pax[1:]
  all_pax = all_pax.drop(all_pax.index[-1])
  all_pax.columns = new_column_names
  all_pax = all_pax.melt(id_vars=["Year", "Airport Code", "Marketing Name", "Market Type Description"], var_name="Month", value_name="Total Pax")
  all_pax = all_pax.rename(columns={
      'Airport Code': 'Airport',
      'Marketing Name': 'Airline',
      'Market Type Description': 'Type'
  })
  all_pax["Month"] = all_pax["Month"].map(month_name)
  all_pax["time"] = 12 * (all_pax["Year"].astype(int) - 2016) + all_pax["Month"]
  all_pax = all_pax[all_pax['time'] >= 0]
  #all_pax['Total Pax'] = all_pax['Total Pax'].replace(r'[\s-]', np.nan, regex=True)
  all_pax['Total Pax'] = all_pax['Total Pax'].str.replace(',', '').str.replace(' ', '').str.replace('-', '').replace('', 0)
  # Drop rows with missing values in 'Total Pax'
  all_pax.dropna(subset=['Total Pax'], inplace=True)
  # Now convert to integers
  all_pax['Total Pax'] = all_pax['Total Pax'].astype(int)
  all_pax = all_pax[all_pax["Airport"].isin(valid_airports)]
  all_pax['Airline'] = all_pax['Airline'].replace(pax_to_seats_dict)
  all_pax["Airline"] = all_pax["Airline"].str.upper()
  current_month = int(all_pax["time"].max())

  Pax_dict = get_dict(all_pax, "Total Pax")

except FileNotFoundError:
  text = "No file found\n" + \
  "NICK SAYS: Make sure your pax datafile is a csv named Monthly Pax.csv in the datasets folder"
  raise Exception(text)
except KeyError:
    text = "Dataframe is missing one or more required keys\n" + \
          "NICK SAYS: Make sure you downloaded your passenger data off the 2024 Pax Estimation Facility Report\n" + \
          "NICK SAYS: Make sure your data has all of the following features: Year, Airport Code, Marketing Name, Marketing Type Description, Month, and Total Pax"
    raise Exception(text)
except ValueError:
  text = "Can not convert value in Monthly Pax column to int\n" + \
  "NICK SAYS: Make sure you downloaded your passenger data off the Short Term Data Excel file"
  raise Exception(text)

"""### Seat Data Preprocessing"""

try:
  all_seats = pd.read_csv("Datasets/Monthly Seats.csv")
  all_seats = all_seats.iloc[8:]
  new_column_names = all_seats.iloc[0]
  all_seats = all_seats.dropna(how='all')
  all_seats = all_seats[1:]
  all_seats = all_seats.drop(all_seats.index[-1])
  all_seats.columns = new_column_names

  all_seats = all_seats.melt(id_vars=["Year", "PA Airport Code", "Marketing Airline Desc", "International Domestic"], var_name="Month", value_name="Total Seats")
  all_seats = all_seats.rename(columns={
      'PA Airport Code': 'Airport',
      'Marketing Airline Desc': 'Airline',
      'International Domestic': 'Type'
  })
  all_seats["Month"] = all_seats["Month"].map(month_name)
  all_seats["time"] = 12 * (all_seats["Year"].astype(int) - 2016) + all_seats["Month"]
  all_seats = all_seats[all_seats['time'] >= 0]
  all_seats['Total Seats'] = all_seats['Total Seats'].str.replace(',', '', regex=False)
  all_seats['Total Seats'] = pd.to_numeric(all_seats['Total Seats'], errors='coerce')
  all_seats['Total Seats'] = all_seats['Total Seats'].fillna(0)
  all_seats['Total Seats'] = all_seats['Total Seats'].astype(int)
  all_seats = all_seats[all_seats["Airport"].isin(valid_airports)]
  all_seats["Airline"] = all_seats["Airline"].str.upper()
  all_seats_past = all_seats[all_seats['time'] <= current_month]
  all_seats_future = all_seats[all_seats['time'] > current_month]
  #all_seats_future = all_seats_future[all_seats_future['time'] < future_month]

  Past_seats_dict = get_dict(all_seats_past, "Total Seats")
  Future_seats_dict = get_dict(all_seats_future, "Total Seats")

except FileNotFoundError:
  text = "No file found\n" + \
  "NICK SAYS: Make sure your seats datafile is a csv named Monthly Seats.csv in the datasets folder"
  raise Exception(text)
except KeyError:
    text = "Dataframe is missing one or more required keys\n" + \
          "NICK SAYS: Make sure you downloaded your seats data off the 2024 Pax Estimation Facility Report\n" + \
          "NICK SAYS: Make sure your data has all of the following features: Year, PA Airport Code, Marketing Airline Desc, International Domestic, Month, and Total Seats"
    raise Exception(text)
except ValueError:
  text = "Can not convert value in Monthly Seats column to int\n" + \
  "NICK SAYS: Make sure you downloaded your seats data off the Short Term Data Excel file"
  raise Exception(text)

""" ### GDP Data Preprocessing """

try: 
  # import datasets using pandas
  GDP = pd.read_csv("Datasets/Monthly Economic.csv")

  # Convert the date column to datetime format
  GDP['date'] = pd.to_datetime(GDP['Period'], format='%m/%d/%Y')

  # Add range of possible dates for entire dataframe
  date_range = pd.date_range(start=GDP['date'].min(), end=GDP['date'].max(), freq='MS')

  # Create a new dataframe with the complete range of months
  date_df = pd.DataFrame({'date': date_range})

  # Merge the original dataframe with the complete date range
  GDP = pd.merge(date_df, GDP, on='date', how='left')

  # Calculate the difference in months since 2016
  GDP['time'] = GDP['date'].apply(lambda x: (x.year - 2016) * 12 + x.month - 1)

  # Rename ' Value ' column to 'Real GDP' 
  GDP = GDP.rename(columns = {' Value ':'Real GDP'})

  # Interpolate the missing values
  GDP['Real GDP'] = GDP['Real GDP'].str.replace(',', '').astype(float)
  GDP['Real GDP'] = GDP['Real GDP'].interpolate(method = "linear")

  # Forward fill the other columns
  GDP[['Concept', 'Unit', 'Period']] = GDP[['Concept', 'Unit', 'Period']].ffill()

  # Add Column of GDP that represents the MoM GDP Percent Change in Real GDP
  GDP["GDP Percent Change"] = GDP['Real GDP'].pct_change() * 100
  GDP['GDP Percent Change'].replace([np.nan, np.inf, -np.inf], 0, inplace=True)

  # Define GDP so only correct factor (Percent Change) is considered
  GDP_Exog = GDP[['GDP Percent Change', "time"]]

except Exception:
   print("Exception raised while importing Monthly Economic Data. \
         \n GDP will not be used as a short term exogenous variable. \
         \n To incorperate GDP as an exogenous variable please have a csv titled Monthly Economic.csv in /datasets")
   
   # Set GDP_Exog to None so exception is not raised in PAARM.py
   GDP_Exog = None

"""###Load Factor Preprocessing"""
# Create the Load Factor dictionary
LF_dict = {}
for key in Pax_dict.keys():
    if key in Past_seats_dict and Pax_dict[key] is not None and Past_seats_dict[key] is not None:
        pax_df = Pax_dict[key]
        seats_df = Past_seats_dict[key] 
        
        # Merge the dataframes on 'time'
        merged_df = pd.merge(pax_df, seats_df, on='time')
        
        # Perform the division and create the result DataFrame
        result_df = merged_df.copy()
        result_df['Total Pax'] = 1000000 * (merged_df['Total Pax'] / merged_df['Total Seats'])
        
        # Add to LF_dict
        LF_dict[key] = result_df[['time', 'Total Pax']]


"""### Yearly Pax Data Prprocessing"""

try: 
  # Import yearly passenger data with same fields from Yearly Pax.csv
  yearly_pax = pd.read_csv("Datasets/Yearly Pax.csv")
  yearly_pax = yearly_pax.iloc[7:]

  # Drop NaN values from df
  yearly_pax = yearly_pax.dropna(how='all', axis = 1)
  yearly_pax = yearly_pax.dropna(how='all', axis = 0)

  # Seperate column names from values
  yearly_column_names = yearly_pax.iloc[0]
  yearly_pax = yearly_pax[1:]
  yearly_pax = yearly_pax.drop(yearly_pax.index[-1])
  yearly_pax.columns = yearly_column_names

  # Rename columns appropriately to match pax/seats datasets
  yearly_pax = yearly_pax.rename(columns={
      'Airport Code': 'Airport',
      'Marketing Name': 'Airline',
      'Market Type Description': 'Type',
      'Total Revenue Pax': 'Total Pax'})


  yearly_pax['Total Pax'] = yearly_pax['Total Pax'].str.replace(',', '').str.replace(' ', '').replace('', 0)
  # Drop rows with missing values in 'yearly Pax'
  yearly_pax = yearly_pax[~yearly_pax['Total Pax'].str.contains('-', na=False)]
  # Convert to integers
  yearly_pax['Total Pax'] = yearly_pax['Total Pax'].astype(int)
  yearly_pax['Airline'] = yearly_pax['Airline'].replace(pax_to_seats_dict)
  yearly_pax["Airline"] = yearly_pax["Airline"].str.upper()

  # Get the years which the datasets starts and ends
  start_year = int(yearly_pax["Year"].min())
  current_year = int(yearly_pax["Year"].max()) - 1

  # Limit dataframe to only values within this date range
  yearly_pax["Year"] = yearly_pax["Year"].astype(int)
  yearly_pax = yearly_pax[(yearly_pax["Year"] >= start_year) & (yearly_pax["Year"] <= current_year)]

  # Create new "time" column in dataframe
  yearly_pax["time"] = yearly_pax["Year"].astype(int) - start_year

  # Divide data using get_dict
  Yearly_pax_dict = get_dict(yearly_pax, "Total Pax")

except Exception:
   print("Exception raised while importing Long Term Seat Data. \
         \n Long Term forecasting not avaliable. \
         \n To enable Long Term forecasting import a csv named Yearly Pax in /datasets \
         \n from the Long Term Data file.")
   
   # Set long term variables to None so exception is not raised in PAARM.py
   Yearly_pax_dict = None
   start_year = None
   current_year = None
   

"""### Long Term GDP Data Preprocessing"""

try: 
  # Import GDP + Fuel Price Data as a .csv
  GDP2 = pd.read_csv("datasets/Yearly Economic.csv")

  # Create time column (starting at start_year)
  GDP2["time"] = GDP2["Year"] - start_year

  GDP2_Exog = GDP2[["time", "% Change GDP", "% Change Fuel"]]

except Exception:
  print("Exception raised while importing Long Term GDP Data. \
        \n GDP/Fuel Prices will not be used as long term exogenous variables. \
        \n To incorperate these variable please have a csv titled Yearly GDP.csv in /datasets")
  
  # Set GDP_Exog to None so exception is not raised in PAARM.py
  GDP2_Exog = None

"""# Install Needed Packages Locally"""

# Check if TensorFlow is installed, and install if not
try:
    pkg_resources.get_distribution('tensorflow')
except pkg_resources.DistributionNotFound:
    print("TensorFlow not found. Installing...")
    subprocess.check_call([
        'pip', 'install', 'tensorflow',
        '--trusted-host', 'pypi.org',
        '--trusted-host', 'files.pythonhosted.org'
    ])
    print("TensorFlow installed successfully.")

# Check if arch is installed, and install if not
try:
    pkg_resources.get_distribution('arch')
except pkg_resources.DistributionNotFound:
    print("arch not found. Installing...")
    subprocess.check_call([
        'pip', 'install', 'arch',
        '--trusted-host', 'pypi.org',
        '--trusted-host', 'files.pythonhosted.org'
    ])
    print("arch installed successfully.") 