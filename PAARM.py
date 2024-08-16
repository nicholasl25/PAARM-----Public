from flask import Flask, request, render_template
from Classes import Future, Past, LongPast, LongFuture, DeepLearnerPast, DeepLearnerFuture, Filter, Covid
from Classes.Helper import get_everything, GDP_Exog, GDP2_Exog, current_month, current_year, start_year
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import os
import ast
import platform
import warnings
import time
import signal
from threading import Thread
import webbrowser

# Schedule server shutdown in a separate thread
def shutdown_server():
    time.sleep(1)  # Wait for a moment before shutting down
    os.kill(os.getpid(), signal.SIGINT)

# Filter out warnings
warnings.filterwarnings("ignore", category=Warning)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Settings.html')

@app.route('/process', methods=['POST'])
def process():
    # Get form data
    long_term = ast.literal_eval(request.form.get('long_term', 'False'))
    deep_learner = request.form.get('deepLearner') == "True"
    report_name = request.form.get('report_name')
    isFuture = request.form.get('isFuture') == 'True'
    opt_pes = request.form.get('opt_pes') == 'True'
    matplot_graphs = request.form.get('matplot_graphs') == 'True'
    model_params = request.form.get('model_params') == 'True'
    # Handle Long Term Modeling Options
    if long_term:
        # Long Term processing
        exogenous = GDP2_Exog if request.form.get("GDP") == "True" else None
        dates = request.form.getlist('year[]')
        time_values = [int(year) - start_year for year in dates]
        covid_include = request.form.get("Covid") == "True"
        # Define covid dummy variable according to user input
        if covid_include:
            start_date = request.form.get('covidStart')
            start_int = int(start_date[:4]) - start_year 
            end_date = request.form.get('covidEnd')
            end_int = int(end_date[:4]) - start_year 
            My_Covid = Covid(covid_include, start_int, end_int)
        else:
            My_Covid = Covid(False, None, None)
    else:
        # Short Term processing
        exogenous = GDP_Exog if request.form.get("GDP") == "True" else None
        dates = request.form.getlist('month[]')
        time_values = [(int(date[:4]) - 2016) * 12 + int(date[5:]) - 1 for date in dates]
        covid_include = request.form.get("Covid") == "True"
        # Define covid dummy variable according to user input
        if covid_include:
            start_date = request.form.get('covidStart')
            start_int = (int(start_date[:4]) - 2016) * 12 + int(start_date[5:]) - 1
            end_date = request.form.get('covidEnd')
            end_int = (int(end_date[:4]) - 2016) * 12 + int(end_date[5:]) - 1
            My_Covid = Covid(covid_include, start_int, end_int)
        else:
            My_Covid = Covid(False, None, None)

    # Handle Deep Learning Model Options
    if deep_learner:
        selected_models = [None]
        weights = [1]
        layers =  int(request.form.get('numLayers'))
        neurons = int(request.form.get('neuronsPerLayer'))
        epochs = int(request.form.get('epochs'))
        LNE = [layers, neurons, epochs]
    else:
        # Define models
        ST_models = {
            "mod1": lambda x, exo: SARIMAX(x, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exo),
            "mod2": lambda x, exo: SARIMAX(x, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), exog=exo),
            "mod3": lambda x, exo: SARIMAX(x, order=(0, 1, 0), seasonal_order=(0, 1, 0, 12), exog=exo),
            "mod4": lambda x, exo: SARIMAX(x, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12), exog=exo),
            "mod5": lambda x, exo: SARIMAX(x, order=(2, 1, 1), seasonal_order=(2, 1, 1, 12), exog=exo),
            "mod6": lambda x, exo: SARIMAX(x, order=(1, 1, 2), seasonal_order=(1, 1, 2, 12), exog=exo),
        }

        LT_models = {
            "mod1": lambda x, exo: ARIMA(x, order=(1, 1, 1), exog=exo),
            "mod2": lambda x, exo: ARIMA(x, order=(0, 1, 1),  exog=exo),
            "mod3": lambda x, exo: ARIMA(x, order=(0, 1, 0), exog=exo),
            "mod4": lambda x, exo: ARIMA(x, order=(1, 1, 0), exog=exo),
            "mod5": lambda x, exo: ARIMA(x, order=(2, 1, 1),  exog=exo),
            "mod6": lambda x, exo: ARIMA(x, order=(1, 1, 2),  exog=exo),
        }

        ARCH_models = {
            "modA": lambda y, exo: arch_model(y, vol='Garch', p=1, q=1)
        }

         # Get selected models from form with weights
        selected_models = []
        weights = []
        
        # Define appropriate model dictionary
        models_dict = LT_models if long_term else ST_models
        # Add models to the ensemble
        for key in request.form.getlist('models'):
            if key in models_dict:
                selected_models.append(models_dict[key])
                weight_key = f'weight_{key}'
                weight_value = int(request.form.get(weight_key, 1))  # Default weight is 1 if not specified
                weights.append(weight_value)

    valid_types = request.form.getlist("domint")
    valid_airports = request.form.getlist("airports")
    valid_airlines = request.form.get('divide by airline') == 'True'
    if platform.system() == 'Darwin':
        path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/Reports/" 
    else:
        path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")[2:] + "/Reports/"  
   

    try:
        # Handle improper inputs with assertion error message
        difference = current_year - start_year if long_term else current_month
        if isFuture:
            assert difference < min(time_values), "NICK SAYS: All times selected must be in future for extrapolating."
            for i in range(len(time_values)):
                time_values[i] = int(time_values[i] - difference)
        else:
            assert difference >= max(time_values), "NICK SAYS: All times selected must be in past for backtesting."
        
        # My_Filter is of class Filter defines what datasets should be included in the report by
        # type, airport, and airline
        My_Filter = Filter(valid_types, valid_airports, valid_airlines)

        # My_Ensemble inherits from class Predictor and defines: what models/weights should be used, 
        # should matplot_graphs and model_params be inluded, and where should the report be saved to
        # My_Ensemble is of class Future if isFuture is True and of class Past otherwise
        if isFuture and long_term:
            My_Ensemble = LongFuture(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid)
        elif isFuture:
            if deep_learner:
                My_Ensemble = DeepLearnerFuture(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid, LNE)
            else:
                My_Ensemble = Future(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid)
        elif long_term:
            My_Ensemble = LongPast(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid)
        else:
            if deep_learner:
                My_Ensemble = DeepLearnerPast(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid, LNE)
            else:
                My_Ensemble = Past(selected_models, weights, matplot_graphs, model_params, path, My_Filter, My_Covid)

        # Returns the final report and exports it to the /Reports folder in the PAARM project
        get_everything(My_Ensemble, time_values, isFuture, opt_pes, report_name, exogenous, long_term)

        if not os.path.isdir(path + report_name):
            raise AssertionError("NICK SAYS: Your report could not successfully run... \
            there may not be enough data to train any models on the airports you submitted. \n" \
            "Make sure you are not leaving airports and dom/int blank!")

        return "Complete."

    except Exception as e:
        Thread(target=shutdown_server).start()
        get_everything(My_Ensemble, time_values, isFuture, opt_pes, report_name, exogenous, long_term)
        return str(e)
    
if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)