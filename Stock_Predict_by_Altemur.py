from pandas_datareader import DataReader
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import threading
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Stock_Predictor:
    def __init__(self, stock_name, start_date, end_date, day_to_forecast, tester_mode = True):
        '''
        stock_name = 'StockID of your stock'
        start date = 'YYYY-MM-DD'
        end_date = 'YYYY-MM-DD'
        day_to_forecast = day as integer to forecast
        tester_mode = True/False (Tests models accuracy by splitteng date to train-test)
        '''
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self.day_to_forecast = day_to_forecast
        self.tester_mode = tester_mode


    def stock_data(self):
        '''
        Reads data and convert column name <Close> to <y>(format for Neural Network)
        returns train and test or only train according to self.tester_mode
        '''
        df = DataReader(self.stock_name, data_source= 'yahoo', start = self.start_date, end = self.end_date)
        df.rename(columns={'Close': 'y'}, inplace=True, errors='raise')
        df['ds'] = df.index

        if self.tester_mode == True:
            train = df[['ds','y']][:-self.day_to_forecast]
            test = df[['ds','y']][-self.day_to_forecast:]

            return train, test

        else:
            train = df[['ds','y']]

            return train

        

    def prediction(self, epochs = 200):
        '''
        Returns prediction and future date_time of data
        accepts epoch number as input but the default value is epochs = 200
        '''
        m = NeuralProphet(daily_seasonality = True,
                 weekly_seasonality = False,
                 yearly_seasonality = False,
                 seasonality_mode='multiplicative',
                 learning_rate=0.01,
                 batch_size = 12,
                 n_forecasts = self.day_to_forecast,
                 n_lags = 7,
                 )
        if self.tester_mode == True:         
            train, _ = Stock_Predictor.stock_data(self)
        else:
            train, _ = Stock_Predictor.stock_data(self)
        m.fit(train, freq="D", epochs = epochs)
        future = m.make_future_dataframe(train, periods = self.day_to_forecast)
        forecast_n = m.predict(future, decompose = True, raw = True)
        y_pred = forecast_n.iloc[-1,1:self.day_to_forecast + 1].values 
        df_datetime = future.iloc[-self.day_to_forecast:,0].dt.date.values
        return y_pred, df_datetime

    
    def mape(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mse(y_true, y_pred):
        return np.square(np.subtract(y_true, y_pred)).mean()

    def mae(y_true, y_pred):
        return np.sum(np.absolute((y_true - y_pred)))
    
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],y[i])
            
    
    def plot_2d(self,  y_pred, df_datetime, y_true = None):
        '''
        Creates 2d plot automaticly of output
        '''
        _x = np.arange(len(df_datetime))
        plt.style.use(['dark_background'])
        plt.figure(figsize=(20,8))

        plt.plot(y_pred, 'm--*', linewidth = 3.0)

        if self.tester_mode == True:
            plt.plot(y_true, 'b-o', linewidth = 4.5)
            plt.ylim([min(np.append(np.array(y_pred),np.array(y_true)))*0.8,
                      max(np.append(np.array(y_pred),np.array(y_true)))*1.1])
            title = 'TRUE: MAPE: %.2f / MSE: %.2f / Total Sum Error: %.2f'\
                    %(
                    Stock_Predictor.mape(y_true, y_pred), Stock_Predictor.mse(y_true, y_pred),
                    100*((sum(y_true)-sum(y_pred))/sum(y_true))
                    )
            plt.title(title, fontweight='bold')
        
        else:

            plt.ylim([min(np.array(y_pred))*0.8, 
                      max(np.array(y_pred))*1.2])
        plt.xlabel('Day')
        plt.xticks(_x, [str(i) for i in df_datetime])
        plt.tick_params(axis = 'x', rotation = 65)
        plt.ylabel('Amount Sold / Day')
        plt.grid(color='w', linestyle=':', linewidth=0.5, alpha = 0.7)
        plt.legend(('Predicted','Actual','Upper','Lower'))

        plt.show()
    
    def plot_error(y_pred, df_datetime, y_true):
        '''
        Creates error table of true value and prediction.
        Warning!: only works with true and predicted values 
        '''
        _x = np.arange(len(df_datetime))
        plt.style.use(['dark_background'])
        fig, ([fig1, fig2], [fig3, fig4]) = plt.subplots(2, 2, figsize=(20,8))

        width = 0.45
        title = 'MAPE: %.2f, MSE: %.2f, MAE: %.2f Total Sum Error: %.2f'%(Stock_Predictor.mape(y_true, y_pred),
                                                                          Stock_Predictor.mse(y_true, y_pred), 
                                                                          Stock_Predictor.mae(y_true, y_pred), 
                                                                          100*((sum(y_true)-sum(y_pred))/sum(y_true)))
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        my_cmap = cm.get_cmap('jet')
        my_norm = Normalize(vmin=0, vmax = len(y_pred))


        value = [Stock_Predictor.mape(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        MAPE_val = [str('%.1f'%(i)) for i in value]
        fig1.bar(MAPE_val, value, color = my_cmap(my_norm(np.arange(len(y_pred)))))
        fig1.set_title('MAPE')
        fig1.tick_params(axis='x', rotation = 90)


        value2= [Stock_Predictor.mse(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        MSE_val = [str('%.1f'%(i)) for i in value2]
        fig2.bar(MSE_val, value2, color = my_cmap(my_norm(np.arange(len(y_pred)))))
        fig2.tick_params(axis='x', rotation = 90)
        fig2.set_title('MSE')

        value3= [Stock_Predictor.mae(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        MAE_val = [str('%.1f'%(i)) for i in value3]
        fig3.bar(MAE_val, value3, color = my_cmap(my_norm(np.arange(len(y_pred)))))
        fig3.tick_params(axis='x', rotation = 90)
        fig3.set_title('MAE')

        fig4.bar(np.arange(len(df_datetime)) + 1 - width/2 , y_true, width)
        fig4.bar(np.arange(len(df_datetime)) + 1 + width/2, y_pred, width)
        fig4.legend(['Real', 'Predicted'])
        fig4.set_title(f'Real vs Predicted Values Between {df_datetime[0]}/{df_datetime[-1]}')

        plt.show()

    def easy_mode(stock_name, day):
        '''
        Creates prediction easily, only accepts stock_name and day to forecast
        '''
        from datetime import date
        stock = Stock_Predictor(stock_name = stock_name,
                        start_date = '2013-01-01',
                        end_date = str(date.today())[:-2] + str(int(str(date.today())[-2:])-1),
                        day_to_forecast = day,
                        tester_mode = False)
                    
        pred, df_datetime = stock.prediction(epochs = 200)
        threading.Thread(target = Stock_Predictor.plot_2d(stock, pred, df_datetime))

        return pred

    def easy_mode_model_test(stock_name, day):
        '''
        Tests models easily, only accepts stock_name and day to forecast
        '''
        from datetime import date
        stock = Stock_Predictor(stock_name = stock_name,
                        start_date = '2013-01-01',
                        end_date = str(date.today())[:-2] + str(int(str(date.today())[-2:])-1),
                        day_to_forecast = day,
                        tester_mode = True)
        _, test = stock.stock_data()    
        true = test.y.values
        pred, df_datetime = stock.prediction(epochs = 200)
        threading.Thread(target = Stock_Predictor.plot_2d(stock, pred, df_datetime, true))
        threading.Thread(target = Stock_Predictor.plot_error(pred, df_datetime, true))
        return pred



