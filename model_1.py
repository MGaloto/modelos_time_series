
# Predecir vs pronosticar:
    
# Pronosticar: Que va a pasar con el futuro
# Predecir: Lo usaremos para datos que ya conocemos


#%%

# Importacion de Librerias


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
!pip install pmdarima
from pmdarima.arima import auto_arima
!pip install arch
from arch import arch_model
!pip install yfinance
import yfinance
import warnings
warnings.filterwarnings("ignore")
sns.set()



#%%


# Cargamos los datos desde yahoo finance


raw_data = yfinance.download (tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", 
                              end = "2019-09-01", interval = "1d", group_by = 'ticker', auto_adjust = True, treads = True)



# Hacemos una copia del data set

df_comp = raw_data.copy()


df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]





#%%


# Eliminamos los valores que no nos interesan


df_comp = df_comp.iloc[1:]
del df_comp['^N225']
del df_comp['^GSPC']
del df_comp['^GDAXI']
del df_comp['^FTSE']
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')




#%%


# Creamos los retornos

df_comp['ret_spx'] = df_comp.spx.pct_change(1).mul(100)
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1).mul(100)
df_comp['ret_dax'] = df_comp.dax.pct_change(1).mul(100)
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1).mul(100)



df_comp['norm_ret_spx'] = df_comp.ret_spx.div(df_comp.ret_spx[1])*100
df_comp['norm_ret_ftse'] = df_comp.ret_ftse.div(df_comp.ret_ftse[1])*100
df_comp['norm_ret_dax'] = df_comp.ret_dax.div(df_comp.ret_dax[1])*100
df_comp['norm_ret_nikkei'] = df_comp.ret_nikkei.div(df_comp.ret_nikkei[1])*100





#%%

# Datos de entrenamiento y de prueba

size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]




#%%

# Utilizaremos el modelo AR(1)


model_ar = ARIMA(df.ftse, order = (1,0,0))
results_ar = model_ar.fit()



#%%


# Para pronosticar hay que encontrar el patron del pasado y extenderlo al futuro


# Vemos cuando termina el data set de los datos de entrenamiento

df.tail()


# Fechas de periodo de pronostico, arranca de la fecha que no tenemos valores, despues del final del conjunto de datos de entrenamiento ( dia habil por ser cotizaciones bursatiles )

start_date = "2014-07-16"

# La fecha de finalizacion (cuanto mas largo el periodo mas dificil de ver los comportamientos)

end_date = "2015-01-01"



# Creamos el comando predict con los resultados del modelo

df_pred = results_ar.predict(start = start_date, end = end_date)



# Vemos una linea recta con pendiente negativa. Una estimacion que no coincide con las tendencias reales. El modelo simple AR(1) no es significativo para estas estimaciones


df_pred[start_date:end_date].plot(figsize = (20,5), color = "blue")
plt.title("Predicciones", size = 24)
plt.show()





#%%


# Cambiamos el modelo a una fecha mas lejana

# Vemos que el modelo AR no es muy bueno para explicar datos no estacionarios

end_date = "2019-10-23"
df_pred = results_ar.predict(start = start_date, end = end_date)


df_pred[start_date:end_date].plot(figsize = (20,5), color = "blue")
plt.title("Predicciones", size = 24)
plt.show()



#%%

# Prediiccion vs datos reales

# Podemos ver claramente las malas predicciones del modelo AR para datos no estacionarios

df_pred[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ftse[start_date:end_date].plot(color = "blue")
plt.title("Predicciones vs Actual", size = 24)
plt.show()



#%%

# Probando un modelo MA


end_date = "2015-01-01"

# Ponemos a partir de la segunda linea:

    
# El modelo pronostica que los valores son cercanos a cero

# Si aumentamos el orden tampoco es significativo

model_ret_ma = ARIMA(df.ret_ftse[1:], order=(0,0,1))
results_ret_ma = model_ret_ma.fit()

df_pred_ma = results_ret_ma.predict(start = start_date, end = end_date) 

df_pred_ma[start_date:end_date].plot(figsize = (20,5), color = "red")   
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predicciones vs Actual (Retornos)", size = 24)
plt.show()


# Los coeficientes no son significativos

for i in range(len(results_ret_ma.pvalues)):
    if i >= 0.05:
        print('Hay por lo menos un coeficiente no significativo')
        
    



#%%


# Estimando un modelo ARMA

# El patron es similar al anterior ya que seguimos trabajando con datos no estacionarios.

model_ret_arma = ARIMA(df.ret_ftse[1:], order=(1,0,1))
results_ret_arma = model_ret_arma.fit()

df_pred_arma = results_ret_arma.predict(start = start_date, end = end_date)

df_pred_arma[start_date:end_date].plot(figsize = (20,5), color = "red")   
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actual (Returns)", size = 24)
plt.show()


df_pred_arma.head()


df_pred_arma.tail()



#%%


# Probando un modelo ARMAX, sin la parte de integracion y variables exogenas.

# Estos modelos no son convenientes para proyectar el futuro ya que usa variables exogenas 


model_ret_armax = ARIMA(df.ret_ftse[1:], exog = df[["ret_spx","ret_dax","ret_nikkei"]][1:], order = (1,0,1))


results_ret_armax = model_ret_armax.fit()

start_date = "2014-07-16"


end_date = "2015-01-01"


df_pred_armax = results_ret_armax.predict(start = start_date, end = end_date, 
                                          exog = df_test[["ret_spx","ret_dax","ret_nikkei"]][start_date:end_date]) 

df_pred_armax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Prediccion vs Actual (Returns)", size = 24)
plt.show()





#%%




# Utilizando modelos estacionales SARIMAX



end_date = "2015-01-01"


# Los modelos estacionales van a tener 4 ordenes mas , los primeros son los estacionales y el ultimo es la duracion del ciclo estacional

# En el ultimo ponemos 5 ya que estudiamos la semana y esos son los dias habiles 


model_ret_sarma = SARIMAX(df.ret_ftse[1:], order = (3,0,4), seasonal_order = (3,0,2,5))

results_ret_sarma = model_ret_sarma.fit()

df_pred_sarma = results_ret_sarma.predict(start = start_date, end = end_date)


# En los resultados vemos valores muy cercanos a cero 


df_pred_sarma[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actual (SARMA)", size = 24)
plt.show()






#%%


# Introduciendo variables exogenas al modelo estacional SARIMAX


end_date = "2015-01-01"


# Ponemos las variables exogenas en el ajuste y la prediccion:

model_ret_sarimax = SARIMAX(df.ret_ftse[1:], exog = df[["ret_spx","ret_dax","ret_nikkei"]][1:], 
                            order = (3,0,4), seasonal_order = (3,0,2,5))
results_ret_sarimax = model_ret_sarimax.fit()

df_pred_sarimax = results_ret_sarimax.predict(start = start_date, end = end_date, 
                                              exog = df_test[["ret_spx","ret_dax","ret_nikkei"]][start_date:end_date]) 

df_pred_sarimax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Prediccion vs Actual (SARIMAX)", size = 24)
plt.show()







#%%

# Utilizando el AUTO ARIMA para ver cual es el mejor modelo




model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:],
                       m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5)


df_auto_pred = pd.DataFrame(model_auto.predict(n_periods = len(df_test[start_date:end_date]),
                            exogenous = df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date]),
                            index = df_test[start_date:end_date].index)





#%%


# La mayoria de los modelos no estaban integrados, por lo tanto, las previsiones de precios fueron peores que las de retornos.


# Grafico con todos los modelos estimados

end_date = "2015-01-01"
df_pred_ar[start_date:end_date].plot(figsize = (20,10), color = "yellow")
df_pred_ma[start_date:end_date].plot(color = "pink")
df_pred_arma[start_date:end_date].plot(color = "cyan")
df_pred_armax[start_date:end_date].plot(color = "green")
df_pred_sarma[start_date:end_date].plot(color = "magenta")
df_pred_sarimax[start_date:end_date].plot(color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.legend(['AR','MA','ARMA','ARMAX','SARMA','SARMAX'])
plt.title("Todos los modelos", size = 24)
plt.show()





#%%


# Prediccion de la volatilidad


# Estos son los valores predeterminados que tiene el comando arch_model.

mod_garch = arch_model(df_comp.ret_ftse[1:], vol = "GARCH", p = 1, q = 1, mean = "constant", dist = "Normal")


# En lasto obs le ponemos la fecha de comienzo, esto nos asegurara que solo estemos ajustando conjunto de entrenamiento en el modelo.

res_garch = mod_garch.fit(last_obs = start_date, update_freq = 10)



# En  este comando haremos el pronostico. El horizonte es 1 ya que es diario. Align determina si queremos hacer coincidir el valor con la fecha para la que se hace la prediccion o con la que se supone que representa


pred_garch = res_garch.forecast(horizon = 1, align = 'target')



# Con zorder ponemos uno arriba del otro

pred_garch.residual_variance[start_date:].plot(figsize = (20,5), color = "red", zorder = 2)
df_test.ret_ftse.abs().plot(color = "blue", zorder = 1)
plt.title("Volatilidad Predicciones", size = 24)
plt.show()


# Hasta ahora se han realizado predicciones dentro del conjunto de pruebas en vez de pronosticar el futuro. 





#%%


# Modelo de regresion multivariado:
    
# Se llaman modelos VAR = Vectoriales

# Vectorial Autoregresivo


from statsmodels.tsa.api import VAR


# Creamos el data set de las cuatro series temporales juntas

df_ret = df[['ret_spx', 'ret_dax', 'ret_ftse', 'ret_nikkei']][1:]


# Le pasamos el data set a la funcion VAR

model_var_ret = VAR(df_ret)

# El modelo seleccione un orden especifico, el valor numerico va a ser el orden maximo del modelo, cuantas mas series de tiempo mayor sera este valor

model_var_ret.select_order(20)

# El modelo se ajustara bajo AIC

results_var_ret = model_var_ret.fit(ic = 'aic')

# Al final del summary podemos ver una matriz de correlaciones

results_var_ret.summary()


#%%



# El parametro lag order ret sera una variable que contiene el orden de retraso del modelo que deriva del resultado del modelo anterior (5)

# El metodo forecast se encarga de los pronosticos


lag_order_ret = results_var_ret.k_ar

var_pred_ret = results_var_ret.forecast(df_ret.values[-lag_order_ret:], len(df_test[start_date:end_date]))

df_ret_pred = pd.DataFrame(data = var_pred_ret, index = df_test[start_date:end_date].index,
                                columns = df_test[start_date:end_date].columns[4:8])



df_ret_pred.ret_nikkei[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_nikkei[start_date:end_date].plot(color = "blue")
plt.title("Real vs Prediccion", size = 24)
plt.show()




results_var_ret.plot_forecast(1000)
plt.show()


#%%






















#%%





















#%%


















#%%


















#%%


















#%%


















#%%





















#%%


















#%%


















#%%


















#%%





















#%%


















#%%

