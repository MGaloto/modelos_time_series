
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


df_pred[start_date:end_date].plot(figsize = (20,5), color = "red")
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




















#%%






















#%%



























#%%























#%%
























#%%



























#%%

