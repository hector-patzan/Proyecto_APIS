import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats


def getColsName(df):
    """ 
    Autores: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para obtener las columnas del dataset
    """
    cols_name =  df.columns.to_list()
    return cols_name


def getColumnsDataTypes(df):
    """ 
    Autores: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para obtener los tipos de variables de un dataframe
    """
    categoric_vars = []
    discrete_vars = []
    continues_vars = []

    for colname in df.columns:
        if(df[colname].dtype == 'object'):
            categoric_vars.append(colname)
        else:
            cantidad_valores = len(df[colname].value_counts())
            if(cantidad_valores<=30):
                discrete_vars.append(colname)
            else:
                continues_vars.append(colname)
    return categoric_vars, discrete_vars, continues_vars


def plotCategoricalVals(df, colums_name, y):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para analisis de variables categoricas
    """

    for column in colums_name:
        plt.figure(figsize=(12,6))
        plot = sns.countplot(x=df[column], hue=df[y])
        plt.title(column)
        plt.show()
        print(df[column].value_counts())


def plotDiscreteVals(df, colums_name, y):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para analisis de variables discretas
    """

    for column in colums_name:
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        sns.boxplot(x=df[y], y=df[column])
        plt.title(column)

        plt.subplot(122)
        sns.countplot(x=df[column], hue=df[y])
        plt.title(column)
        plt.show()
        print(df[column].value_counts())


def plotContinueVals(df, colums_name, y):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para analisis de variables discretas
    """

    for column in colums_name:
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        sns.scatterplot(data=df, x=column, y=y, hue=y)
        plt.title(column)

        plt.subplot(122)
        df[column].plot.density(color="red")
        plt.title(column)
        plt.show()


def getNanGoodCols(df, rate = 0.05):
    """ 
    Autor: Hector Patzan / Juan Pablo Castro
    Version: 1.0.0
    Descripción: Función para obtener columnas con campos para aplicar CCA
    """

    cols_procesables = []
    for col in df.columns:
        if((df[col].isnull().mean()<rate) & (df[col].isnull().mean()>0)):
            cols_procesables.append(col)
    return cols_procesables


def transformations (dataset, field, target):
    pd.options.mode.chained_assignment = None
    
    correlaciones = pd.DataFrame(columns=['transformacion', 'indice', 'corr'])
    base = dataset.copy()
    parametros = []
    for x in field:

        if (base[x].min()> 0):

            base[x + '_log'] = np.log(base[x])
            correlaciones.loc[0] = ['logaritmica', 0, np.corrcoef(base[x + '_log'], base[target])[0,1]]

            base[x + '_inv'] = (1 /(base[x]))
            correlaciones.loc[1] =['inversa', 1, np.corrcoef(base[x + '_inv'], base[target])[0,1]]

            base[x + '_quadratic'] = (base[x]**2)
            correlaciones.loc[2] = ['quadratic', 2, np.corrcoef(base[x + '_quadratic'], base[target])[0,1]]

            base[x + '_boxcox'], lambdaX = stats.boxcox(base[x])
            correlaciones.loc[3] = ['boxcox', 3, np.corrcoef(base[x + '_boxcox'], base[target])[0,1]]

            base[x + '_YJ'], lambdaX = stats.yeojohnson(base[x])
            correlaciones.loc[4]  = ['YJ', 4, np.corrcoef(base[x + '_YJ'], base[target])[0,1]]

            maxima = correlaciones[correlaciones['corr']==correlaciones['corr'].max()]
            maxima.reset_index(inplace=True)
            maxima['variable'] = x
            parametros.append(maxima[['transformacion', 'variable']].to_numpy().tolist())


    for x in parametros:
        operacion = x[0][0]
        campo = x[0][1]
        if (operacion=='boxcox'):
            dataset[campo], lambdaX = stats.boxcox(dataset[campo])

        elif (operacion=='logaritmica'):
            dataset[campo]  = np.log(dataset[campo])

        elif (operacion=='YJ'):
            dataset[campo], lambdaX = stats.yeojohnson(dataset[campo])

        elif (operacion=='quadratic'):
            dataset[campo]  = (dataset[campo]**2)

        elif (operacion=='inversa'):
            dataset[campo]  =  (1 /(dataset[campo]))
