import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.impute import KNNImputer
import scipy.stats as ss
import warnings
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, \
    silhouette_score, recall_score, precision_score, \
    roc_auc_score, f1_score, precision_recall_curve, accuracy_score, \
    mean_squared_error, r2_score, ConfusionMatrixDisplay
import scikitplot as skplt

from sklearn.inspection import PartialDependenceDisplay

def dame_variables_categoricas(dataset=None, max_unicos=50): 

    """ La función nos devuelve dos listas: una de variables categóricas y otra de no categóricas a partir de la introducción de un dataset.
    Se indica el valor de valores únicos máximo(max_unicos) como un parametro de manera que si lo escribimos a lo largo de la función
    más de una vez solo se tenga que cambiar en una línea. 
    
    Inputs:
        -- dataset: pandas dataframe que contiene los datos
        -- max_unicos: máximo de unicos que se desean por columna(variable)
    
    Returns:
        lista_variables_categoricas: lista de nombres de variables categóricas con menos de `threshold` valores únicos.
        other: lista de nombres de variables no categóricas o que exceden el umbral.
        """




    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1

    lista_variables_categoricas = []
    other = []

    for i in dataset.columns:
        unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
        if dataset[i].dtype in ['object', 'category'] or unicos < max_unicos: # A traves de la lista simplificamos el código.
                lista_variables_categoricas.append(i)
        else:
                other.append(i)

    return lista_variables_categoricas, other




def plot_feature(df, col_name, is_continuous, target, figsize=(12, 6), color="#5975A4"):
    """
    Visualiza una variable categórica o continua con y sin segmentación por una variable objetivo (target).
    
    Parámetros:
    df: dataFrame que contiene los datos.
    col_name: nombre de la columna a graficar.
    is_continuous: booleano. True si la variable es continua, False si es discreta.
    target: nombre de la variable objetivo para segmentar los datos.
    figsize: tamaño de la figura (opcional, por defecto (12, 6)).
    color: color principal de los gráficos (opcional).

    Outputs:
    Muestra los gráficos
    """
    # Configuración inicial del tamaño de la figura y subgráficos
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=90)

    # Calcular el número de valores nulos en la columna
    count_null = df[col_name].isnull().sum()
    null_text = f"Número de nulos: {count_null}" if count_null > 0 else "Sin valores nulos"

    # Gráfico 1: Visualización general sin segmentación
    if is_continuous:
        # Histograma para variables continuas
        sns.histplot(df[col_name].dropna(), kde=True, color=color, ax=ax1)
        ax1.set_title(f"Distribución de {col_name}\n({null_text})")
        ax1.set_xlabel(col_name)
        ax1.set_ylabel("Frecuencia")
    else:
        # Gráfico de conteo para variables discretas
        sns.countplot(
            x=col_name, 
            data=df, 
            order=df[col_name].value_counts().index, 
            color=color, 
            ax=ax1
        )
        ax1.set_title(f"Conteo de {col_name}\n({null_text})")
        ax1.set_xlabel(col_name)
        ax1.set_ylabel("Frecuencia")
        ax1.tick_params(axis="x", rotation=90)

    # Gráfico 2: Visualización segmentada por el target
    if is_continuous:
        # Boxplot para analizar la distribución por la variable objetivo
        sns.boxplot(x=target, y=col_name, data=df, ax=ax2, color=color)
        ax2.set_title(f"{col_name} por {target}")
        ax2.set_xlabel(target)
        ax2.set_ylabel(col_name)
    else:
        # Cálculo de proporciones por categoría y segmentación por target
        prop_data = (
            df.groupby(col_name)[target]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        sns.barplot(
            x=col_name, 
            y="proportion", 
            hue=target, 
            data=prop_data, 
            ax=ax2, 
            saturation=1
        )
        ax2.set_title(f"{col_name} segmentado por {target} (Proporción)")
        ax2.set_xlabel(col_name)
        ax2.set_ylabel("Proporción")
        ax2.tick_params(axis="x", rotation=90)

    # Ajuste de los gráficos
    plt.tight_layout()
    plt.show()


def iv_woe(df, variable, target, bins=None):
    """
    Calcula el Weight of Evidence (WoE) y el Information Value (IV) para una variable respecto a una variable objetivo binaria.

    Parámetros:
    df (DataFrame): DataFrame que contiene los datos.
    variable (str): nombre de la variable para calcular WoE e IV.
    target (str): nombre de la variable objetivo (debe ser binaria).
    bins (int, optional): número de bins si la variable es continua y necesita binning.

    Retorna:
    tabla: DataFrame que contiene los valores de WoE e IV para cada categoría/bin de la variable.
    iv_total: Information Value (IV) total para la variable.
    """
    # Si la variable es continua, aplicar binning
    if bins:
        df[variable] = pd.cut(df[variable], bins=bins, duplicates='drop')
    
    # Crear una tabla de frecuencia agrupando por la variable y calculando los buenos y malos
    tabla = df.groupby(variable).agg(
        buenos=(target, lambda x: (x == 0).sum()),
        malos=(target, lambda x: (x == 1).sum())
    ).reset_index()
    
    # Calcular totales de buenos y malos
    total_buenos = (df[target] == 0).sum()
    total_malos = (df[target] == 1).sum()
    
    # Calcular proporciones, WoE e IV
    tabla['dist_buenos'] = tabla['buenos'] / total_buenos
    tabla['dist_malos'] = tabla['malos'] / total_malos
    tabla['WoE'] = np.log(tabla['dist_buenos'] / tabla['dist_malos'].replace(0, np.nan))
    tabla['IV'] = (tabla['dist_buenos'] - tabla['dist_malos']) * tabla['WoE']
    
    # Reemplazar valores infinitos con 0
    tabla['WoE'] = tabla['WoE'].replace([np.inf, -np.inf], 0)
    
    # Calcular el IV total
    iv_total = tabla['IV'].sum()
    
    return tabla, iv_total




def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0



def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0, errors='ignore')
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



def cramer_matrix(data, categorical_vars):
    '''Función que calcula la matriz de valores v de Cramer para posteriormente poder graficarla
        - Inputs: dataframe, lista de variables categoricas
        - Output: matriz
        '''
    matrix = pd.DataFrame(index=categorical_vars, columns=categorical_vars, dtype=float)

    for var1 in categorical_vars:
        for var2 in categorical_vars:
            confusion_matrix = pd.crosstab(data[var1], data[var2])
            matrix.loc[var1, var2] = cramers_v(confusion_matrix.values)

    return matrix


def plot_model_evaluation(model, X_test, y_test):
    """
    Genera las siguientes gráficas:
    - Curva ROC AUC
    - Matriz de confusión (normalizada y sin normalizar)
    - Cumulative gains curve
    - Lift curve
    
    Parámetros:
    model: El modelo ya entrenado
    X_test: Features del conjunto de test
    y_test: Etiquetas reales del conjunto de test
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para ROC y otras curvas
    
    # Figura general
    plt.figure(figsize=(20, 16))
    
    # Curva ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Línea base
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    # Matriz de Confusión (sin normalizar)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 2, 2)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=plt.gca(), cmap='Blues', colorbar=False)
    plt.title('Matriz de Confusión (Sin Normalizar)')
    
    # Matriz de Confusión (normalizada)
    conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.subplot(2, 2, 3)
    ConfusionMatrixDisplay(conf_matrix_norm).plot(ax=plt.gca(), cmap='Blues', colorbar=False)
    plt.title('Matriz de Confusión (Normalizada)')
    
    # Cumulative Gains Curve
    plt.subplot(2, 2, 4)
    skplt.metrics.plot_cumulative_gain(y_test, model.predict_proba(X_test), ax=plt.gca())
    plt.title('Cumulative Gains Curve')
    
    # Crear figura separada para Lift Curve
    plt.figure(figsize=(10, 6))
    skplt.metrics.plot_lift_curve(y_test, model.predict_proba(X_test))
    plt.title('Lift Curve')
    
    plt.tight_layout()
    plt.show()

# Uso de la función
# plot_model_evaluation(model, X_test_selected, y_test)

def analyze_thresholds(model, X_test, y_test, X_train=None, features=None):
    """
    Analiza cómo cambian las métricas de rendimiento al ajustar el umbral de decisión del modelo.
    También genera Partial Dependence Plots si se proporcionan `X_train` y `features`.

    Parámetros:
    model: El modelo ya entrenado
    X_test: Features del conjunto de prueba
    y_test: Etiquetas reales del conjunto de prueba
    X_train: Features del conjunto de entrenamiento (opcional, para PDP)
    features: Lista de características para generar los PDP (opcional)
    """
    thresholds = np.arange(0, 1.1, 0.1)  # Rango de umbrales
    precision_list = []
    recall_list = []
    f1_list = []

    # Obtener probabilidades de predicción
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        # Ajustar el umbral
        y_pred_threshold = (y_pred_proba > threshold).astype(int)

        # Calcular métricas
        precision_list.append(precision_score(y_test, y_pred_threshold))
        recall_list.append(recall_score(y_test, y_pred_threshold))
        f1_list.append(f1_score(y_test, y_pred_threshold))

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_list, label="Precision", marker='o')
    plt.plot(thresholds, recall_list, label="Recall", marker='o')
    plt.plot(thresholds, f1_list, label="F1-score", marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Métricas en función del Umbral")
    plt.grid()
    plt.legend()
    plt.show()

    # Llamar a los Partial Dependence Plots si se proporcionan X_train y features
    if X_train is not None and features is not None:
        plot_partial_dependence_plots(model, X_train, features)


def plot_partial_dependence_plots(model, X_train, features):
    """
    Genera Partial Dependence Plots para las características especificadas.

    Parámetros:
    model: El modelo ya entrenado
    X_train: Features del conjunto de entrenamiento
    features: Lista de características para las que se generarán los PDP
    """
    from sklearn.inspection import PartialDependenceDisplay
    
    # Generar los Partial Dependence Plots
    plt.figure(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_train,
        features=features,
        kind='average',
        n_jobs=-1
    )
    plt.tight_layout()
    plt.show()


# Uso de la función
# features_to_plot = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_BIRTH']  # Ejemplo de características importantes
# plot_partial_dependence_plots(random_search.best_estimator_, X_train_selected, features_to_plot)