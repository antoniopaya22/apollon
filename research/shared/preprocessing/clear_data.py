# ========================== Clear Data ==========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ================================================================

# ==================> Imports
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shared.models.model as m

from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

pd.options.mode.chained_assignment = None  # default='warn'


# ==================> Classes
class ClearData:
    def __init__(self, df: pd.DataFrame, seed: int, quantile: float = 0.95, max_mediana: int = 10, log_unic: int = 50,
                 label_f: int = 6) -> None:
        """__init__

        This method is used to initialize the ClearData class.

        Parameters:
            df (pd.DataFrame): Dataframe to clean.
            quantile (float): Percentile to cut the data.
            max_mediana (int): Max value to cut the data.
            log_unic (int): Max unique values to apply log.
            label_f (int): Label to predict.
        Output:
            None
        """
        self.df = df
        self.quantile = quantile
        self.max_mediana = max_mediana
        self.log_unic = log_unic
        self.label_f = label_f
        self.x = None
        self.y = None
        # Lo marco a False ya que no que por defecto me haga el logaritmo
        self.do_log = False
        # Lo marco a False ya que no que por defecto me saque la grafica de las caracteristicas
        self.show_features = False
        # Lo marco a False ya que no que por defecto me guarde el dataframe preprocesado
        self.save = False
        self.seed = seed

    def clear_data(self) -> None:
        """clear_data

        This method is used to clean the data. Now, it only pass.
        This method is override in the future, by the class's childrens.
        """
        pass

    def save_data(self) -> None:
        """save_data

        This method is used to save the data. Now, it only pass. This method is overrided in the future, by the class's
        childrens.
        """
        pass

    def load_data(self):
        """load_data

        This method is used to load the data.
        """
        pass

    def best_features_func(self) -> None:
        """best_features_func

        This method is used to select the best features.

        Output:
            None
        """
        self.drop_one_features()
        self.drop_duplicate_columns()
        self.drop_bad_elements()
        df_numeric = self.df.select_dtypes(include=[np.number])
        df_before = self.clear_valores_atipicos(df_numeric)
        if self.do_log:
            df_cat = self.clear_log(df_before, df_numeric)
        else:
            df_cat = self.df.select_dtypes(exclude=[np.number])
        self.clear_freq(df_cat)
        self.drop_bad_elements()

    def drop_bad_elements(self) -> None:
        """drop_bad_elements

        This method is used to drop bad elements.

        Output:
            None
        """
        pass
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0)

        # self.df = self.df[~self.df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    def drop_bad_elements_x(self) -> None:
        """drop_bad_elements_x

        This method is used to drop bad elements.

        Output:
            None
        """
        self.x[np.isnan(self.x)] = 0
        self.x[np.isinf(self.x)] = 0
        self.x[np.isneginf(self.x)] = 0
        # self.x[self.x < 0] = 0

    def drop_duplicate_columns(self) -> None:
        """drop_duplicate_columns

        This method is used to drop duplicate columns.

        Output:
            None
        """
        # drop duplicate rows
        self.df.drop_duplicates(keep="first")
        self.df.reset_index(drop=True, inplace=True)

    def drop_one_features(self) -> None:
        """drop_one_features

        This method is used to drop one features.

        Output:
            None
        """
        # drop one variable features
        one_variable_list = []
        for i in self.df.columns:
            if self.df[i].value_counts().nunique() < 2:
                one_variable_list.append(i)
        self.df.drop(one_variable_list, axis=1, inplace=True)

    def clear_valores_atipicos(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        """clear_valores_atipicos

        Aqui se podan los valores atipicos, para ello las caracteristicas con un valor T (10) veces superior a la
        mediana se podan hasta el percentil Z (95), se escoge las que son T (10) veces superior ya que con esto
        evitamos la poda de caracteristicas con valores pequeños y distribuciones bimodales (?).
        Si el percentil Z (95) esta más cerca del maximo (por ecima de dicho percentil), entonces puede llegar a tener
        informacion interesante que no queramos descartar.

        Parameters:
            df_numeric (pd.DataFrame): Dataframe to clean.
        Output:
            df_before (pd.DataFrame): Dataframe cleaned.
        """

        for feature in df_numeric.columns:
            if df_numeric[feature].max() > self.max_mediana * df_numeric[feature].median():
                self.df[feature] = np.where(self.df[feature] < self.df[feature].quantile(self.quantile),
                                            self.df[feature], self.df[feature].quantile(self.quantile))
        return df_numeric.copy()

    def clear_log(self, df_before: pd.DataFrame, df_numeric: pd.DataFrame) -> pd.DataFrame:
        """clear_log

        Aqui se aplica la funcion logaritmica a aquellas caracteristicas que superen los Y (50) valores unicos.
        La razón por la que se buscan más de Y (50) valores únicos es para filtrar las características basadas
        en números enteros que actúan de forma más categórica.

        Parameters:
            df_before (pd.DataFrame): Dataframe to clean.
            df_numeric (pd.DataFrame): Dataframe to clean.
        Output:
            df_cat (pd.DataFrame): Dataframe cleaned.
        """
        for feature in df_before.columns:
            if df_numeric[feature].nunique() > self.log_unic:
                if df_numeric[feature].min() == 0:
                    self.df[feature] = np.log(self.df[feature] + 1)
                else:
                    self.df[feature] = np.log(self.df[feature])
        return self.df.select_dtypes(exclude=[np.number])

    def clear_freq(self, df_cat: pd.DataFrame) -> None:
        """clear_freq
            Aqui se escogen las X (6) etiquetas más frecuentes de la característica
            y establecer el resto como etiquetas "-" (poco utilizadas)

        Parameters:
            df_cat (pd.DataFrame): Dataframe to clean.
        Output:
            None
        """
        for feature in df_cat.columns:
            if df_cat[feature].nunique() > self.label_f and df_cat[feature].nunique() / self.label_f:
                self.df[feature] = np.where(self.df[feature].isin(self.df[feature].value_counts().head().index),
                                            self.df[feature], '-')
        self.drop_bad_elements()

    def reduce_tam(self) -> None:
        """reduce_tam

        This method is used to reduce the size of the dataframe.

        Output:
            None
        """
        integer = []
        f = []
        for i in self.df.columns[:-1]:
            if self.df[i].dtype == "int64":
                integer.append(i)
            else:
                f.append(i)

        self.df[integer] = self.df[integer].astype("int32")
        self.df[f] = self.df[f].astype("float32")

    def loss_data(self) -> None:
        """loss_data

        This method is used to calculate the loss of data.

        Output:
            None
        """
        df1 = self.df[self.df["Label"] == "Benign"][:len(
            self.df[self.df["Label"] == "Malicious"])]
        df2 = self.df[self.df["Label"] == "Malicious"][:len(
            self.df[self.df["Label"] == "Malicious"])]
        self.df = pd.concat([df1, df2], axis=0)

    def cast_time(self, label: str, time_format: str) -> None:
        """cast_time

        This method is used to cast the time to a specific format.

        Parameters:
            label (str): Label of the column to cast.
            time_format (str): Format to cast.
        Output:
            None
        """
        df_aux = self.df[label]
        df_aux_2 = []
        for i in df_aux:
            element = datetime.datetime.strptime(i, time_format)
            timestamp = datetime.datetime.timestamp(element)
            df_aux_2.append(timestamp)

        self.df[label] = df_aux_2

    def one_shotear(self, list_columns: list) -> np.array:
        """one_shotear

        This method is responsible for applying the one hot encoder function to given columns

        Parameters:
            list_columns (list): List of columns to apply the one hot encoder.
        Output:
            df_one_hot (np.array): Array with the one hot encoder applied.
        """
        ct = ColumnTransformer(transformers=[(
            'encoder', OneHotEncoder(), list_columns)], remainder='passthrough')
        self.x = np.array(ct.fit_transform(self.x))

    def replace(self, list_B_columns: list = None, list_M_columns: list = None) -> None:
        """replace

        This method is responsible for replacing the label's values.
        Standardize those values to Benign for normal traffic and Malicious for malicious traffic.

        Parameters:
            list_B_columns (list): List of columns to replace the Benign label.
            list_M_columns (list): List of columns to replace the Malicious label.
        Output:
            None
        """
        if not type(list_B_columns) is None:
            self.y.replace(to_replace=list_B_columns, value="0", inplace=True)
        if not type(list_M_columns) is None:
            self.y.replace(to_replace=list_M_columns, value="1", inplace=True)

    def show_img_features(self, fit) -> None:
        """show_img_features

        This method is responsible for showing the features selected by the SelectKBest function.

        Parameters:
            fit SelectKBest function.
        Output:
            None
        """
        df_scores = pd.DataFrame(fit.scores_)
        df_col = pd.DataFrame(self.x.columns)

        feature_score = pd.concat([df_col, df_scores], axis=1)
        feature_score.columns = ['feature', 'score']
        feature_score.sort_values(by=['score'], ascending=True, inplace=True)

        fig = go.Figure(go.Bar(
            x=feature_score['score'][0:21], y=feature_score['feature'][0:21], orientation='h'))

        fig.update_layout(title="Top 20 Features",
                          height=1200, showlegend=False, )
        fig.show()

    def show_img_features_names(self, model: m.Model, df_cat: pd.DataFrame) -> None:
        """show_img_features_names

        This method is responsible for showing the features selected by the SelectKBest function.

        Parameters:
            model (m.Model): Model to show the features.
            df_cat (pd.DataFrame): Dataframe with the categorical features.
        Output:
            None
        """

        feature_names = list(self.x.columns)
        for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
            feature_names.insert(0, label)

        for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
            feature_names.insert(0, label)

        for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
            feature_names.insert(0, label)

        plt.rcParams['figure.figsize'] = 10, 10
        sns.set_style("white")
        feat_importances = pd.Series(
            model.feature_importances_, index=feature_names)
        feat_importances = feat_importances.groupby(level=0).mean()
        feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
        sns.despine()
        plt.show()
