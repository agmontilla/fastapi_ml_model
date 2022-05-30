""" Model module """
import pickle
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lib.logger import logger


class AuthenticationFeatures(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float


class ETL:
    def load_dataset(self, path_file: str):
        df = pd.read_csv(path_file)
        return df


class Model:
    def __init__(self, estimator: Optional[BaseEstimator]) -> None:
        self.estimator = estimator
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=101
        )
        self.model = self.estimator.fit(X_train.values, y_train.values)

        predictions = self.model.predict(X_test.values)

        logger.info(
            f"Classification report:\n{classification_report(y_test, predictions)}"
        )

        logger.info("finished training")

    def serialize(self, model_name: str) -> str:
        with open(model_name, "wb") as file:
            pickle.dump(self.model, file)

    def unserialize(self, model_name: str) -> None:
        with open(model_name, "rb") as file:
            self.model = pickle.load(file)


# class Model:
#     # 6. Class constructor, loads the dataset and loads the model
#     #    if exists. If not, calls the _train_model method and
#     #    saves the model
#     def __init__(self):
#         self.df = pd.read_csv("iris.csv")
#         self.model_fname_ = "iris_model.pkl"
#         try:
#             self.model = joblib.load(self.model_fname_)
#         except Exception as _:
#             self.model = self._train_model()
#             joblib.dump(self.model, self.model_fname_)

#     # 4. Perform model training using the RandomForest classifier
#     def _train_model(self):
#         X = self.df.drop("species", axis=1)
#         y = self.df["species"]
#         rfc = RandomForestClassifier()
#         model = rfc.fit(X, y)
#         return model

#     # 5. Make a prediction based on the user-entered data
#     #    Returns the predicted species with its respective probability
#     def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
#         data_in = [[sepal_length, sepal_width, petal_length, petal_length]]
#         prediction = self.model.predict(data_in)
#         probability = self.model.predict_proba(data_in).max()
#         return prediction[0], probability
