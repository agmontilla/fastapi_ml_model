from model.model import Model, ETL
from sklearn.linear_model import LogisticRegression
from uuid import uuid4

if __name__ == "__main__":
    # 1. Load the dataset
    etl = ETL()
    df = etl.load_dataset("../etl/BankNote_Authentication.csv")

    # 2. Create the model
    clf = Model(LogisticRegression())

    # 3. Train the model
    X = df.drop(["class"], axis=1)
    y = df["class"]
    clf.train(X, y)

    # 4. Serialize the model
    clf.serialize(f"../models_dump/{uuid4()}.pkl")
