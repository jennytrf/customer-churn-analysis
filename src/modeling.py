from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model
