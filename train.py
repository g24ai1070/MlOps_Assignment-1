# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model, RANDOM_STATE

def main():
    print("== MLOps A1: DecisionTreeRegressor ==")
    df = load_data(cache=True)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, scale=False)
    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test, label="DecisionTreeRegressor")

if __name__ == "__main__":
    main()
