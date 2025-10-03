# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    print("== MLOps A1: KernelRidge ==")
    df = load_data(cache=True)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, scale=True)
    model = KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test, label="KernelRidge (rbf)")

if __name__ == "__main__":
    main()
