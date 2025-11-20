
if __name__ == "__main__":
    import joblib
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    model = joblib.load("model.joblib")

    print("n_features_in =", model.n_features_in_)
    # print("Number of support vectors =", len(model.support_))

    # If you only saved SVC(model), wrap manually:
    # model = Pipeline([("svc", model)])

    n_features = model.n_features_in_

    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("SVC (probability=True) exported to ONNX successfully.")