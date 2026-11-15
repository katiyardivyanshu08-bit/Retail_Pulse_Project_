import joblib

model = joblib.load("Models/kmeans_model.pkl")

print("Model type:", type(model))

if isinstance(model, dict):
    print("\nDictionary keys:", model.keys())

    for key, value in model.items():
        print(f"\n{key}:")
        print(type(value))
        print(value if key == "features" or key == "cluster_names" else "")

else:
    print("\nModel details:")
    print(model)

    if hasattr(model, "named_steps"):
        print("\nPipeline steps:", model.named_steps.keys())