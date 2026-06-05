"""
Thin CLI wrapper around api/predictor.py.
Use api/predictor.py directly when importing from FastAPI routes.
"""
import json
import argparse
from api.predictor import predict, predict_with_explanation


def main():
    parser = argparse.ArgumentParser(description="Run a single fraud prediction from CLI")
    parser.add_argument("--input", required=True, help="JSON string of transaction fields")
    parser.add_argument("--explain", action="store_true", help="Include SHAP explanations")
    args = parser.parse_args()

    data = json.loads(args.input)
    if args.explain:
        result = predict_with_explanation(data)
    else:
        result = predict(data)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
