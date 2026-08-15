import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = "file:./mlruns"

MODEL_NAME = "ecommerce-hybrid-recommender"

RUN_ID = "da3f7735781e4bada249729fe989f522"

ARTIFACT_PATH = "model_artifacts"


if __name__ == "__main__":

    print("=" * 60)
    print("REGISTERING CHAMPION MODEL")
    print("=" * 60)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    # Existing artifacts from our champion run
    source = (
        f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
    )

    print("\nCreating model version...")

    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=RUN_ID,
    )

    version = model_version.version

    print(
        f"Registered model: {MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

    # Add description
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            "Champion hybrid ecommerce recommendation "
            "system using item-item collaborative filtering "
            "and popularity ranking."
        ),
    )

    # Assign champion alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )

    print(
        f"\nAlias assigned:"
        f" {MODEL_NAME}@champion"
    )

    print("\n" + "=" * 60)
    print("MODEL REGISTRY COMPLETE")
    print("=" * 60)