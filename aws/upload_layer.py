import boto3
import os
from dotenv import load_dotenv

load_dotenv()


def upload_layer_to_s3():

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "eu-north-1")
    )

    bucket = os.getenv("S3_BUCKET_NAME")

    layer_path = "lambda-layer.zip"
    layer_key = "lambda/lambda-layer.zip"

    if not os.path.exists(layer_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {layer_path}"
        )

    print(
        f"📤 Upload de la Layer vers "
        f"s3://{bucket}/{layer_key} ..."
    )

    s3.upload_file(
        layer_path,
        bucket,
        layer_key
    )

    print("✅ Layer uploadée avec succès !")


if __name__ == "__main__":
    upload_layer_to_s3()
