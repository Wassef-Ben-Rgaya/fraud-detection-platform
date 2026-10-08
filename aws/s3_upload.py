import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def upload_model_to_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "eu-west-1")
    )

    bucket     = os.getenv("S3_BUCKET_NAME")
    model_path = "mlflow/best_model.pkl"

    print(f"📤 Upload du modèle vers s3://{bucket}/models/best_model.pkl ...")
    s3.upload_file(model_path, bucket, "models/best_model.pkl")
    print("✅ Modèle uploadé avec succès !")

    # Générer URL publique temporaire
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": "models/best_model.pkl"},
        ExpiresIn=3600
    )
    print(f"🔗 URL temporaire : {url}")
    return url

if __name__ == "__main__":
    upload_model_to_s3()