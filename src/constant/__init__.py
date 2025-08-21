import os


AWS_S3_BUCKET_NAME = "sensor-faulty"
AWS_ACCESS_KEY="AKIA3GRS27XW77DAPWU5"
AWS_SECRET_ACCESS_KEY="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"
AWS_DEFAULT_REGION="US East (N. Virginia)"
AWS_ECR_REPO_URI="769979514349.dkr.ecr.us-east-1.amazonaws.com/sensor-final-deployment"


MONGO_DATABASE_NAME = "ML"
MONGO_COLLECTION_NAME = "Sensor"


TARGET_COLUMN = "quality"
MONGO_DB_URL="mongodb+srv://sakilambhargav:bhargav2006@mlprojects/ML"#url//username  pass @cluster/db name


MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder =  "artifacts"