import json
import base64

from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

instances = {
      "token": "ya29.a0Ad52N3_D3gdh7XhyONIz3VmdznQAEX9nOUGhfCfKnGbTt1kqqoCW4t2khrfxdjxSCS7laPOjhknZ68NIUl-kH9rEf_gRw5Hq5iadBTA0PEvPT_-LRijHpURMVdhssEiGO-29LNm1GZp0JQVblyFHpOKIGaOXpOE9rtfWaCgYKAckSARISFQHGX2Mixcm5gCIc_4Yu2nE45blFbg0171",
      "bucket_name": "biometry-416410_cloudbuild",
      "object_name": "true.avi"
    }
predict_custom_trained_model_sample(
    project="768262733295",
    endpoint_id="3617087591150518272",
    location="us-central1",
    instances=instances
)