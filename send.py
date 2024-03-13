import json
import base64

# input_text = b"Jaw dropping visual effects and action! One of the best I have seen to date."
# print(f"Input text: \n\t{input_text.decode('utf-8')}\n")
with open("demo/true.avi", "rb") as file:
    b64_encoded = base64.b64encode(file.read())

instance = {
    'instances': [{
        "data": str(b64_encoded.decode('utf-8'))
    }]
}

with open('data.json', 'w') as file:
    json.dump(instance, file)


# prediction = endpoint.predict(instances=instance)
# print(f"Prediction response: \n\t{prediction}")