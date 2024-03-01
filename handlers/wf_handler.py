import base64

def pre_processing(data, context):
    '''
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    '''
    if data is None:
        return data
    b64_data = []
    for row in data:
        data = row.get('image')
        # selfie_data = row.get('selfie_class')
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        # b64_data.append({'front_id': front_data, 'back_id': back_data})
        b64_data.append({
            'image': base64.b64encode(data).decode(),
            # 'selfie': base64.b64encode(selfie_data).decode(),
        })
    return b64_data
