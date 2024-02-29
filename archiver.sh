torch-model-archiver -f --model-name s3fd --version 1.0 \
    --model-file models/s3fd.py --serialized-file model_weights/s3fd.pth \
    --handler handlers/s3fd_handler.py --export-path model_store \
    --extra-files models/s3fd_utils.py,models/s3fd_modules.py

# torch-model-archiver -f --model-name lightcnn --version 1.0 \
#     --model-file models/lightcnn.py --serialized-file model_weights/lightcnn.pth \
#     --handler handlers/lightcnn_handler.py --export-path model_store \
#     --extra-files handlers/lightcnn_utils.py,models/lightcnn_modules.py

torch-workflow-archiver -f --workflow-name fc_wf --spec-file workflow.yaml \
    --handler handlers/wf_handler.py --export-path workflow_store/