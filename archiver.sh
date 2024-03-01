torch-model-archiver -f --model-name s3fd --version 1.0 \
    --model-file models/s3fd.py --serialized-file model_weights/s3fd.pth \
    --handler handlers/s3fd_handler.py --export-path model_store \
    --extra-files models/s3fd_utils.py,models/s3fd_modules.py,handlers/fd_utils.py

torch-model-archiver -f --model-name talk --version 1.0 \
    --model-file models/talk.py --serialized-file model_weights/talk.pth \
    --handler handlers/talk_handler.py --export-path model_store \
    --extra-files models/talk_modules.py

torch-workflow-archiver -f --workflow-name asd_wf --spec-file workflow.yaml \
    --handler handlers/wf_handler.py --export-path wf_store/