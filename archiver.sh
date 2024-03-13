torch-model-archiver -f --model-name mono --version 1.0 \
    --model-file models/mono.py --serialized-file model_weights/mono.pth \
    --handler handlers/mono_handler.py --export-path model_store \
    --extra-files models/s3fd.py,models/s3fd_utils.py,models/talk.py,models/talk_modules.py,models/mono_utils.py
