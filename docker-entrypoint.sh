#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties \
        --model-store /home/model-server/model-store/ \
        --workflow-store /home/model-server/workflow-store/
    sleep 5
    curl -X POST "http://0.0.0.0:8081/workflows?url=fc_wf.war"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null