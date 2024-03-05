# TalkNet
This service provides the inference for Active Speaker Detection.

## Build and run
```
git clone https://github.com/rysmende/TalkNet-ASD.git
cd TalkNet-ASD
git branch ts
chmod +x archiver.sh
./archiver.sh
docker build -t talk ./
docker run --rm -p 8080:8080 talk serve
```

## Send inference
`
curl http://127.0.0.1:8080/wfpredict/asd_wf -F "data=@./demo/file.avi"
`

## Result
```
[
  [
    -1.0
    -0.5
     0.0
     0.3
     0.8
     1.0
     ...
  ]
]
```
