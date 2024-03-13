# TalkNet
This service provides the inference for Active Speaker Detection.


## Clone repository
```
git clone https://github.com/rysmende/TalkNet-ASD.git
cd TalkNet-ASD
git checkout ts2
git pull origin ts2
```

## Install prerequisites
```
pip install -r requirements.txt
```

## Setup model weights
```
cd models
python setup.py
cd ..
```

## Build and run
```
mkdir model_store
chmod +x archiver.sh
./archiver.sh
docker build -t talk ./
docker run --rm -p 8080:8080 talk serve
```

## Send inference
`
curl http://0.0.0.0:8080/predictions/mono -F "data=@./demo/file.avi"
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
