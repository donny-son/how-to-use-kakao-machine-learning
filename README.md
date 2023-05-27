# README

## Install kiml

```bash
pip install kiml --index-url https://pypi.ml.kakaoicloud-kr-gov.com/simple --trusted-host pypi.ml.kakaoicloud-kr-gov.com --force-reinstall
```

## Login

```
kiml login
```

## Set workspace

```
kiml workspace set zarathucorp
```

## Upload data to storage

```
kiml data upload mnist_dataset default/mnist_dataset

kiml data list default/mnist_dataset # check upload status
```

## Create Dataset

```
kiml dataset create mnist_dataset --path /default/mnist_dataset

kiml dataset list # check dataset status
```

## Training(Experiment)

```
kiml experiment create mnist-experiment


kiml run submit \
--dataset mnist-dataset \
--experiment mnist-experiment \
--image cosmos-pytorch1.10.0-cuda11.3-cudnn8-py3.7-ubuntu18.04 \
--instance-type 1A100-16-MO \
--num-replica 1 "python mnist.py --data_dir /app/input/dataset/mnist-dataset --output_dir /app/outputs"
```

## View Experiment

```
kiml run list

kiml run log <RUN_ID> --follow

kiml experiment tensorboard <EXPERIMENT_ID>
```

## Download Outputs

```
kiml data list default/experiments/<EXPERIMENT_ID>/runs/<RUN_ID>/outputs

kiml data download default/experiments/<EXPERIMENT_ID>/runs/<RUN_ID>/outputs
```

## 기술 문의 

- 1688-0301 -> 2번
- icloudhelpdesk@kakaoenterprise.com

