import torch
import random
import os
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from config_parser import ConfigParser
from data_loader.transform import TransformSelector
from data_loader.dataset import CustomDataset
from model.model import ModelSelector
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import peft

# 모델 추론을 위한 함수
def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions

def test(config):
    device = config['device']
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    testdata_dir = config['testdata_dir']
    testdata_info_file = config['testdata_info_file']
    train_result_path = os.path.join(config['train_result_path'], config['exp_name'])
    test_result_path = os.path.join(config['test_result_path'], config['exp_name'])
    os.makedirs(test_result_path, exist_ok=True)

    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_info = pd.read_csv(testdata_info_file)

    # 총 class 수.
    num_classes = 500

    # 추론에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = "albumentations"
    )
    test_transform = transform_selector.get_transform(is_train=False)
    # 추론에 사용할 Dataset을 선언.
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )

    # 추론에 사용할 DataLoader를 선언.
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        drop_last=False
    )

    # 추론에 사용할 장비를 선택.
    # torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
    device = config['device']



    # lora model
    if config.get('lora') and config['lora']['use']:
        # 추론에 사용할 Model을 선언.
        model_selector = ModelSelector(
            model_type=config['model_type'], 
            num_classes=num_classes,
            model_name=config['model_name'], 
            pretrained=True
        )
        model = model_selector.get_model()
        model = peft.PeftModel.from_pretrained(model, os.path.join(train_result_path, "best_model"))

    # general model
    else:
        # 추론에 사용할 Model을 선언.
        model_selector = ModelSelector(
            model_type=config['model_type'], 
            num_classes=num_classes,
            model_name=config['model_name'], 
            pretrained=False
        )
        model = model_selector.get_model()
        # best epoch 모델을 불러오기.
        model.load_state_dict(
            torch.load(
                os.path.join(train_result_path, "best_model.pt"),
                map_location='cpu',
                weights_only=True
            )
        )

    # predictions를 CSV에 저장할 때 형식을 맞춰서 저장
    # 테스트 함수 호출
    predictions = inference(
        model=model, 
        device=device, 
        test_loader=test_loader
    )

    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info

    # DataFrame 저장
    test_info.to_csv(os.path.join(test_result_path, "test_output.csv"), index=False)
    print(f"Test info was saved in {test_result_path}")
    print(f"Test done")

def valid(config):
    device = config['device']
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    traindata_dir = config['traindata_dir']
    traindata_info_file = config['traindata_info_file']
    train_result_path = os.path.join(config['train_result_path'], config['exp_name'])
    test_result_path = os.path.join(config['test_result_path'], config['exp_name'])
    os.makedirs(test_result_path, exist_ok=True)

    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_info = pd.read_csv(traindata_info_file)

    # 총 class 수.
    num_classes = len(train_info['target'].unique())
    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    _, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=config['seed'],
    )
    # 추론에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = "albumentations"
    )
    val_transform = transform_selector.get_transform(is_train=False)
    
    val_infer_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform,
        is_inference=True
    )

    val_infer_loader = DataLoader(
        val_infer_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )

    # 추론에 사용할 장비를 선택.
    # torch라이브러리에서 gpu를 인식할 경우, cuda로 설정.
    device = config['device']



    # lora model
    if config.get('lora') and config['lora']['use']:
        # 추론에 사용할 Model을 선언.
        model_selector = ModelSelector(
            model_type=config['model_type'], 
            num_classes=num_classes,
            model_name=config['model_name'], 
            pretrained=True
        )
        model = model_selector.get_model()
        model = peft.PeftModel.from_pretrained(model, os.path.join(train_result_path, "best_model"))

    # general model
    else:
        # 추론에 사용할 Model을 선언.
        model_selector = ModelSelector(
            model_type=config['model_type'], 
            num_classes=num_classes,
            model_name=config['model_name'], 
            pretrained=False
        )
        model = model_selector.get_model()
        # best epoch 모델을 불러오기.
        model.load_state_dict(
            torch.load(
                os.path.join(train_result_path, "best_model.pt"),
                map_location='cpu',
                weights_only=True
            )
        )

    # predictions를 CSV에 저장할 때 형식을 맞춰서 저장
    # 테스트 함수 호출
    val_predictions = inference(
        model=model,
        device=device,
        test_loader=val_infer_loader
    )
    
    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    val_df['prediction'] = val_predictions
    val_df['is_correct'] = val_df['target'] == val_df['prediction']
    incorrect_predictions = val_df[val_df['is_correct'] == False]

    correct_predictions = val_df['is_correct'].sum()
    total_predictions = len(val_df)
    val_accuracy = correct_predictions / total_predictions * 100

    # DataFrame 저장
    val_df.to_csv(os.path.join(test_result_path, "valid_output.csv"), index=False)
    incorrect_predictions.to_csv(os.path.join(test_result_path, "valid_incorrect_preds.csv"), index=False)
    print(f"Total valid data: {total_predictions}")
    print(f"Correct valid data: {correct_predictions}")
    print(f"Valid Accuracy: {val_accuracy:.2f}%")
    print(f"Valid info was saved in {test_result_path}")
    print("Valid done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the configuration YAML file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', action='store_true', help="Run in test mode")
    group.add_argument('--valid', action='store_true', help="Run in validation mode")
    group.add_argument('--all', action='store_true', help="Run in test and validation mode" )
                        
    args = parser.parse_args()

    config_parser = ConfigParser(args.config)
    config = config_parser.config
    
    # fix random seeds for reproducibility
    SEED = config['seed']
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if args.test:
        test(config = config)
    elif args.valid:
        valid(config = config)
    elif args.all:
        valid(config = config)
        test(config = config)