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
def inference_ensemble(
    models: list,  # 모델 리스트
    weights: list,  # 모델 가중치 리스트
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    for model in models:
        model.to(device)
        model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 각 모델의 softmax 확률 계산
            logits_list = []
            for i, model in enumerate(models):
                logits = model(images)
                logits = F.softmax(logits, dim=1)  # 소프트맥스 적용 (확률 분포)
                logits_list.append(logits * weights[i])  # 가중치 곱하기
            
            # 가중치를 적용한 모델의 확률 합산
            avg_logits = torch.sum(torch.stack(logits_list), dim=0)
            
            # 평균된 확률에서 가장 높은 값의 클래스를 예측
            preds = avg_logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions


def test():
    device = 'cuda'
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    testdata_dir = './data/test'
    testdata_info_file = './data/test.csv'

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

    models = []

    model_names = [
        ('timm', 'caformer_b36.sail_in22k_ft_in1k'),
        ('timm','deit3_large_patch16_224.fb_in22k_ft_in1k'),
        ('timm','swin_large_patch4_window7_224.ms_in22k_ft_in1k'),
        ('hub', 'dinov2_vitl14_reg_lc'),
        ('timm', 'vit_large_patch14_clip_224.openai_ft_in1k'),
        ('timm', 'beitv2_large_patch16_224.in1k_ft_in22k_in1k')
    ]

    model_weights = [1, 1, 1, 1, 1, 2]

    # lora model
    # 추론에 사용할 Model을 선언.
    for i in range(len(model_names)):
        model_selector = ModelSelector(
            model_type=model_names[i][0], 
            num_classes=num_classes,
            model_name=model_names[i][1], 
            pretrained=True
        )
        model = model_selector.get_model()
        model = peft.PeftModel.from_pretrained(model, os.path.join(f'./train_result/{model_names[i][1]}', "best_model"))
        models.append(model)

    # predictions를 CSV에 저장할 때 형식을 맞춰서 저장
    # 테스트 함수 호출
    predictions = inference_ensemble(
        models=models, 
        weights=model_weights,  # 모델 가중치 전달
        device=device, 
        test_loader=test_loader
    )

    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info

    # DataFrame 저장
    test_info.to_csv(os.path.join('./', "soft_ens_output.csv"), index=False)
    print(f"Test info was saved in ./")
    print(f"Test done")

if __name__ == "__main__":
    # fix random seeds for reproducibility
    SEED = 999
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    test()

