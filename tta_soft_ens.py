import torch
import random
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data_loader.transform import TransformSelector
from data_loader.dataset import CustomDataset
from model.model import ModelSelector
import torch.nn.functional as F
import peft
import albumentations as A  # albumentations 라이브러리 추가
from albumentations.pytorch import ToTensorV2

# TTA용 Augmentation Transform 정의
def get_tta_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # 좌우 반전
        A.RandomRotate90(p=0.5),  # 90도 회전
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),  # 랜덤 이동, 스케일, 회전
        A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 조정
        ToTensorV2()
    ])

# 모델 추론을 위한 TTA 및 소프트 앙상블 함수
def inference_ensemble_tta(
    models: list,  # 모델 리스트
    weights: list,  # 모델 가중치 리스트
    device: torch.device, 
    test_loader: DataLoader,
    tta_times: int = 5  # TTA 횟수 (각 이미지에 대해 몇 번의 변환을 적용할지)
):
    # 모델을 평가 모드로 설정
    for model in models:
        model.to(device)
        model.eval()
    
    predictions = []
    tta_transform = get_tta_transform()  # TTA 변환 설정

    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            original_images = images.clone().to(device)  # 원본 이미지 저장

            # TTA를 적용하여 여러 번 변환된 이미지를 모델에 입력
            logits_list = []
            for tta_index in range(tta_times):
                # 이미지를 변환 (TTA 적용)
                augmented_images = torch.stack([
                    tta_transform(image=image.permute(1, 2, 0).cpu().numpy())['image']
                    for image in original_images
                ]).to(device)

                # 각 모델의 softmax 확률 계산
                for i, model in enumerate(models):
                    logits = model(augmented_images)
                    logits = F.softmax(logits, dim=1)  # 소프트맥스 적용 (확률 분포)
                    logits_list.append(logits * weights[i])  # 가중치 곱하기

            # TTA 적용된 여러 예측의 평균값 계산
            avg_logits = torch.sum(torch.stack(logits_list), dim=0) / tta_times
            
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
        ('timm', 'vit_large_patch14_clip_224.openai_ft_in1k')
    ]

    model_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
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
    # TTA 적용된 앙상블 함수 호출
    predictions = inference_ensemble_tta(
        models=models, 
        weights=model_weights,  # 모델 가중치 전달
        device=device, 
        test_loader=test_loader,
        tta_times=5  # TTA 횟수
    )

    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info

    # DataFrame 저장
    test_info.to_csv(os.path.join('./', "tta_soft_ens_output.csv"), index=False)
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
