import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms

class AlbumentationsTransform:
    def __init__(self, is_train: bool = True, transform_config:str=None):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(self._create_augmentations(transform_config) + common_transforms)
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def _create_augmentations(self, augmentations):
        # 각 증강 설정을 기반으로 변환 생성
        aug_list = []
        for aug in augmentations:
            aug_class = getattr(A, aug["type"])  # A.HorizontalFlip 등으로 변환
            aug_list.append(aug_class(**aug["params"]))  # 매개변수 적용
        return aug_list
    
    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        return transformed['image']  # 변환된 이미지의 텐서를 반환

class TorchvisionTransform:
    def __init__(self, is_train: bool = True, transform_config:str=None):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            transforms.Resize(224, 224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = transforms.Compose(self._create_augmentations(transform_config) + common_transforms)
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def _create_augmentations(self, augmentations):
        # 각 증강 설정을 기반으로 변환 생성
        aug_list = []
        for aug in augmentations:
            aug_class = getattr(A, aug["type"])  # A.HorizontalFlip 등으로 변환
            aug_list.append(aug_class(**aug["params"]))  # 매개변수 적용
        return aug_list
    
    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        
        return transformed['image']  # 변환된 이미지의 텐서를 반환

class AutoAug:
    def __init__(self, is_train: bool = True, transform_config:str=None):
        common_transforms = [ 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize((224, 224)),
        ]
        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.AutoAugment()] + common_transforms)
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose([transforms.ToTensor()] + common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image)  # 이미지에 설정된 변환을 적용
        return transformed  # 변환된 이미지의 텐서를 반환

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str, transform_config: str=None):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["albumentations", "torchvision", "autoaugment"]:
            self.transform_type = transform_type
            self.transform_config = transform_config
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train, transform_config=self.transform_config)
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train, transform_config=self.transform_config)
        if self.transform_type == 'autoaugment':
            transform = AutoAug()
        return transform