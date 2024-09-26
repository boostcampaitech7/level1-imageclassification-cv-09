import os
import logging
from tqdm import tqdm
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
import csv
import pandas as pd

# diffusers 라이브러리의 로깅 설정: WARNING 이상 레벨만 출력되도록 설정
logging.getLogger("diffusers").setLevel(logging.WARNING)

# 기본 설정
device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
)
sd_pipe = sd_pipe.to(device)

# 데이터 경로 설정
train_dir = "./data/train"
csv_path = "./data/train.csv"
output_csv_path = "./data/synthetic_images.csv"

# train.csv 파일을 읽어서 class_name, target을 dictionary로 저장
train_df = pd.read_csv(csv_path, header=None, names=["class_name", "image_path", "target"])
class_to_target = dict(zip(train_df["class_name"], train_df["target"]))

# 이미지 변환을 위한 transform 설정
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])

# synthetic 이미지 생성 함수
def generate_synthetic_images(image_path, output_dir, start_index, num_synthetic=2):
    # 원본 이미지 불러오기 (흑백 이미지는 RGB로 변환)
    im = Image.open(image_path).convert("RGB")
    
    # 이미지를 모델에 입력하기 위한 변환 적용
    inp = tform(im).to(device).unsqueeze(0)
    
    # 한 번의 호출로 num_synthetic 개의 이미지 생성 (num_images_per_prompt 사용)
    out = sd_pipe(inp, guidance_scale=3, num_inference_steps=50, num_images_per_prompt=num_synthetic)
    
    # 생성된 synthetic 이미지 저장 및 이미지 정보 반환
    synthetic_image_info = []
    for i, image in enumerate(out["images"]):
        if out["nsfw_content_detected"][i]:
            continue
        output_image_path = os.path.join(output_dir, f"s_sketch_{start_index + i}.JPEG")
        image.save(output_image_path)
        synthetic_image_info.append(output_image_path)
    
    return synthetic_image_info

# 폴더 내 모든 이미지를 처리하는 함수
def process_images_in_folder(folder_path, class_name, target, synthetic_image_data):
    # 이미지 파일 목록을 가져옴 (.으로 시작하는 파일은 제외)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.JPEG') and not f.startswith('.')]
    
    # 전체 이미지에 대해 순차적으로 synthetic 이미지를 생성
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        
        # 두 개의 synthetic 이미지를 생성하여 같은 폴더에 저장
        start_index = idx * 2
        synthetic_images = generate_synthetic_images(image_path, folder_path, start_index, num_synthetic=2)
        
        # synthetic 이미지 정보 수집 (class_name, image_path, target)
        for synthetic_image_path in synthetic_images:
            relative_image_path = os.path.relpath(synthetic_image_path, start=folder_path)
            synthetic_image_data.append([class_name, relative_image_path, target])

# train 디렉토리 하위의 모든 클래스 폴더를 순회하며 이미지 처리
def process_all_folders_in_train(train_dir):
    # Synthetic 이미지 정보를 저장할 리스트
    synthetic_image_data = []

    # train 폴더 내 모든 서브폴더를 탐색
    subfolders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    ii = 0

    # tqdm을 사용하여 폴더 처리 진행 상황만 표시
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        print("####################", ii)
        subfolder_path = os.path.join(train_dir, subfolder)
        
        # class_name과 target 값을 가져옴
        class_name = subfolder
        target = class_to_target.get(class_name, -1)  # 만약 class_name이 없으면 -1을 할당
        
        # 각 폴더에 대해 synthetic 이미지 생성 및 정보 수집
        process_images_in_folder(subfolder_path, class_name, target, synthetic_image_data)
        ii+=1
    # synthetic 이미지 정보를 CSV 파일로 저장
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "image_path", "target"])  # 헤더
        writer.writerows(synthetic_image_data)

# 실제 실행 코드: train 폴더의 모든 서브폴더에 대해 처리
process_all_folders_in_train(train_dir)
