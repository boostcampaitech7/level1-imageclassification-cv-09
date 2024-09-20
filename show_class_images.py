# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from glob import glob
# from collections import defaultdict
# from PIL import Image
# import numpy as np
# import cv2

# # 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정
# traindata_dir = "./data/train"
# traindata_info_file = "./data/train.csv"

# # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기
# train_data = pd.read_csv(traindata_info_file)

# # glob을 이용하여 이미지 파일의 경로를 읽어옴
# train_images = glob(traindata_dir + "/*/*")

# image_prop = defaultdict(list)

# for i, path in enumerate(train_images):
#     with Image.open(path) as img:
#         image_prop['height'].append(img.height)
#         image_prop['width'].append(img.width)
#         image_prop['img_aspect_ratio'] = img.width / img.height
#         image_prop['mode'].append(img.mode)
#         image_prop['format'].append(img.format)
#         image_prop['size'].append(round(os.path.getsize(path) / 1e6, 2))
#     image_prop['path'].append(path)
#     image_prop['image_path'].append(path.split('/')[-2] + "/" + path.split('/')[-1])

# image_data = pd.DataFrame(image_prop)

# image_data = image_data.merge(train_data, on='image_path')
# #image_data.sort_values(by='target', inplace=True)

# # 같은 target을 가진 이미지 전체 출력
# def display_images(data, target):
#     len_data = len(data[data['target'] == target])
#     fig, axs = plt.subplots((len_data // 5)+1, 5, figsize=(16, 10))
#     images = data[data['target'] == target]['path'].values
#     for i, path in enumerate(images):
#         img = Image.open(path)
#         ax = axs[i // 5, i % 5]  # Use double indexing for 2D subplots
#         ax.imshow(img)
#         ax.axis('off')
#     plt.savefig(f'class_train_view_images/{target}_images.png') 


# # target이 0인 이미지 출력
# display_images(image_data, target=1)
# print("? 왜 안됨")

import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from PIL import Image

# 경로 설정
traindata_dir = "./data/train"
traindata_info_file = "./data/train.csv"
output_dir = "./class_train_view_images"

# 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# CSV 파일 읽기
train_data = pd.read_csv(traindata_info_file)

# 이미지 파일 경로 읽기
train_images = glob(traindata_dir + "/*/*")

# 이미지 속성 수집
image_prop = defaultdict(list)
for path in train_images:
    with Image.open(path) as img:
        image_prop['height'].append(img.height)
        image_prop['width'].append(img.width)
        image_prop['mode'].append(img.mode)
        image_prop['format'].append(img.format)
        image_prop['size'].append(round(os.path.getsize(path) / 1e6, 2))
    image_prop['path'].append(path)
    image_prop['image_path'].append(os.path.join(path.split('/')[-2], path.split('/')[-1]))

# DataFrame 생성 및 병합
image_data = pd.DataFrame(image_prop)
image_data = image_data.merge(train_data, on='image_path')

def display_images(data, target):
    images = data[data['target'] == target]['path'].values
    len_data = len(images)
    rows = (len_data - 1) // 5 + 1
    fig, axs = plt.subplots(rows, 5, figsize=(20, 4*rows))
    axs = axs.flatten()
    
    for i, path in enumerate(images):
        img = Image.open(path)
        axs[i].imshow(img)
        axs[i].axis('off')
    
    for j in range(len_data, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'target_{target}_images.png'))
    plt.close()

# 모든 클래스에 대해 이미지 출력 및 저장
for target in image_data['target'].unique():
    print(f"Generating images for target {target}")
    display_images(image_data, target)

print("All images have been generated and saved.")
