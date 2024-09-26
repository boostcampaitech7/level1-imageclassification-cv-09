import pandas as pd
from collections import Counter

# 각 모델의 예측 결과 CSV 불러오기
caforemr = pd.read_csv('./output/caformer_b36.sail_in22k_ft_in1k/test_output.csv')  # 첫 번째 모델 결과
deit = pd.read_csv('./output/deit3_large_patch16_224.fb_in22k_ft_in1k/test_output.csv')  # 두 번째 모델 결과
swin = pd.read_csv('./output/swin_large_patch4_window7_224.ms_in22k_ft_in1k/test_output.csv')  # 세 번째 모델 결과
dinov2 = pd.read_csv('./output/dinov2_vitl14_reg_lc/test_output.csv')  # 네 번째 모델 결과

# 각 모델의 예측을 모아 앙상블하기
ensemble_results = []

for i in range(len(caforemr)):
    predictions = [
        caforemr.loc[i, 'target'], 
        deit.loc[i, 'target'], 
        swin.loc[i, 'target'], 
        dinov2.loc[i, 'target']
    ]
    
    # 다수결 투표
    final_prediction = Counter(predictions).most_common(1)[0][0]
    ensemble_results.append(final_prediction)

# 결과를 데이터프레임으로 저장
ensemble_df = caforemr[['ID', 'image_path']].copy()
ensemble_df['target'] = ensemble_results

# 최종 앙상블 결과를 CSV로 저장
ensemble_df.to_csv('ensemble_majority_voting.csv', index=False)
