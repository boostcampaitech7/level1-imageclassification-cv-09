# Imagenet-Sketch Project

## Requirements
1. Install the required Python packages by running the following command:
  <br/>It is highly recommended to use a virtual environment `venv`.
   ```bash
   pip install -r requirements.txt
   ```
2. The imagenet_sketch dataset must be stored in the data folder. To download the dataset, execute following script:
    ```bash
    bash download_sketch_dataset.sh
    ```
## Usage
### Config file format
Config files are in `.yaml` format:
```yaml
exp_name: "base_experiment"
seed: 999
traindata_dir: './data/train'
testdata_dir: './data/test'

traindata_info_file: './data/train.csv'
testdata_info_file: './data/test.csv'

train_result_path: './train_result'
test_result_path: './output'

transform:
  transform_type: albumentations
  augmentations:
    - type: HorizontalFlip
      params:
        p: 0.5
    - type: Rotate
      params:
        limit: 30
        p: 0.7
    - type: RandomBrightnessContrast
      params:
        brightness_limit: 0.2
        contrast_limit: 0.2
        p: 0.5

device: 'cuda'

model_type: 'timm'
model_name: 'resnet18'
pretrained: true

loss: 'CELoss'

optimizer:
  type: Adam
  params:
    lr: 0.001

scheduler:
  type: StepLR
  params:
    step_size: 376
    gamma: 0.1

num_epochs: 5
batch_size: 64

```

Add addional configurations if you need.

### Training with config example
Modify the configurations in `.yaml` config files, then run:

  ```
  python train.py --config ./config/config.yaml
  ```

## How to test trained model?
### Test example
  ```
  # Output will be saved in ./{test_result_path}/{exp_name}/
  # valid only
  python test.py --config ./config/config.yaml --valid

  # test only
  python test.py --config ./config/config.yaml --test

  # all
  python test.py --config ./config/config.yaml --all
  ```

### Test output
The script will save the results in the `test_result_path` directory (as specified in the config).

- For validation, a summary of correct/incorrect predictions and overall accuracy will be printed, and the results will be saved as:
  - `valid_output.csv`: Full validation results.
  - `valid_incorrect_preds.csv`: Only incorrect predictions.

- For testing, the predictions will be saved in:
  - `test_output.csv`: Predictions for the test data.