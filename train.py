import torch, gc
import numpy as np
from data_loader.dataset import CustomDataset
from data_loader.transform import TransformSelector
from model.model import ModelSelector
from model import loss
import torch.nn as nn
from trainer.trainer import Trainer, LoRATrainer, MPTrainer
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
import argparse
from config_parser import ConfigParser
import random
import peft

def main(config, config_path):
    device = config['device']
    gc.collect()
    torch.cuda.empty_cache()
    # Load datasets
    traindata_dir = config['traindata_dir']
    traindata_info_file = config['traindata_info_file']
    train_result_path = config['train_result_path']
    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    train_info = pd.read_csv(traindata_info_file)
    # 총 class의 수를 측정.
    num_classes = len(train_info['target'].unique())

    # 각 class별로 8:2의 비율이 되도록 학습과 검증 데이터를 분리.
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=config['seed'],
    )

    # 학습에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = config['transform']['transform_type'],
        transform_config = config['transform']["augmentations"]
    )
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # 학습에 사용할 Dataset을 선언.
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    # 학습에 사용할 Model을 선언.
    model_selector = ModelSelector(model_type=config['model_type'], model_name=config['model_name'], num_classes=num_classes, pretrained=config['pretrained'])
    model = model_selector.get_model().to(device)


    # peft, LoRA fine tuning
    if config.get('lora') and config['lora']['use']:
        
        # for name, module in [(n, type(m)) for n, m in model.named_modules()][100:]:
        #     print(f"Name: {name}, Type: {module}")    

        # print([(n, type(m)) for n, m in model.named_modules()][-10:])
        # exit()

        # target_classes = (torch.nn.modules.linear.Linear, torch.nn.modules.conv.Conv2d)
        # target_modules = [name for name, module in model.named_modules() if isinstance(module, target_classes) and not 'head' in name]
        # save_modules = [name for name, module in model.named_modules() if isinstance(module, target_classes) and 'head' in name]

        lora_config = peft.LoraConfig(**config['lora']['params'])
        peft_model = peft.get_peft_model(model, lora_config).to(device)
        optimizer = getattr(optim, config['optimizer']['type'])(peft_model.parameters(), **config['optimizer']['params'])
        scheduler = getattr(optim.lr_scheduler, config['scheduler']['type'])(optimizer, **config['scheduler']['params'])
        peft_model.print_trainable_parameters()

        loss_fn = getattr(loss, config['loss'])()

        # 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
        trainer = LoRATrainer(
            model=peft_model, 
            device=device, 
            train_loader=train_loader,
            val_loader=val_loader, 
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn, 
            epochs=config['num_epochs'],
            result_path=train_result_path,
            exp_name=config['exp_name'],
            config_path=config_path,
        )
        
        # 모델 학습.
        trainer.train()

    # general training
    else:
        # 학습에 사용할 optimizer를 선언
        optimizer = getattr(optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['params'])
        # 스케줄러 선언
        scheduler = getattr(optim.lr_scheduler, config['scheduler']['type'])(optimizer, **config['scheduler']['params'])
        # 학습에 사용할 Loss를 선언.

        loss_fn = getattr(loss, config['loss'])()

        # 앞서 선언한 필요 class와 변수들을 조합해, 학습을 진행할 Trainer를 선언. 
        if config['MPTrainer'].get() and config['MPTrainer']==True:
            trainer = MPTrainer(
                model=model, 
                device=device, 
                train_loader=train_loader,
                val_loader=val_loader, 
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn, 
                epochs=config['num_epochs'],
                result_path=train_result_path,
                exp_name=config['exp_name'],
                config_path=config_path,
            )

        else:
            trainer = Trainer(
                model=model, 
                device=device, 
                train_loader=train_loader,
                val_loader=val_loader, 
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn, 
                epochs=config['num_epochs'],
                result_path=train_result_path,
                exp_name=config['exp_name'],
                config_path=config_path,
            )
        # 모델 학습.
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the configuration YAML file")

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
    
    main(config = config, config_path = args.config)