import yaml
import os

class ConfigParser:
    def __init__(self, config_file: str):
        """
        YAML 파일에서 설정을 불러오는 파서.
        
        :param config_file: 설정 파일 경로 (yaml 형식)
        """
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: str) -> dict:
        """
        YAML 파일을 로드하여 Python 딕셔너리로 반환.
        
        :param config_file: 설정 파일 경로
        :return: Python 딕셔너리 형태의 설정 데이터
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} does not exist.")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)  # YAML 파일을 파싱하여 딕셔너리로 변환
        
        return config

    def get(self, key: str, default=None):
        """
        특정 설정 값을 가져옴.
        
        :param key: 설정의 키값
        :param default: 기본값 (설정에 해당 키가 없을 경우 반환될 값)
        :return: 설정된 값 또는 기본값
        """
        return self.config.get(key, default)

    def get_transform_config(self):
        """
        transform_config를 가져오는 헬퍼 메서드.
        
        :return: transform_config의 딕셔너리 값
        """
        return self.config.get('transform', {})
    
    # def get_loss_config(self):
    #     """
    #     loss 설정을 가져오는 헬퍼 메서드.
        
    #     :return: loss 설정의 딕셔너리 값
    #     """
    #     return self.config.get('loss', {})

    def get_optimizer_config(self):
        """
        optimizer 설정을 가져오는 헬퍼 메서드.
        
        :return: optimizer 설정의 딕셔너리 값
        """
        return self.config.get('optimizer', {})

    def get_scheduler_config(self):
        """
        scheduler 설정을 가져오는 헬퍼 메서드.
        
        :return: scheduler 설정의 딕셔너리 값
        """
        return self.config.get('scheduler', {})

    def save(self, output_file: str):
        """
        현재 설정을 YAML 파일로 저장.
        
        :param output_file: 저장할 파일 경로
        """
        with open(output_file, 'w') as f:
            yaml.dump(self.config, f)

# 예시 사용
if __name__ == "__main__":
    config_parser = ConfigParser("config.yaml")
    
    # 특정 설정 가져오기
    device = config_parser.get('device', 'cpu')
    print(f"Device: {device}")
    
    # transform_config 가져오기
    transform_config = config_parser.get_transform_config()
    print(f"Transform Config: {transform_config}")
    
    # # loss 설정 가져오기
    # loss_config = config_parser.get_loss_config()
    # print(f"Loss Config: {loss_config}")

    # optimizer 설정 가져오기
    optimizer_config = config_parser.get_optimizer_config()
    print(f"Optimizer Config: {optimizer_config}")
    
    # 스케줄러 설정 가져오기
    scheduler_config = config_parser.get_scheduler_config()
    print(f"Scheduler Config: {scheduler_config}")
    
    # 설정 업데이트 후 저장하기
    config_parser.save("updated_config.yaml")
