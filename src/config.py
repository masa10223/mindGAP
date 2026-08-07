"""
設定管理モジュール

ELA解析の設定を管理し、OSS対応のための柔軟な設定システムを提供します。
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class QuestionnaireType(Enum):
    """質問票タイプの列挙型"""
    PHQ9_HQ25_SOCIAL_INTEGRATED = "PHQ9+HQ25_social_integrated"
    PHQ9_HQ25_ISOLATION_INTEGRATED = "PHQ9+HQ25_isolation_integrated"
    PHQ9_HQ25_ISOLATION_SOCIAL_INTEGRATED = "PHQ9+HQ25_Isolation+social_integrated"
    PHQ9_ALL = "PHQ9_all"
    HQ25_SOCIAL_ALL = "HQ25_social_all"
    HQ25_ISOLATION_ALL = "HQ25_Isolation_all"


class HQ25Threshold(Enum):
    """HQ-25閾値の列挙型"""
    TWO = 2
    THREE = 3


@dataclass
class DataConfig:
    """データ設定"""
    data_dir: str = "data"
    csv_dir: str = "outputs/csvs"
    output_dir: str = "outputs"
    backup_dir: str = "backups"
    
    def __post_init__(self):
        """パスの正規化（ELA_analysis内の相対パスとして処理）"""
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.data_dir = str(ela_analysis_dir / self.data_dir)
        self.csv_dir = str(ela_analysis_dir / self.csv_dir)
        self.output_dir = str(ela_analysis_dir / self.output_dir)
        self.backup_dir = str(ela_analysis_dir / self.backup_dir)


@dataclass
class ProcessingConfig:
    """処理設定"""
    hq25_threshold: int = 3
    device_id: int = 0
    batch_size: int = 1000
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    verbose: bool = False
    
    def __post_init__(self):
        """設定値の検証"""
        if self.hq25_threshold not in [2, 3]:
            raise ValueError("hq25_threshold must be 2 or 3")
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")


@dataclass
class VisualizationConfig:
    """可視化設定"""
    save_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    color_scheme: str = "viridis"
    show_plots: bool = False


@dataclass
class LoggingConfig:
    """ログ設定"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_console_logging: bool = True


@dataclass
class ELAConfig:
    """ELA設定の統合クラス"""
    data: DataConfig
    processing: ProcessingConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    
    def __init__(self, 
                 data_config: Optional[DataConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None,
                 visualization_config: Optional[VisualizationConfig] = None,
                 logging_config: Optional[LoggingConfig] = None):
        """
        ELA設定の初期化
        
        Args:
            data_config: データ設定
            processing_config: 処理設定
            visualization_config: 可視化設定
            logging_config: ログ設定
        """
        self.data = data_config or DataConfig()
        self.processing = processing_config or ProcessingConfig()
        self.visualization = visualization_config or VisualizationConfig()
        self.logging = logging_config or LoggingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return {
            'data': asdict(self.data),
            'processing': asdict(self.processing),
            'visualization': asdict(self.visualization),
            'logging': asdict(self.logging)
        }
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """設定をJSONファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """設定をYAMLファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ELAConfig':
        """辞書から設定を作成"""
        return cls(
            data_config=DataConfig(**config_dict.get('data', {})),
            processing_config=ProcessingConfig(**config_dict.get('processing', {})),
            visualization_config=VisualizationConfig(**config_dict.get('visualization', {})),
            logging_config=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'ELAConfig':
        """JSONファイルから設定を読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'ELAConfig':
        """YAMLファイルから設定を読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> 'ELAConfig':
        """環境変数から設定を作成"""
        config = cls()
        
        # 環境変数からの設定読み込み
        if 'ELA_DATA_DIR' in os.environ:
            config.data.data_dir = os.environ['ELA_DATA_DIR']
        if 'ELA_OUTPUT_DIR' in os.environ:
            config.data.output_dir = os.environ['ELA_OUTPUT_DIR']
        if 'ELA_HQ25_THRESHOLD' in os.environ:
            config.processing.hq25_threshold = int(os.environ['ELA_HQ25_THRESHOLD'])
        if 'ELA_DEVICE_ID' in os.environ:
            config.processing.device_id = int(os.environ['ELA_DEVICE_ID'])
        if 'ELA_VERBOSE' in os.environ:
            config.processing.verbose = os.environ['ELA_VERBOSE'].lower() == 'true'
        
        return config
    
    def create_directories(self) -> None:
        """必要なディレクトリを作成"""
        directories = [
            self.data.data_dir,
            self.data.csv_dir,
            self.data.output_dir,
            self.data.backup_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """設定の妥当性を検証"""
        # ディレクトリの存在確認
        if not Path(self.data.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {self.data.data_dir}")
        
        # 設定値の検証
        self.processing.__post_init__()
        
        # 出力ディレクトリの作成
        self.create_directories()


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        設定管理の初期化
        
        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = config_file
        self._config: Optional[ELAConfig] = None
    
    def load_config(self) -> ELAConfig:
        """設定の読み込み"""
        if self._config is not None:
            return self._config
        
        # 設定ファイルが指定されている場合
        if self.config_file and Path(self.config_file).exists():
            config_path = Path(self.config_file)
            if config_path.suffix.lower() == '.json':
                self._config = ELAConfig.from_json(config_path)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                self._config = ELAConfig.from_yaml(config_path)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        else:
            # デフォルト設定または環境変数から読み込み
            self._config = ELAConfig.from_env()
        
        # 設定の検証
        self._config.validate()
        return self._config
    
    def save_config(self, config: ELAConfig, 
                   filepath: Union[str, Path],
                   format: str = 'json') -> None:
        """
        設定の保存
        
        Args:
            config: 保存する設定
            filepath: 保存先ファイルパス
            format: 保存形式 ('json' または 'yaml')
        """
        if format.lower() == 'json':
            config.to_json(filepath)
        elif format.lower() in ['yml', 'yaml']:
            config.to_yaml(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_config(self) -> ELAConfig:
        """設定の取得"""
        return self.load_config()


# デフォルト設定の作成
def create_default_config() -> ELAConfig:
    """デフォルト設定の作成"""
    return ELAConfig()


def create_example_config_file(filepath: Union[str, Path] = "ela_config.json") -> None:
    """例示用設定ファイルの作成"""
    config = create_default_config()
    config.to_json(filepath)
    print(f"Example config file created: {filepath}")


# 環境変数による設定の例
ENV_VAR_EXAMPLES = {
    'ELA_DATA_DIR': '/path/to/data',
    'ELA_OUTPUT_DIR': '/path/to/output',
    'ELA_HQ25_THRESHOLD': '3',
    'ELA_DEVICE_ID': '0',
    'ELA_VERBOSE': 'true'
}
