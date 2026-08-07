"""
ELA Analysis Source Package

Energy Landscape Analysis (ELA) のソースコードパッケージ
"""

from .data_processing import DataProcessor
from .ela_pipeline import ELAPipeline
from .fixed_w_pipeline import FixedWPipeline
from .batch_processor import BatchProcessor, BatchJob, BatchConfig
from .config import ELAConfig, ConfigManager, create_default_config
from .logging_utils import ELALogger, create_logger_from_args
from .execution_history import ExecutionHistoryManager, ExecutionRecord, create_execution_summary
from .hj_analysis import (
    DEFAULT_BACKGROUND_ITEMS_0B,
    DEFAULT_TARGET_ITEMS_0B,
    HjPaperOutputDirs,
    MCMCAnalysisResult,
    build_edge_dataframe,
    compute_jij_embedding_outputs,
    create_direct_edge_detail,
    create_embedding_group_summary,
    default_paper_output_dirs,
    configure_paper_plot_style,
    run_embedding_permutation_tests,
    run_subject_mcmc_analysis,
    save_hj_analysis_csvs,
    save_hj_analysis_figures,
    to_one_based_items,
)

__version__ = "1.0.0"
__author__ = "ELA Analysis Team"

__all__ = [
    "DataProcessor",
    "ELAPipeline",
    "FixedWPipeline",
    "BatchProcessor",
    "BatchJob",
    "BatchConfig",
    "ELAConfig",
    "ConfigManager",
    "create_default_config",
    "ELALogger",
    "create_logger_from_args",
    "ExecutionHistoryManager",
    "ExecutionRecord",
    "create_execution_summary",
    "DEFAULT_BACKGROUND_ITEMS_0B",
    "DEFAULT_TARGET_ITEMS_0B",
    "to_one_based_items",
    "build_edge_dataframe",
    "compute_jij_embedding_outputs",
    "create_direct_edge_detail",
    "create_embedding_group_summary",
    "run_embedding_permutation_tests",
    "HjPaperOutputDirs",
    "default_paper_output_dirs",
    "configure_paper_plot_style",
    "save_hj_analysis_csvs",
    "save_hj_analysis_figures",
    "MCMCAnalysisResult",
    "run_subject_mcmc_analysis",
]
