from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ActivityDetectionConfig:
    enabled: bool = True
    threshold: float = 0.85
    compare_size: Tuple[int, int] = (128, 128)


@dataclass
class YOLOConfig:
    enabled: bool = True
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.5


@dataclass
class PipelineConfig:
    activity_detection: ActivityDetectionConfig = field(
        default_factory=ActivityDetectionConfig
    )
    yolo: YOLOConfig = field(default_factory=YOLOConfig)


pipeline_config = PipelineConfig()
