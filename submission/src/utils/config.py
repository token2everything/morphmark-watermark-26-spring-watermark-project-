"""配置系统：YAML 配置解析 + 类型校验"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class WatermarkConfig:
    type: str                       # "kgw" | "morphmark_exp" | "morphmark_linear" | "morphmark_log"
    key: int                        # private key for hashing
    gamma: float = 0.5              # green list ratio
    delta: Optional[float] = None   # KGW fixed delta
    p0: Optional[float] = None      # MorphMark threshold
    k: Optional[float] = None       # MorphMark growth parameter
    epsilon: float = 1e-10          # MorphMark minimal strength
    prefix_length: int = 1          # context window for hash
    f_scheme: str = "time"          # hash scheme
    z_threshold: float = 4.0        # detection threshold


@dataclass
class ModelConfig:
    name: str = "facebook/opt-1.3b"
    device: str = "cuda"
    torch_dtype: str = "float16"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    min_new_tokens: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    no_repeat_ngram_size: int = 4
    prompt_tokens: int = 30


@dataclass
class DetectionConfig:
    method: str = "zscore"          # "zscore" | "ewd"
    gamma: float = 0.5


@dataclass
class DataConfig:
    dataset: str = "c4"             # "c4" | "wmt16" | "cnn_dailymail" | "humaneval"
    num_samples: int = 400
    seed: int = 42


@dataclass
class ExperimentConfig:
    watermark: WatermarkConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> ExperimentConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    wm_raw = raw['watermark']
    watermark = WatermarkConfig(
        type=wm_raw['type'],
        key=wm_raw.get('key', 15485863),
        gamma=wm_raw.get('gamma', 0.5),
        delta=wm_raw.get('delta'),
        p0=wm_raw.get('p0'),
        k=wm_raw.get('k'),
        epsilon=wm_raw.get('epsilon', 1e-10),
        prefix_length=wm_raw.get('prefix_length', 1),
        f_scheme=wm_raw.get('f_scheme', 'time'),
        z_threshold=wm_raw.get('z_threshold', 4.0),
    )

    model_raw = raw.get('model', {})
    model = ModelConfig(
        name=model_raw.get('name', 'facebook/opt-1.3b'),
        device=model_raw.get('device', 'cuda'),
        torch_dtype=model_raw.get('torch_dtype', 'float16'),
    )

    gen_raw = raw.get('generation', {})
    generation = GenerationConfig(
        max_new_tokens=gen_raw.get('max_new_tokens', 200),
        min_new_tokens=gen_raw.get('min_new_tokens', 0),
        temperature=gen_raw.get('temperature', 0.7),
        top_p=gen_raw.get('top_p', 0.9),
        top_k=gen_raw.get('top_k', 0),
        do_sample=gen_raw.get('do_sample', True),
        no_repeat_ngram_size=gen_raw.get('no_repeat_ngram_size', 4),
        prompt_tokens=gen_raw.get('prompt_tokens', 30),
    )

    det_raw = raw.get('detection', {})
    detection = DetectionConfig(
        method=det_raw.get('method', 'zscore'),
        gamma=det_raw.get('gamma', 0.5),
    )

    data_raw = raw.get('data', {})
    data = DataConfig(
        dataset=data_raw.get('dataset', 'c4'),
        num_samples=data_raw.get('num_samples', 400),
        seed=data_raw.get('seed', 42),
    )

    return ExperimentConfig(
        watermark=watermark,
        model=model,
        generation=generation,
        detection=detection,
        data=data,
    )
