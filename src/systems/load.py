import typing

from src.utils.load import get_class_in_module
from .base import System


SRC_DIR = "src/systems"

BASIC = {
    "pma": (f"{SRC_DIR}/pma.py", "PMASystem"),
    "af3-aad": (f"{SRC_DIR}/af3_aad.py", "AF3AADSystem"),
    "af3-amti": (f"{SRC_DIR}/af3_amti.py", "AF3AMTISystem"),
    "af3-dola": (f"{SRC_DIR}/af3_dola.py", "AF3DoLASystem"),
    "af3-acd": (f"{SRC_DIR}/af3_acd.py", "AF3ACDSystem"),
}

QWEN = {
    "qwen": (f"{SRC_DIR}/qwen2_5_omni/qwen2_5_omni.py", "Qwen2_5OmniSystem"),
    "qwen-aad": (f"{SRC_DIR}/qwen2_5_omni/aad.py", "AADSystem"),
    "qwen-acd": (f"{SRC_DIR}/qwen2_5_omni/acd.py", "ACDSystem"),
    "qwen-amti": (f"{SRC_DIR}/qwen2_5_omni/amti.py", "AMTISystem"),
    "qwen-dola": (f"{SRC_DIR}/qwen2_5_omni/dola.py", "DoLASystem"),
}

DESTA = {
    "desta": (f"{SRC_DIR}/desta2_5/desta2_5.py", "Desta2_5System"),
    "desta-official": (f"{SRC_DIR}/desta2_5/desta2_5.py", "OfficialSystem"),
    "desta-aad": (f"{SRC_DIR}/desta2_5/aad.py", "AADSystem"),
    "desta-acd": (f"{SRC_DIR}/desta2_5/acd.py", "ACDSystem"),
    "desta-amti": (f"{SRC_DIR}/desta2_5/amti.py", "AMTISystem"),
    "desta-dola": (f"{SRC_DIR}/desta2_5/dola.py", "DoLASystem"),
}

AUDIO_FLAMINGO = {
    "af3": (f"{SRC_DIR}/audio_flamingo_3/audio_flamingo_3.py", "AudioFlamingo3System"),
    "af3-aad": (f"{SRC_DIR}/audio_flamingo_3/aad.py", "AADSystem"),
    "af3-acd": (f"{SRC_DIR}/audio_flamingo_3/acd.py", "ACDSystem"),
    "af3-amti": (f"{SRC_DIR}/audio_flamingo_3/amti.py", "AMTISystem"),
    "af3-dola": (f"{SRC_DIR}/audio_flamingo_3/dola.py", "DoLASystem"),
}

SYSTEM_MAPPING = {
    **QWEN,
    **DESTA,
    **AUDIO_FLAMINGO
}


def get_system_cls(name: str) -> typing.Type[System]:
    module_path, class_name = SYSTEM_MAPPING[name]
    return get_class_in_module(class_name, module_path)


def load_system(system_name: str, system_config: dict, checkpoint: str=None) -> System:
    system_cls = get_system_cls(system_name)
    if checkpoint is None:
        return system_cls(system_config)
    print(f'Load from {checkpoint}...')
    system = get_system_cls(system_name)(system_config)
    system.load_checkpoint(checkpoint)
    return system
