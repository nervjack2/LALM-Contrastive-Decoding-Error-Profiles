from torch.utils.data import Dataset

from src.utils.load import get_class_in_module


SRC_DIR = "src/tasks"

def get_test_task(name) -> Dataset:
    if name.startswith("sakura"):
        from .sakura import SAKURASequence
        tmp = name.split('-')
        subject = tmp[0].split('_')[1]

        from .sakura import SAKURASequence
        return SAKURASequence(
            subject=subject,
            multi=('m' in tmp),
            judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
        )
    elif name.startswith("mmau-test-mini"):
        from .mmau import MMAUMINIMCQASequence
        tmp = name.split('-')
        return MMAUMINIMCQASequence(
            judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
        )
    elif name.startswith("mmar"):
        from .mmar import MMARMCQASequence
        tmp = name.split('-')
        return MMARMCQASequence(
            judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
        )
    else:
        raise ValueError(f"Unknown task name: {name}")
