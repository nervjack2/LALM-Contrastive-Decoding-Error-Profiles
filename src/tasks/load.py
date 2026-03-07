from torch.utils.data import Dataset

from src.utils.load import get_class_in_module


SRC_DIR = "src/tasks"

def get_test_task(name) -> Dataset:
    if name.startswith("sakura"):
        from .sakura import SAKURASequence
        tmp = name.split('-')
        subject = tmp[0].split('_')[1]

        if 'ord' in tmp:
            from .sakura_c import FixOrderSequence
            return FixOrderSequence(
                subject=subject,
                reasoning=('r' in tmp)
            )
        else:
            from .sakura import SAKURASequence
            return SAKURASequence(
                subject=subject,
                multi=('m' in tmp),
                reasoning=('r' in tmp),
                llm_judge=(('ja' in tmp)or('jl' in tmp)),
                judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
                prompt_mode = 'cot-a' if 'pca' in tmp else ('cot' if 'pc' in tmp else ('regular-a' if 'pra' in tmp else ('regular' if 'pr' in tmp else ('direct' if 'pd' in tmp else ''))))
            )
    elif name.startswith("mmau-test-mini"):
        from .mmau import MMAUMINIMCQASequence
        tmp = name.split('-')
        return MMAUMINIMCQASequence(
            reasoning=('r' in tmp),
            llm_judge=(('ja' in tmp)or('jl' in tmp)),
            judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
            prompt_mode = 'cot-a' if 'pca' in tmp else ('cot' if 'pc' in tmp else ('regular-a' if 'pra' in tmp else ('regular' if 'pr' in tmp else ('direct' if 'pd' in tmp else ''))))
        )
    elif name.startswith("mmar"):
        from .mmar import MMARMCQASequence
        tmp = name.split('-')
        return MMARMCQASequence(
            reasoning=('r' in tmp),
            llm_judge=(('ja' in tmp)or('jl' in tmp)),
            judge_mode='api' if 'ja' in tmp else ('local' if 'jl' in tmp else ''),
            prompt_mode = 'cot-a' if 'pca' in tmp else ('cot' if 'pc' in tmp else ('regular-a' if 'pra' in tmp else ('regular' if 'pr' in tmp else ('direct' if 'pd' in tmp else ''))))
        )
    else:
        raise ValueError(f"Unknown task name: {name}")
