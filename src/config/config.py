from typing import Dict, Tuple

from src.config import cookpad
from src.config.base_configs import TrainConfig, EvalConfig


def get_config(dataset: str, dataset_id: int, model_name: str, epochs: int, docs: Dict) -> Tuple[
    TrainConfig, EvalConfig]:
    if dataset == 'cookpad':
        from src.data.cookpad.preprocessors import ConcatDataProcessor
        data_processor = ConcatDataProcessor(docs)
        train_config, eval_config = {
            'ebr': cookpad.ebr_config,
            'naive': cookpad.naive_config,
            'nrmf_simple_query': cookpad.nrmf_simple_query_config,
            'nrmf_simple_all': cookpad.nrmf_simple_all_config,
            'nrmf_simple_query_with_1st': cookpad.nrmf_simple_query_with_1st_config,
            'fwfm_query': cookpad.fwfm_query_config,
            'fwfm_all': cookpad.fwfm_all_config,
            'fwfm_selected': cookpad.fwfm_selected_config,
            'fwfm_all_without_1st': cookpad.fwfm_all_without_1st_config,
        }[model_name](f'{dataset}.{dataset_id}', epochs, data_processor)
    else:
        raise ValueError(f"Invalid dataset type is given: {dataset}")

    return train_config, eval_config
