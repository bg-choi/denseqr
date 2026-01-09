import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Tuple, Dict, List

from transformers import PreTrainedTokenizerFast, Trainer
from transformers.file_utils import PaddingStrategy

from config import Arguments
from logger_config import logger
from .dataloader_utils import group_psg_ids

class CustomDataLoader:
    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer

        # corpus_path = os.path.join(args.data_dir, 'collection', 'passages_sample.jsonl')
        # self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset = self._get_transformed_datasets()

        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)

        passage_lists = group_psg_ids(
            examples=examples,
            train_n_passages=self.args.train_n_passages,
            offset=current_epoch,
            use_first_positive=self.args.use_first_positive
        )
        # passage_lists: List[str] = [self.corpus[psg_id]['contents'].lower() for psg_id in input_psg_ids]

        psg_batch_dict = self.tokenizer(
            passage_lists,
            max_length=self.args.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_attention_mask=False,
        )

        all_qry_inputs = []
        for ctx, qry in zip(examples['context'], examples['query']):
            masks = "".join("<mask>" for i in range(self.args.n_masks))
            context = " ".join(c for c in reversed(ctx))
            qry_input_text = f'{qry.lower()}{masks} </s> {context.lower()}'
            all_qry_inputs.append(qry_input_text)
        
        qry_batch_dict = self.tokenizer(
            all_qry_inputs,
            padding=PaddingStrategy.DO_NOT_PAD,
            max_length=self.args.q_max_len,
            truncation=True,
            return_attention_mask=False,
        )

        merged_dict = {'q_{}'.format(k): v for k, v in qry_batch_dict.items()}
        
        step_size = self.args.train_n_passages
        for k, v in psg_batch_dict.items():
            k = 'p_{}'.format(k)
            merged_dict[k] = []
            for idx in range(0, len(v), step_size):
                merged_dict[k].append(v[idx:(idx + step_size)])
        
        return merged_dict


    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files['train'] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files['validation'] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset = None

        if self.args.do_train:
            if 'train' not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets['train']
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)
        
        return train_dataset