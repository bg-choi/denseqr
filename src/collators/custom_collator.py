import re
import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding

from config import Arguments


@dataclass
class CustomCollator(DataCollatorWithPadding):
    args: Arguments = None

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        
        qry_inputs = []
        psg_inputs = []
        for f in features:
            p_input_ids = f.get('p_input_ids', [])
            for pids in p_input_ids:
                psg_inputs.append(pids)
            
            qry_input_ids = f.get('q_input_ids', [])
            qry_inputs.append(qry_input_ids)
        
        q_collated = self.tokenizer.pad(
            {'input_ids': qry_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        p_collated = self.tokenizer.pad(
            {'input_ids': psg_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )

        for k in list(q_collated.keys()):
            q_collated['q_' + k] = q_collated[k]
            del q_collated[k]
        for k in p_collated:
            q_collated['p_' + k] = p_collated[k]

        merged_batch_dict = q_collated
        labels = torch.zeros(len(q_collated['q_input_ids']), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        return merged_batch_dict
    

@dataclass
class CustomCollatorForIndex(DataCollatorWithPadding):
    args: Arguments = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        psg_inputs = []
        for f in features:
            p_input_ids = f.get('input_ids', [])
            psg_inputs.append(p_input_ids)

        p_collated = self.tokenizer.pad(
            {'input_ids': psg_inputs},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        return p_collated


@dataclass
class CustomCollatorForQuery(DataCollatorWithPadding):
    args: Arguments = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        qry_inputs = []
        for f in features:
            qry_input_ids = f.get('q_input_ids', [])
            qry_inputs.append(qry_input_ids)

        q_collated = self.tokenizer.pad(
            {'input_ids': qry_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        
        return q_collated