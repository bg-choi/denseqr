import os
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers.activations import gelu
from transformers.modeling_outputs import ModelOutput
from transformers import  (
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
)

from config import Arguments
from logger_config import logger
from utils import full_contrastive_scores_and_labels

from .modeling_ance import ANCE

@dataclass
class End2EndOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    qp_loss: Optional[torch.Tensor] = None
    qt_loss: Optional[torch.Tensor] = None
    

class End2EndModel(nn.Module):
    def __init__(
            self,
            args: Arguments,
            retriever: ANCE,
            llm: AutoModelForMaskedLM,
            tokenizer
        ):
        super().__init__()

        self.args = args
        
        self.retriever: ANCE = retriever.eval()

        self.llm: AutoModelForMaskedLM = llm
        self.tokenizer = tokenizer

        self.cl_loss = nn.CrossEntropyLoss(reduction='mean')
        
        from trainers import CustomTrainer
        self.trainer: Optional[CustomTrainer] = None

    def forward(
            self,
            query: Dict[str, torch.Tensor]=None,
            passage: Dict[str, torch.Tensor]=None,
        ):

        p_inputs_embeds = self.retriever.roberta.embeddings.word_embeddings(passage['input_ids'])
        p_reps = self.retriever.doc_emb(
            inputs_embeds=p_inputs_embeds,
            attention_mask=passage['attention_mask']
        ).contiguous()

        q_inputs_embeds = self.llm.roberta(
            **{k: v for k, v in query.items()},
            output_hidden_states=True,
            return_dict=True
        ).last_hidden_state
        q_inputs_embeds = self.llm.lm_head.dense(q_inputs_embeds)
        q_inputs_embeds = self.llm.lm_head.layer_norm(gelu(q_inputs_embeds))

        retriever_embeds = self.retriever.roberta.embeddings.word_embeddings(query['input_ids'])
        q_inputs_embeds[query['input_ids']==0] = retriever_embeds[query['input_ids']==0]
        q_inputs_embeds[query['input_ids']==2] = retriever_embeds[query['input_ids']==2]
        
        q_reps = self.retriever.query_emb(
            inputs_embeds=q_inputs_embeds,
            attention_mask=query['attention_mask']
        ).contiguous()

        """ Contrastive loss: QP """
        all_qp_scores, all_qp_labels = self._compute_scores(q_reps, p_reps)
        all_qt_scores = self._compute_quantized_scores(q_reps, p_reps)

        qp_loss = self.cl_loss(all_qp_scores, all_qp_labels)
        qt_loss = self.cl_loss(all_qt_scores, all_qp_labels)
        
        loss = qp_loss + qt_loss
        
        return End2EndOutput(
            loss=loss, qp_loss=qp_loss, qt_loss=qt_loss
        )
    
    def _compute_scores(
            self,
            q_reps: torch.Tensor=None,
            p_reps: torch.Tensor=None,
            use_all_pairs: bool=True,
        ) -> Tuple:

        scores, labels = full_contrastive_scores_and_labels(
            query=q_reps,
            key=p_reps,
            use_all_pairs=use_all_pairs
        )

        return scores, labels
    
    def _compute_quantized_scores(
            self,
            q_reps: torch.Tensor=None,
            p_reps: torch.Tensor=None,
        ) -> Tuple:
        p_reps_per_q = p_reps.clone().view(q_reps.size(0), self.args.train_n_passages, -1) # (Q, P//Q, H)
        mean_p_reps = p_reps_per_q.mean(dim=1) # (Q, H)
        
        uq_reps = q_reps - mean_p_reps
        up_reps = p_reps - mean_p_reps.unsqueeze(1).repeat(1, self.args.train_n_passages, 1).view(p_reps.size(0), -1)

        scores, _ = full_contrastive_scores_and_labels(
            query=uq_reps,
            key=up_reps,
            use_all_pairs=True
        )
        
        return scores

    @classmethod
    def build(cls, args: Arguments, config, tokenizer):
        logger.info(f'loading RETRIEVER weights from {args.retriever_name_or_path}')
        retriever = ANCE.from_pretrained(args.retriever_name_or_path, config=config)
        retriever.requires_grad_(False)
        
        logger.info(f'loading TARGET weights from {args.model_name_or_path}')
        llm = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)

        model = cls(args=args, retriever=retriever, llm=llm, tokenizer=tokenizer)

        return model
    
    def save(self, output_dir: str):
        self.llm.save_pretrained(os.path.join(output_dir))


class End2EndModelForInference(End2EndModel):
    def __init__(
            self, args: Arguments,
            retriever: ANCE,
            llm: AutoModelForMaskedLM,
        ):
        nn.Module.__init__(self)
        self.args = args
        self.retriever: ANCE = retriever.eval()
        self.llm: AutoModelForMaskedLM = llm.eval()
    
    @torch.no_grad()
    def forward(self,
                query: Dict[str, torch.Tensor] = None,
                passage: Dict[str, torch.Tensor] = None):
        
        if query is None:
            q_reps = None
        else:
            q_inputs_embeds = self.llm.roberta(
                **{k: v for k, v in query.items()},
                output_hidden_states=True,
                return_dict=True
            ).last_hidden_state
            q_inputs_embeds = self.llm.lm_head.dense(q_inputs_embeds)
            q_inputs_embeds = self.llm.lm_head.layer_norm(gelu(q_inputs_embeds))

            retriever_embeds = self.retriever.roberta.embeddings.word_embeddings(query['input_ids'])
            q_inputs_embeds[query['input_ids']==0] = retriever_embeds[query['input_ids']==0]
            q_inputs_embeds[query['input_ids']==2] = retriever_embeds[query['input_ids']==2]
            
            q_reps = self.retriever.query_emb(
                inputs_embeds=q_inputs_embeds,
                attention_mask=query['attention_mask']
            )
        
        if passage is None:
            p_reps = None
        else:
            p_outputs = self.llm(**{k: v for k, v in passage.items()})
            p_reps = p_outputs.last_hidden_state[:, -1]

        return q_reps, p_reps
    
    @classmethod
    def build(cls, args: Arguments, config):
        logger.info(f'loading RETRIEVER weights from {args.retriever_name_or_path}')
        retriever = ANCE.from_pretrained(args.retriever_name_or_path, config=config)
        retriever.requires_grad_(False)
        
        logger.info(f'loading TARGET weights from {args.model_name_or_path}')
        llm = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
        llm.requires_grad_(False)

        model = cls(args=args, retriever=retriever, llm=llm)

        return model