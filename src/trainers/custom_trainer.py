import os
import torch
from typing import Optional, Dict, Tuple

from transformers.trainer import Trainer

from models import End2EndModel, End2EndOutput
from logger_config import logger
from metrics import accuracy, batch_mrr
from utils import AverageMeter

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    qry_batch_dict = {k[len('q_'):]: v for k, v in inputs.items() if k.startswith('q_')}
    psg_batch_dict = {k[len('p_'):]: v for k, v in inputs.items() if k.startswith('p_')}

    if not qry_batch_dict:
        qry_batch_dict = None
    if not psg_batch_dict:
        psg_batch_dict = None
    
    return qry_batch_dict, psg_batch_dict


class CustomTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(CustomTrainer, self).__init__(*pargs, **kwargs)
        self.model: End2EndModel

        self.qp_meter = AverageMeter('QP', round_digits=2)
        self.qt_meter = AverageMeter('QT', round_digits=2)

        self.last_epoch = 0
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        query, passage = _unpack_qp(inputs)
        outputs: End2EndOutput = model(query=query, passage=passage)
        loss = outputs.loss

        if self.model.training:
            self.qp_meter.update(outputs.qp_loss.item())
            self.qt_meter.update(outputs.qt_loss.item())
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float]=None) -> None:
        """
        Intercepts the trainer's logging call.
        Adds custom metrics from our meters to the logs, then resets the meters.
        """
        # Add the averaged values from your meters to the logs dictionary
        if self.model.training:
            logs['qp_loss'] = round(self.qp_meter.avg, 2)
            logs['qt_loss'] = round(self.qt_meter.avg, 2)

        # The parent `log` method handles the actual logging (e.g., to console, W&B, TensorBoard)
        super().log(logs)

        # Reset meters for the next logging window
        if self.model.training:
            self._reset_meters_if_needed()

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.qp_meter.reset()
            self.qt_meter.reset()
