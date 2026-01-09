import tqdm
from typing import Dict, List
from dataclasses import dataclass, field

from logger_config import logger

@dataclass
class ScoredDoc:
    qid: str
    pid: str
    rank: int
    score: float = field(default=-1)
    

def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    assert path.endswith('.txt')

    # qid -> pid -> score
    qrels = {}
    for line in open(path, 'r', encoding='utf-8'):
        qid, _, pid, score = line.strip().split('\t')
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][pid] = int(score)

    logger.info('Load {} queries {} qrels from {}'.format(len(qrels), sum(len(v) for v in qrels.values()), path))
    return qrels


def load_msmarco_predictions(path: str) -> Dict[str, List[ScoredDoc]]:
    assert path.endswith('.txt')

    qid_to_scored_doc = {}
    for line in tqdm.tqdm(open(path, 'r', encoding='utf-8'), desc='load prediction', mininterval=3):
        fs = line.strip().split('\t')
        qid, pid, rank = fs[:3]
        rank = int(rank)
        score = round(1 / rank, 4) if len(fs) == 3 else float(fs[3])

        if qid not in qid_to_scored_doc:
            qid_to_scored_doc[qid] = []
        scored_doc = ScoredDoc(qid=qid, pid=pid, rank=rank, score=score)
        qid_to_scored_doc[qid].append(scored_doc)

    qid_to_scored_doc = {qid: sorted(scored_docs, key=lambda sd: sd.rank)
                         for qid, scored_docs in qid_to_scored_doc.items()}

    logger.info('Load {} query predictions from {}'.format(len(qid_to_scored_doc), path))
    return qid_to_scored_doc


def save_preds_to_msmarco_format(preds: Dict[str, List[ScoredDoc]], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as writer:
        for qid in preds:
            for idx, scored_doc in enumerate(preds[qid]):
                writer.write('{}\t{}\t{}\t{}\n'.format(qid, scored_doc.pid, idx + 1, round(scored_doc.score, 3)))
    logger.info('Successfully saved to {}'.format(out_path))