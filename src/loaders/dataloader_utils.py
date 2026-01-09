from typing import Dict, List

def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]

def group_psg_ids(
        examples: Dict[str, List],
        train_n_passages: int,
        offset: int,
        use_first_positive: bool = False
    ) -> List[int]:
    pos_psg: List[str] = []
    positives: List[Dict[str, List]] = examples['positives']
    for idx, ex_pos in enumerate(positives):
        all_pos_psg_ids = ex_pos['contents']

        if use_first_positive:
            all_pos_psg = [
                psg for p_idx, psg in enumerate(all_pos_psg)
            ]
        
        cur_pos_psg_id = _slice_with_mod(all_pos_psg_ids, offset=offset, cnt=1)[0]
        pos_psg.append(cur_pos_psg_id)
    
    neg_psg: List[List[str]] = []
    negatives: List[Dict[str, List]] = examples['negatives']
    for ex_neg in negatives:
        cur_neg_psg = _slice_with_mod(ex_neg['contents'], offset=offset * train_n_passages, cnt=train_n_passages)
        cur_neg_psg = [psg for psg in cur_neg_psg]
        neg_psg.append(cur_neg_psg)
    
    assert len(pos_psg) == len(neg_psg), '{} != {}'.format(len(pos_psg), len(neg_psg))
    assert all(len(psg) == train_n_passages for psg in neg_psg)

    input_psg: List[str] = []
    for pos_psg_id, neg_ids in zip(pos_psg, neg_psg):
        input_psg.append(pos_psg_id)
        neg_ids = neg_ids[:train_n_passages-1]

        input_psg += neg_ids
        
    return input_psg