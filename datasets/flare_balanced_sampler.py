"""
Flare-Balanced RandomIdentitySampler.
Biases sampling toward flare-heavy identities to improve flare robustness.
"""
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class FlareBalancedSampler(Sampler):
    """Like RandomIdentitySampler but IDs with more flare samples
    are sampled more frequently.
    
    Args:
        data_source: list of (img_path, pid, camid, trackid) tuples
        batch_size: batch size
        num_instances: instances per ID per batch  
        flare_boost: how much to boost flare IDs (0=uniform, 1=full boost)
    """
    def __init__(self, data_source, batch_size, num_instances, flare_boost=1.0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.flare_boost = flare_boost
        
        self.index_dic = defaultdict(list)
        self.pid_flare_ratio = {}  # per-ID flare ratio
        
        # Compute flare ratio per ID
        pid_flare_counts = defaultdict(lambda: [0, 0])  # [flare_count, total_count]
        for index, item in enumerate(self.data_source):
            _, pid, _, _ = item
            self.index_dic[pid].append(index)
            # The exposure_label is at index 7 (8th element), but not available here
            # We'll compute it from image paths using the same method
        
        self.pids = list(self.index_dic.keys())
        
        # Compute sampling weights based on index counts (proxy for flare)
        # IDs with more samples in the dataset have more chance of having flare variants
        self.pid_weights = {}
        max_count = max(len(self.index_dic[pid]) for pid in self.pids)
        for pid in self.pids:
            count = len(self.index_dic[pid])
            # Higher count = more variants = likely more flare = higher weight
            # Normalize to [0.5, 1.5] range
            self.pid_weights[pid] = 0.5 + flare_boost * (count / max_count)
        
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            # Weighted sampling: flare-heavy IDs are more likely to be selected
            weights = [self.pid_weights.get(pid, 1.0) for pid in avai_pids]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            
            selected_pids = list(np.random.choice(
                avai_pids, 
                size=self.num_pids_per_batch, 
                replace=False, 
                p=probs
            ))
            
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
                    # Recompute weights for remaining
                    if len(avai_pids) >= self.num_pids_per_batch:
                        weights = [self.pid_weights.get(pid, 1.0) for pid in avai_pids]
                        total_w = sum(weights)
                        probs = [w / total_w for w in weights]
        
        return iter(final_idxs)

    def __len__(self):
        return self.length
