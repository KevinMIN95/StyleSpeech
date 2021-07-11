from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from text import text_to_sequence
from utils import pad_1D, pad_2D, process_meta


def prepare_dataloader(data_path, filename, batch_size, shuffle=True, num_workers=2, meta_learning=False, seed=0):
    dataset = TextMelDataset(data_path, filename)
    if meta_learning:
        sampler = MetaBatchSampler(dataset.sid_to_indexes, batch_size, seed=seed)
    else:
        sampler = None
    shuffle = shuffle if sampler is None else None
    if meta_learning:
        loader = DataLoader(dataset, batch_sampler=sampler, 
                        collate_fn=dataset.collate_fn, num_workers=num_workers, pin_memory=True) 
    else:
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, 
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=num_workers) 
    return loader


def replace_outlier(values, max_v, min_v):
    values = np.where(values<max_v, values, max_v)
    values = np.where(values>min_v, values, min_v)
    return values


def norm_mean_std(x, mean, std):
    x = (x - mean) / std
    return x


class TextMelDataset(Dataset):
    def __init__(self, data_path, filename="train.txt",):
        self.data_path = data_path
        self.basename, self.text, self.sid = process_meta(os.path.join(data_path, filename))

        self.sid_dict = self.create_speaker_table(self.sid)

        with open(os.path.join(data_path, 'stats.json')) as f:
            data = f.read()
        stats_config = json.loads(data)
        self.f0_stat = stats_config["f0_stat"] # max, min, mean, std
        self.energy_stat = stats_config["energy_stat"] # max, min, mean, std
        
        self.create_sid_to_index()
        print('Speaker Num :{}'.format(len(self.sid_dict)))
    
    def create_speaker_table(self, sids):
        speaker_ids = np.sort(np.unique(sids))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_sid_to_index(self):
        _sid_to_indexes = {} 
        # for keeping instance indexes with the same speaker ids
        for i, sid in enumerate(self.sid):
            if sid in _sid_to_indexes:
                _sid_to_indexes[sid].append(i)
            else:
                _sid_to_indexes[sid] = [i]
        self.sid_to_indexes = _sid_to_indexes

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        sid = self.sid_dict[self.sid[idx]]
        phone = np.array(text_to_sequence(self.text[idx], []))
        mel_path = os.path.join(
            self.data_path, "mel", "libritts-mel-{}.npy".format(basename))
        mel_target = np.load(mel_path)
        D_path = os.path.join(
            self.data_path, "alignment", "libritts-ali-{}.npy".format(basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            self.data_path, "f0", "libritts-f0-{}.npy".format(basename))
        f0 = np.load(f0_path)
        f0 = replace_outlier(f0,  self.f0_stat[0], self.f0_stat[1])
        f0 = norm_mean_std(f0, self.f0_stat[2], self.f0_stat[3])
        energy_path = os.path.join(
            self.data_path, "energy", "libritts-energy-{}.npy".format(basename))
        energy = np.load(energy_path)
        energy = replace_outlier(energy, self.energy_stat[0], self.energy_stat[1])
        energy = norm_mean_std(energy, self.energy_stat[2], self.energy_stat[3])
        
        sample = {"id": basename,
                "sid": sid,
                "text": phone,
                "mel_target": mel_target,
                "D": D,
                "f0": f0,
                "energy": energy}
                
        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        sids = [batch[ind]["sid"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])
        
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + 1.)

        out = {"id": ids,
               "sid": np.array(sids),
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        output = self.reprocess(batch, index_arr)

        return output


class MetaBatchSampler():
    def __init__(self, sid_to_idx, batch_size, max_iter=100000, seed=0):
        # iterdict contains {sid: [idx1, idx2, ...]}
        np.random.seed(seed)

        self.sids = list(sid_to_idx.keys())
        np.random.shuffle(self.sids)

        self.sid_to_idx = sid_to_idx
        self.batch_size = batch_size
        self.max_iter = max_iter       
        
    def __iter__(self):
        for _ in range(self.max_iter):
            selected_sids = np.random.choice(self.sids, self.batch_size, replace=False)
            batch = []
            for sid in selected_sids:
                idx = np.random.choice(self.sid_to_idx[sid], 1)[0]
                batch.append(idx)

            assert len(batch) == self.batch_size
            yield batch

    def __len__(self):
        return self.max_iter

