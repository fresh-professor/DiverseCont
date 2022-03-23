import os
from copy import deepcopy
import tqdm
import torch
import torch.nn.functional as F
import colorful
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter
from .reservoir import reservoir
from components import Net
from utils import BetaMixture1D

class DiverseCont(torch.nn.Module):
    """ Train Continual Model self-supervisedly
        Freeze when required to eval and finetune supervisedly using Purified Buffer.
    """
    def __init__(self, config, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.writer = writer

        self.diverse_buffer = reservoir['diverse'](config, config['diverse_buffer_size'], config['diverse_buffer_q_poa'])
        self.delay_buffer = reservoir['delay'](config, config['delayed_buffer_size'], config['delayed_buffer_q_poa'])

        self.expert_step = 0
        self.base_step = 0
        self.base_ft_step = 0

        self.base = self.get_init_base(config)

    def get_init_base(self, config):
        """get initialized base model"""
        base = Net[config['net']](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        lr_scheduler_config['options'].update({'T_max': config['base_train_epochs']})

        base.setup_optimizer(optim_config)
        base.setup_lr_scheduler(lr_scheduler_config)
        return base

    def get_init_base_ft(self, config):
        """get initialized eval model"""
        base_ft = Net[config['net'] + '_ft'](config)
        optim_config = config['optimizer_ft']
        lr_scheduler_config = config['lr_scheduler_ft']

        base_ft.setup_optimizer(optim_config)
        base_ft.setup_lr_scheduler(lr_scheduler_config)
        return base_ft

    def learn(self, x, y, corrupt, idx, step=None):
        x, y = x.cuda(), y.cuda()
        for i in range(len(x)):
            self.delay_buffer.update(imgs=x[i: i + 1], cats=y[i: i + 1], corrupts=corrupt[i: i + 1], idxs=idx[i: i + 1])

            if self.delay_buffer.is_full():
                self.train_self_base()
                diverse_idx, clean_p = self.cluster_and_sample()
                #self.update_purified_buffer(clean_idx, clean_p, step)
                imgs, cats, corrupts, idxs, clean_p = self.cluster_and_sample()
                self.update_purified_buffer_full_info(imgs, cats, corrupts, idxs, clean_p, step)

    def update_purified_buffer(self, clean_idx, clean_p, step):
        """
        update purified buffer with the filtered samples
        based on index
        """
        self.purified_buffer.update(
            imgs=self.delay_buffer.get('imgs')[clean_idx],
            cats=self.delay_buffer.get('cats')[clean_idx],
            corrupts=self.delay_buffer.get('corrupts')[clean_idx],
            idxs=self.delay_buffer.get('idxs')[clean_idx],
            clean_ps=clean_p)

        self.delay_buffer.reset()
        print(colorful.bold_yellow(self.purified_buffer.state('corrupts')).styled_string)
        self.writer.add_scalar(
            'buffer_corrupts', torch.sum(self.purified_buffer.get('corrupts')), step)

    def update_purified_buffer_full_info(self, imgs, cats, corrupts, idxs, clean_p, step):
        """
        update purified buffer with the filtered samples
        based on full_information
        """
        self.purified_buffer.update(
            imgs=imgs,
            cats=cats,
            corrupts=corrupts,
            idxs=idxs,
            clean_ps=clean_p)

        self.delay_buffer.reset()
        print(colorful.bold_yellow(self.purified_buffer.state('corrupts')).styled_string)
        self.writer.add_scalar(
            'buffer_corrupts', torch.sum(self.purified_buffer.get('corrupts')), step)

    def cluster_and_sample(self):
        """filter samples in delay buffer"""
        self.base.eval()
        with torch.no_grad():
            xs = self.delay_buffer.get('imgs')
            ys = self.delay_buffer.get('cats')
            corrs = self.delay_buffer.get('corrupts')
            idxs = self.delay_buffer.get('idxs')

            # merge delay_buffer and diverse_buffer
            if self.diverse_buffer.is_full():
                divb_xs = self.diverse_buffer.get('imgs')
                divb_ys = self.diverse_buffer.get('cats')
                divb_corrs = self.diverse_buffer.get('corrupts')
                divb_idxs = self.diverse_buffer.get('idxs')

                xs = torch.cat((xs, divb_xs), dim=0)
                ys = torch.cat((ys, divb_ys), dim=0)
                corrs = torch.cat((corrs, divb_corrs), dim=0)
                idxs = torch.cat((idxs, divb_idxs), dim=0)

            features = self.base(xs)
            features = F.normalize(features, dim=1)

            clean_p = list()
            clean_idx = list()

            imgs = torch.Tensor().cuda()
            cats = torch.Tensor().cuda()
            corrupts = torch.Tensor()
            indexs = torch.Tensor()
            print("*" * 30)
            # Find the proper number of y class
            divb_size = self.diverse_buffer.rsvr_total_size

            # sorting unique_ys based on its counts
            unique_ys, unique_ys_count= torch.unique(ys, return_counts = True)
            unique_ys = unique_ys[torch.sort(unique_ys_count)[1]]
            unique_ys_count = torch.sort(unique_ys_count)[0]

            # get the proper_size (balanaced sampling, diverse_buffer.size/num_class)
            proper_size = int(divb_size / unique_ys.shape[0])
            print(f"Proper_size per class = {proper_size}")

            print(f"Start clustering per class")
            for u_y, u_y_c in zip(unique_ys, unique_ys_count):
                print(f"class {u_y}, count {u_y_c}")
                y_mask = (ys == u_y)

                x = xs[y_mask]
                y = ys[y_mask]
                corr = corrs[y_mask]
                idx = idxs[y_mask]

                print("*" * 30)
                if (proper_size >= u_y_c) :
                    print(f"Iterative Sample Selection (just pass dataset (not enough data))")
                    # pass all samples to purified buffer
                    clean_idx.extend(torch.nonzero(y_mask)[:, -1].tolist())
                    clean_p.extend(torch.zeros_like(torch.nonzero(y_mask)[:, -1]).tolist())
                else :
                    feature = features[y_mask]
                    # ignore negative similairties
                    _similarity_matrix = torch.relu(F.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=-1))
                    similarity_matrix = _similarity_matrix.type(torch.float32)
                    similarity_matrix[similarity_matrix == 0] = 1e-5  # add small num for ensuring positive matrix

                    g = nx.from_numpy_matrix(similarity_matrix.cpu().numpy())
                    info = nx.eigenvector_centrality(g, max_iter=6000, weight='weight') # index: value
                    centrality = [info[i] for i in range(len(info))]

                    # get the most important node's index(m)
                    centrality = torch.tensor(centrality)
                    m = torch.argmax(centrality)

                    clean_idx.extend(torch.nonzero(y_mask)[:, -1][m].tolist())
                    clean_p.extend(centrality[m].tolist())

                print("*" * 30)
                #clean_idx.extend(torch.nonzero(y_mask)[:, -1][m].tolist())
                #clean_p.extend(_clean_ps[m].tolist())
                #print(xs.shape)
                #print(ys.shape)
                #print(clean_idx)
                #print(len(clean_idx))
                #print(len(clean_p))
                print("class: {}".format(u_y))
                print("--- num of selected samples: {}".format(min(u_y_c, proper_size)))
                #print("--- num of selected samples: {}".format(torch.sum(m).item()))
                #print("--- num of selected corrupt samples: {}".format(torch.sum(corr[m]).item()))
            print("***********************************************")
            imgs = torch.cat((imgs, xs[clean_idx]), dim=0)
            cats = torch.cat((cats, ys[clean_idx]), dim=0)
            corrupts = torch.cat((corrupts, corrs[clean_idx]), dim=0)
            indexs = torch.cat((indexs, idxs[clean_idx]), dim=0)

            print(imgs.shape)
            print(cats.shape)
            print(corrupts.shape)
            print(indexs.shape)
            print(len(clean_p))
        #return clean_idx, torch.Tensor(clean_p)
        cats = cats.long()
        return imgs, cats, corrupts, indexs, clean_p

    def train_self_base(self):
        """
        Self Replay. train base model with samples from delay and purified buffer
        """
        bs = self.config['base_batch_size']
        # If diverse buffer is full, train using it also
        # db_bs: delay buffer batch size
        # divb_bs: diverse buffer batch size
        db_bs = (bs // 2) if self.diverse_buffer.is_full() else bs
        db_bs = min(db_bs, len(self.delay_buffer))
        divb_bs = min(bs - db_bs, len(self.diverse_buffer))

        self.base.train()
        self.base.init_ntxent(self.config, batch_size=db_bs + divb_bs)

        # Remember
        # ***_buffer.get_dataloader(self, batch_szie, drop_last, shuffle)
        #   -> return DataLoader
        # ***_buffer.sample(self, num, cat=None)
        #   -> return samples

        dataloader = self.delay_buffer.get_dataloader(batch_size=db_bs, shuffle=True, drop_last=True)

        for epoch_i in tqdm.trange(self.config['base_train_epochs'], desc="base training", leave=False):
            for inner_step, data in enumerate(dataloader):
                x = data['imgs']
                self.base.zero_grad()
                # sample data from diverse buffer and merge
                if divb_bs > 0:
                    replay_data = self.diverse_buffer.sample(num=divb_bs)
                    x = torch.cat([replay_data['imgs'], x], dim=0)

                loss = self.base.get_selfsup_loss(x)
                loss.backward()
                self.base.optimizer.step()

                self.writer.add_scalar(
                    'continual_base_train_loss', loss,
                    self.base_step + inner_step + epoch_i * len(dataloader))

            # warmup for the first 10 epochs
            if epoch_i >= 10:
                self.base.lr_scheduler.step()

        self.writer.flush()
        self.base_step += self.config['base_train_epochs'] * len(dataloader)

    def get_finetuned_model(self):
        """copy the base and fine-tune for evaluation"""
        base_ft = self.get_init_base_ft(self.config)
        # overwrite entries in the state dict
        ft_dict = base_ft.state_dict()
        ft_dict.update({k: v for k, v in self.base.state_dict().items() if k in ft_dict})
        base_ft.load_state_dict(ft_dict)

        base_ft.train()
        dataloader = self.purified_buffer.get_dataloader(batch_size=self.config['ft_batch_size'], shuffle=True, drop_last=True)
        for epoch_i in tqdm.trange(self.config['ft_epochs'], desc='finetuning', leave=False):
            for inner_step, data in enumerate(dataloader):
                x, y = data['imgs'], data['cats']
                base_ft.zero_grad()
                loss = base_ft.get_sup_loss(x, y).mean()
                loss.backward()
                base_ft.clip_grad()
                base_ft.optimizer.step()
                base_ft.lr_scheduler.step()

                self.writer.add_scalar(
                    'ft_train_loss', loss,
                    self.base_ft_step + inner_step + epoch_i * len(dataloader))

        self.writer.flush()
        self.base_ft_step += self.config['ft_epochs'] * len(dataloader)
        base_ft.eval()
        return base_ft

    def forward(self, x):
        pass
