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

        self.purified_buffer = reservoir['purified'](config, config['purified_buffer_size'], config['purified_buffer_q_poa'])
        self.delay_buffer = reservoir['delay'](config, config['delayed_buffer_size'], config['delayed_buffer_q_poa'])

        self.E_max = config['E_max']

        self.expert_step = 0
        self.base_step = 0
        self.base_ft_step = 0

        self.base = self.get_init_base(config)

        #self.expert_number = 0
        #self.expert = self.get_init_expert(config)

        self.ssl_dir = os.path.join(os.path.dirname(os.path.dirname(self.config['log_dir'])),
                                    'noiserate_{}'.format(config['corruption_percent']),
                                    'expt_{}'.format(config['expert_train_epochs']),
                                    'randomseed_{}'.format(config['random_seed']))

        if os.path.exists(self.ssl_dir):
            with open(os.path.join(self.ssl_dir, 'idx_sets.npy'), 'rb') as f:
                self.debug_idxs = np.load(f, allow_pickle=True)

    def get_init_base(self, config):
        """get initialized base model"""
        base = Net[config['net']](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        lr_scheduler_config['options'].update({'T_max': config['base_train_epochs']})

        base.setup_optimizer(optim_config)
        base.setup_lr_scheduler(lr_scheduler_config)
        return base
    '''
    def get_init_expert(self, config):
        """get initialized expert model"""
        expert = Net[config['net']](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        lr_scheduler_config['options'].update({'T_max': config['expert_train_epochs']})

        expert.setup_optimizer(optim_config)
        expert.setup_lr_scheduler(lr_scheduler_config)
        return expert
    '''
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
                '''
                if not os.path.exists(os.path.join(self.ssl_dir, 'model{}.ckpt'.format(self.expert_number))):
                    self.expert = self.get_init_expert(self.config)
                    self.train_self_expert()
                else:
                    self.expert.load_state_dict(
                        torch.load(os.path.join(self.ssl_dir, 'model{}.ckpt'.format(self.expert_number)),
                                    map_location=self.device))
                    ################### data consistency check ######################
                    if torch.sum(self.delay_buffer.get('idxs') != torch.Tensor(self.debug_idxs[self.expert_number])) != 0:
                        raise Exception("it seems there is a data consistency problem: exp_num {}".format(self.expert_number))
                    ################### data consistency check ######################
                '''
                self.train_self_base()

                #clean_idx, clean_p = self.cluster_and_sample()
                #self.update_purified_buffer(clean_idx, clean_p, step)
                imgs, cats, corrupts, idxs, clean_p = self.cluster_and_sample()
                self.update_purified_buffer_full_info(imgs, cats, corrupts, idxs, clean_p, step)
                #self.expert_number += 1

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

            if self.purified_buffer.is_full():
                pb_xs = self.purified_buffer.get('imgs')
                pb_ys = self.purified_buffer.get('cats')
                pb_corrs = self.purified_buffer.get('corrupts')
                pb_idxs = self.purified_buffer.get('idxs')

                xs = torch.cat((xs, pb_xs), dim=0)
                ys = torch.cat((ys, pb_ys), dim=0)
                corrs = torch.cat((corrs, pb_corrs), dim=0)
                idxs = torch.cat((idxs, pb_idxs), dim=0)

            features = self.base(xs)
            features = F.normalize(features, dim=1)

            clean_p = list()
            clean_idx = list()
            imgs = torch.Tensor().cuda()
            cats = torch.Tensor().cuda()
            corrupts = torch.Tensor()
            indexs = torch.Tensor()
            print("***********************************************")
            # Find the proper number of y class
            pb_size = self.purified_buffer.rsvr_total_size

            # sorting unique_ys based on its counts
            unique_ys, unique_ys_count= torch.unique(ys, return_counts = True)
            unique_ys = unique_ys[torch.sort(unique_ys_count)[1]]
            unique_ys_count = torch.sort(unique_ys_count)[0]

            # get the proper_size
            proper_size = int(pb_size / unique_ys.shape[0])
            print(f"Proper_size = {proper_size}")

            #for u_y in torch.unique(ys).tolist():
            for u_y, u_y_c in zip(unique_ys, unique_ys_count):
                print(f"unique y / unique y count : {u_y} / {u_y_c}")
                y_mask = (ys == u_y)
                corr = corrs[y_mask]

                print("************************************************************")
                if (proper_size >= u_y_c) :
                    print(f"Iterative Sample Selection (just pass dataset (not enough data))")
                    # pass all samples to purified buffer
                    clean_idx.extend(torch.nonzero(y_mask)[:, -1].tolist())
                    clean_p.extend(torch.zeros_like(torch.nonzero(y_mask)[:, -1]).tolist())
                else :
                    feature = features[y_mask]
                    # ignore negative similairties
                    _similarity_matrix = torch.relu(F.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=-1))

                # stochastic ensemble
                # _clean_ps = torch.zeros((self.E_max, len(feature)), dtype=torch.double)
                #_clean_ps = torch.zeros(len(feature), dtype=torch.double)
                # XXX modified version
                # this is for stocastic process
                # similarity_matrix = (_similarity_matrix > torch.rand_like(_similarity_matrix)).type(torch.float32)
                    similarity_matrix = _similarity_matrix.type(torch.float32)
                    similarity_matrix[similarity_matrix == 0] = 1e-5  # add small num for ensuring positive matrix

                    print(f"Iterative Sample Selection (get {proper_size} from {u_y_c})")
                    for i in tqdm.trange(proper_size, desc="iterative sampling", leave=False):
                        g = nx.from_numpy_matrix(similarity_matrix.cpu().numpy())
                        info = nx.eigenvector_centrality(g, max_iter=6000, weight='weight') # index: value
                        centrality = [info[i] for i in range(len(info))]

#                get posterior for clean sample (p(z=clean|centrality))
#                bmm_model = BetaMixture1D(max_iters=10)
#                # fit beta mixture model
#                c = np.asarray(centrality)
#                c, c_min, c_max = bmm_model.outlier_remove(c)
#                c = bmm_model.normalize(c, c_min, c_max)
#                bmm_model.fit(c)
#                bmm_model.create_lookup(1) # 0: noisy, 1: clean
#
#                # get posterior
#                c = np.asarray(centrality)
#                c = bmm_model.normalize(c, c_min, c_max)
#                p = bmm_model.look_lookup(c)
#                # this is for stocastic process
#                # _clean_ps[_i] = torch.from_numpy(p)
#                _clean_ps = torch.from_numpy(p)
                        '''
                        for _i in range(self.E_max):
                            similarity_matrix = (_similarity_matrix > torch.rand_like(_similarity_matrix)).type(torch.float32)
                            similarity_matrix[similarity_matrix == 0] = 1e-5  # add small num for ensuring positive matrix

                            g = nx.from_numpy_matrix(similarity_matrix.cpu().numpy())
                            info = nx.eigenvector_centrality(g, max_iter=6000, weight='weight') # index: value
                            centrality = [info[i] for i in range(len(info))]

                            bmm_model = BetaMixture1D(max_iters=10)
                            # fit beta mixture model
                            c = np.asarray(centrality)
                            c, c_min, c_max = bmm_model.outlier_remove(c)
                            c = bmm_model.normalize(c, c_min, c_max)
                            bmm_model.fit(c)
                            bmm_model.create_lookup(1) # 0: noisy, 1: clean

                            # get posterior
                            c = np.asarray(centrality)
                            c = bmm_model.normalize(c, c_min, c_max)
                            p = bmm_model.look_lookup(c)
                            _clean_ps[_i] = torch.from_numpy(p)
                        '''
                        #_clean_ps = torch.mean(_clean_ps, dim=0)
#                _clean_ps = _clean_ps
                        # This is not threshold.
                        # This code is used for "sampling with probability"
                        #m = _clean_ps > torch.rand_like(_clean_ps)

                        # get the most important node
                        centrality = torch.Tensor(centrality)
                        centrality = (centrality - centrality.min())/(centrality.max() - centrality.min())
                        m = torch.argmax(centrality)
                        #clean_idx.extend(torch.nonzero(y_mask)[:, -1][m].tolist())
                        #clean_p.extend(centrality[m].tolist())
                        clean_idx.append(torch.nonzero(y_mask)[:, -1][m].tolist())
                        clean_p.append(centrality[m].tolist())

                        # Set 0 to m's edge
                        similarity_matrix[m, :] = 1e-5
                        similarity_matrix[:, m] = 1e-5

                print("************************************************************")
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
        """Self Replay. train base model with samples from delay and purified buffer"""
        bs = self.config['base_batch_size']
        # If purified buffer is full, train using it also
        db_bs = (bs // 2) if self.purified_buffer.is_full() else bs
        db_bs = min(db_bs, len(self.delay_buffer))
        pb_bs = min(bs - db_bs, len(self.purified_buffer))

        self.base.train()
        self.base.init_ntxent(self.config, batch_size=db_bs + pb_bs)

        dataloader = self.delay_buffer.get_dataloader(batch_size=db_bs, shuffle=True, drop_last=True)
        for epoch_i in tqdm.trange(self.config['base_train_epochs'], desc="base training", leave=False):
            for inner_step, data in enumerate(dataloader):
                x = data['imgs']
                self.base.zero_grad()
                # sample data from purified buffer and merge
                if pb_bs > 0:
                    replay_data = self.purified_buffer.sample(num=pb_bs)
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

    '''
    def train_self_expert(self):
        """train expert model with samples from delay"""
        batch_size =min(self.config['expert_batch_size'], len(self.delay_buffer))
        self.expert.train()
        self.expert.init_ntxent(self.config, batch_size=batch_size)

        dataloader = self.delay_buffer.get_dataloader(batch_size=batch_size, shuffle=True, drop_last=True)
        for epoch_i in tqdm.trange(self.config['expert_train_epochs'], desc='expert training', leave=False):
            for inner_step, data in enumerate(dataloader):
                x = data['imgs']
                self.expert.zero_grad()
                loss = self.expert.get_selfsup_loss(x)
                loss.backward()
                self.expert.optimizer.step()

                self.writer.add_scalar(
                    'expert_train_loss', loss,
                    self.expert_step + inner_step + len(dataloader) * epoch_i)

            # warmup for the first 10 epochs
            if epoch_i >= 10:
                self.expert.lr_scheduler.step()

        self.writer.flush()
        self.expert_step += self.config['expert_train_epochs'] * len(dataloader)
    '''

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
