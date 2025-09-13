import gc
import os
import sys
import time
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from models.ARMDMain.Utils.io_utils import instantiate_from_config, get_model_parameters_info
#from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'])
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        #print(samples.shape)
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            #print(sample.shape)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def sample_forecast(self, raw_dataloader, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        #print(samples.shape)

        for idx, batch in enumerate(raw_dataloader):
            if len(batch)==2:
                x, t_m = batch
                x, t_m = x.to(self.device), t_m.to(self.device)
            else:
                x = batch
                x = x.to(self.device)
            sample = self.ema.ema_model.generate_mts(x)
            #print(sample.shape)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            #reals = None
            reals = np.row_stack([reals, x[:,shape[0]:,:].detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals
    
    def sample_forecast_probabilistic(self, raw_dataloader, num_samples=100, shape=None):
        """
        Generates a probabilistic forecast using the CPU for memory-intensive storage,
        avoiding high VRAM usage.

        Args:
            raw_dataloader: The dataloader for the test set.
            num_samples (int): The number of sample trajectories to generate for each data point.
            shape (list): The shape [seq_len, feat_num] of the forecast.

        Returns:
            A tuple of (samples, reals) as NumPy arrays:
            - samples (np.ndarray): Ensemble of forecasts with shape [total_points, num_samples, seq_len, feat_num].
            - reals (np.ndarray): Ground truth with shape [total_points, seq_len, feat_num].
        """
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info(
                f'Begin to sample ({num_samples} trajectories per input, CPU storage)...'
            )

        all_samples_list = []
        all_reals_list = []

        # Cache for speed
        device = self.device
        generate = self.ema.ema_model.generate_mts
        seq_len, feat_num = shape

        for idx, batch in enumerate(raw_dataloader):
            # Move the input data to GPU
            if len(batch) == 2:
                x, t_m = batch
                x, t_m = x.to(device), t_m.to(device)
            else:
                x = batch.to(device)

            B = x.size(0)

            # Store ground truth immediately on CPU
            real_for_batch_np = x[:, seq_len:, :].cpu().numpy()
            all_reals_list.append(real_for_batch_np)

            # Preallocate batch array on CPU
            batch_samples_np_array = np.empty(
                (B, num_samples, seq_len, feat_num), dtype=np.float32
            )

            # Generate samples one by one (or in small chunks if VRAM allows)
            for i in range(num_samples):
                with torch.no_grad():
                    sample_gpu = generate(x)
                batch_samples_np_array[:, i] = sample_gpu.cpu().numpy()
                del sample_gpu  # free GPU tensor

            all_samples_list.append(batch_samples_np_array)

            # Free GPU memory
            del x
            torch.cuda.empty_cache()
            gc.collect()

        if self.logger is not None:
            self.logger.log_info(
                'Sampling done, time: {:.2f}'.format(time.time() - tic)
            )

        # Concatenate all batches into final arrays
        final_samples = np.concatenate(all_samples_list, axis=0)
        final_reals = np.concatenate(all_reals_list, axis=0)

        return final_samples, final_reals


