import itertools
import os
import torch

import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from mdet.data.sample import Sample
from mdet.utils.factory import FI
import mdet.utils.io as io


class PlWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # train
        self.train_model = FI.create(config['model']['train'])
        self.train_codec = FI.create(config['codec']['train'])
        self.train_collater = self.train_codec.get_collater()
        self.train_model.set_train()

        # eval(val, test)
        self.eval_model = FI.create(config['model']['eval'])
        self.eval_codec = FI.create(config['codec']['eval'])
        self.eval_collater = self.eval_codec.get_collater()
        self.eval_model.set_eval()

        # infer(export for depoly)
        self.infer_model = FI.create(config['model']['infer'])
        self.infer_codec = FI.create(config['codec']['infer'])
        self.infer_collater = self.infer_codec.get_collater()
        self.infer_model.set_infer()

        # dataset
        self.train_dataset = FI.create(config['data']['train']['dataset'])
        self.val_dataset = FI.create(config['data']['val']['dataset'])
        self.test_dataset = FI.create(config['data']['test']['dataset'])

        # attach codec to dataset
        self.train_dataset.codec = self.train_codec
        self.val_dataset.codec = self.eval_codec
        self.test_dataset.codec = self.eval_codec

        # config for validation
        self.val_log_loss = config['runtime']['val'].get('log_loss', True)
        self.val_evaluate = config['runtime']['val'].get('evaluate', True)
        self.val_interest_set = config['runtime']['val'].get(
            'interest_set', set())
        self.val_epoch_interest_set = config['runtime']['val'].get(
            'epoch_interest_set', set())
        self.val_step_hook = config['runtime']['val'].get('step_hook', None)
        self.val_epoch_hook = config['runtime']['val'].get('epoch_hook', None)
        if self.val_evaluate:
            self.val_interest_set |= {'anno', 'pred'}
            self.val_epoch_interest_set |= {'anno', 'pred'}

    def forward(self, *input):
        # forward is used for export
        output = self.infer_model(*input)
        pred = self.infer_codec.decode(output, infer=True)
        return pred

    def training_step(self, batch, batch_idx):
        output = self.train_model(batch)
        loss_dict = self.train_codec.loss(output, batch)

        # log loss
        for name, value in loss_dict.items():
            if name != 'loss':
                value.detach_()
            self.log(f'train/{name}', value)

        return loss_dict

    def training_epoch_end(self, epoch_output):
        pass

    def validation_step(self, batch, batch_idx):
        output = self.eval_model(batch)

        # check if should log loss
        if self.val_log_loss:
            loss_dict = self.eval_codec.loss(output, batch)
            for name, value in loss_dict.items():
                self.log(f'val/{name}', value,
                         batch_size=batch['_info_']['size'])

        # construct batch interested
        batch_interest = Sample(_info_=batch['_info_'])
        for key in self.val_interest_set:
            if key == 'pred':
                batch_interest[key] = self.eval_codec.decode(output, batch)
            elif key == 'output':
                batch_interest[key] = output
            elif key in batch:
                batch_interest[key] = batch[key]
            else:
                print(key)
                raise NotImplementedError

        # decollate to sample_list
        sample_list = self.eval_collater.decollate(batch_interest)

        # do callback on samples
        if self.val_step_hook is not None and sample_list:
            for sample in sample_list:
                self.val_step_hook(sample, self)

        # output for epoch end
        val_step_out = []
        if self.val_epoch_interest_set and sample_list:
            for sample in sample_list:
                val_step_out.append(sample.select(self.val_epoch_interest_set))

        return val_step_out

    def validation_epoch_end(self, step_output_list):
        # collate epoch_output
        sample_list = itertools.chain.from_iterable(step_output_list)

        # epoch callback
        if self.val_epoch_hook is not None:
            self.val_epoch_hook(sample_list, self)

        # format and evaluation
        if self.val_evaluate:
            pred_path, anno_path = self.val_dataset.format(sample_list)
            metric = self.val_dataset.evaluate(pred_path, anno_path)
            self.log_dict(metric)

    def test_step(self, batch, batch_idx):
        output = self.eval_model(batch)

        # merge output into batch and decollate into sample_list
        batch['output'] = output
        merged_sample_list = self.eval_collater.decollate(batch)

        # decode samples into standard format
        standard_sample_list = []
        encoded_sample_list = []
        for merged_sample in merged_sample_list:
            standard_sample = self.eval_codec.decode(
                Sample(
                    output=merged_sample['output'],
                    data=merged_sample['data'],
                    meta=merged_sample['meta'],
                )
            )
            standard_sample_list.append(standard_sample)
            # TODO: online plot

            encoded_sample = Sample(
                output=merged_sample['output'],
                input=merged_sample['input'],
                gt=merged_sample['gt'],
                meta=merged_sample['meta'],
            )
            encoded_sample_list.append(encoded_sample)
            # TODO: online plot

        test_output = dict(
            pred=[sample['pred'] for sample in standard_sample_list],
            meta=[sample['meta'] for sample in standard_sample_list],
            output=[sample['output'] for sample in encoded_sample_list],
        )
        return test_output

    def test_epoch_end(self, epoch_output):
        # collate epoch_output
        epoch_output = self.collate_output(epoch_output)

        # save output if required
        self.save_output(
            epoch_output,
            self.config['runtime']['test'].get('prediction_folder', None),
            self.config['runtime']['test'].get('raw_output_folder', None),
        )

        # format according to different dataset
        pred_path, _ = self.test_dataset.format(
            epoch_output,
            self.config['runtime']['test'].get('formated_path', None),
            None,
        )

        print(f'Formatted output is saved to "{pred_path}"')

    def collate_output(self, epoch_output):
        collated = {key: [] for key in epoch_output[0].keys()}
        for batch_output in epoch_output:
            for key in collated.keys():
                collated[key].extend(batch_output[key])
        return collated

    def save_output(self, output, prediction_folder, raw_output_folder):
        # store output if output folder has been set
        if prediction_folder is not None:
            os.makedirs(prediction_folder, exist_ok=True)
            print(f'Saving prediction to \'{prediction_folder}\'')
            for pred, meta in zip(output['pred'], output['meta']):
                sample_name = meta['sample_name']
                fname = f'{prediction_folder}/{sample_name}.pkl'
                io.dump(pred, fname)

        if raw_output_folder is not None:
            os.makedirs(raw_output_folder, exist_ok=True)
            print(f'Saving raw output to \'{raw_output_folder}\'')
            for raw_output, meta in zip(output['output'], output['meta']):
                sample_name = meta['sample_name']
                fname = f'{raw_output_folder}/{sample_name}.pkl'
                io.dump(raw_output, fname)

    def train_dataloader(self):
        dataloader_cfg = self.config['data']['train'].copy()
        dataloader_cfg.pop('dataset')
        return DataLoader(self.train_dataset, collate_fn=self.train_collater, **dataloader_cfg)

    def val_dataloader(self):
        dataloader_cfg = self.config['data']['val'].copy()
        dataloader_cfg.pop('dataset')
        return DataLoader(self.val_dataset, collate_fn=self.eval_collater, **dataloader_cfg)

    def test_dataloader(self):
        dataloader_cfg = self.config['data']['test'].copy()
        dataloader_cfg.pop('dataset')
        return DataLoader(self.test_dataset, collate_fn=self.eval_collater, **dataloader_cfg)

    def configure_optimizers(self):
        # optimizer
        optim_cfg = self.config['fit']['optimizer'].copy()
        optim_type_name = optim_cfg.pop('type')
        optimizer = optim.__dict__[optim_type_name](
            self.train_model.parameters(), **optim_cfg)

        # lr scheduler
        sched_cfg = self.config['fit']['scheduler'].copy()
        sched_type_name = sched_cfg.pop('type')

        # hack for specific lr_scheduler
        if sched_type_name == 'OneCycleLR':
            sched_cfg['total_steps'] = len(
                self.train_dataloader()) * self.config['fit']['max_epochs']

        scheduler = optim.lr_scheduler.__dict__[
            sched_type_name](optimizer, **sched_cfg)

        return [optimizer], [scheduler]

    def track_model(self, m_target, m_follow, name_str=''):
        # track parameters
        m_follow._parameters = m_target._parameters

        # track buffers
        m_follow._buffers = m_target._buffers

        # recursively track submodules
        for name in m_follow._modules:
            self.track_model(
                m_target._modules[name], m_follow._modules[name], f'{name_str}.{name}')

    def on_train_start(self):
        self.track_model(self.train_model, self.eval_model)

    def on_validation_start(self):
        self.track_model(self.train_model, self.eval_model)

    def on_test_start(self):
        self.track_model(self.train_model, self.eval_model)

    def export(self, output_file='tmp.onnx', **kwargs):
        # model
        self.cuda()
        self.track_model(self.train_model, self.infer_model)

        # data
        batch = iter(self.test_dataloader()).next().select(
            ['input']).to('cuda')

        input, input_name, output_name, dynamic_axes = self.infer_codec.get_export_info(
            batch)

        # export
        torch.onnx.export(self,
                          input,
                          output_file,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=dynamic_axes,
                          keep_initializers_as_inputs=False,
                          opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          **kwargs)
