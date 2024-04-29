import os
import torch
import logging
from hyperparameters import *
from dimensions import *
from models.generator import TSCNet
from models import discriminator
from data import dataloader
from torch.nn.functional import F
from utils import power_compress, power_uncompress
from torchinfo import summary
from constants import *
from hyperparameters import *

logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, train_dataset, test_dataset, gpu_id: int):
        self.n_fft = 400
        self.hop = 100
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.setup_model()
        self.setup_discriminator()
        self.setup_optimizers()
        self.gpu_id = gpu_id

    def setup_model(self):
        num_channel = 64
        self.model = TSCNet(num_channel=num_channel, num_features=self.n_fft // 2 + 1).cuda()
        summary(
            self.model, [(1, 2, CUT_LENGTH_CONSTANT // self.hop + 1, int(self.n_fft / 2) + 1)]
        )

    def setup_discriminator(self):
        self.discriminator = discriminator.Discriminator(ndf=DISCRIMINATOR_DEPTH_OF_FEATURE_MAPS).cuda()
        summary(
            self.discriminator,
            [
                (1, 1, int(self.n_fft / 2) + 1, CUT_LENGTH_CONSTANT // self.hop + 1),
                (1, 1, int(self.n_fft / 2) + 1, CUT_LENGTH_CONSTANT // self.hop + 1),
            ],
        )

    def setup_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=INITIAL_LEARNING_RATE_CONSTANT)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * INITIAL_LEARNING_RATE_CONSTANT
        )


    def forward_generator_step(self, clean_audio, noisy_audio):
        c = self._normalize_audio(noisy_audio)
        noisy_spec = self._compute_stft(noisy_audio)
        clean_spec = self._compute_stft(clean_audio)
        noisy_spec_compressed = self._compress_power(noisy_spec)
        clean_spec_compressed = self._compress_power(clean_spec)
        clean_real, clean_imag = self._extract_real_imag(clean_spec_compressed)
        est_real, est_imag = self._model_inference(noisy_spec_compressed)
        est_mag = self._compute_magnitude(est_real, est_imag)
        est_audio = self._compute_estimated_audio(est_real, est_imag)
        clean_mag = self._compute_magnitude(clean_real, clean_imag)
        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def _normalize_audio(self, audio):
        normalization_factor = torch.sqrt(audio.size(-1) / torch.sum((audio**2.0), dim=-1))
        audio_normalized = torch.transpose(audio, 0, 1) * normalization_factor
        return torch.transpose(audio_normalized, 0, 1)

    def _compute_stft(self, audio):
        return torch.stft(
            audio,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

    def _compute_estimated_audio(self, real, imag):
        spec_uncompress = power_uncompress(real, imag).squeeze(1)
        return torch.istft(
            spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
    
    def _model_inference(self, spec):
        est_real, est_imag = self.model(spec)
        return est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    
    def _extract_real_imag(self, spec):
        real = spec[:, 0, :, :].unsqueeze(1)
        imag = spec[:, 1, :, :].unsqueeze(1)
        return real, imag


    def _compute_magnitude(self, real, imag):
        return torch.sqrt(real**2 + imag**2)
    
    def _compress_power(self, spec):
        return power_compress(spec).permute(0, 1, 3, 2)




    def calculate_generator_loss(self, generator_outputs):
        gan_loss = self._calculate_gan_loss(generator_outputs)
        mag_loss = self._calculate_mag_loss(generator_outputs)
        ri_loss = self._calculate_ri_loss(generator_outputs)
        time_loss = self._calculate_time_loss(generator_outputs)
    
        total_loss = (
            TOTAL_LOSS_WEIGHTS_CONSTANT[0] * ri_loss +
            TOTAL_LOSS_WEIGHTS_CONSTANT[1] * mag_loss +
            TOTAL_LOSS_WEIGHTS_CONSTANT[2] * time_loss +
            TOTAL_LOSS_WEIGHTS_CONSTANT[3] * gan_loss
        )
    
        return total_loss


    def _calculate_ri_loss(self, generator_outputs):
        return (
            F.mse_loss(generator_outputs["est_real"], generator_outputs["clean_real"]) +
            F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])
        )

    def _calculate_time_loss(self, generator_outputs):
        return torch.mean(torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"]))

    def _calculate_mag_loss(self, generator_outputs):
        return F.mse_loss(generator_outputs["est_mag"], generator_outputs["clean_mag"])


    def _calculate_gan_loss(self, generator_outputs):
        predict_fake_metric = self.discriminator(generator_outputs["clean_mag"], generator_outputs["est_mag"])
        return F.mse_loss(predict_fake_metric.flatten(), generator_outputs["one_labels"].float())

    def calculate_discriminator_loss(self, generator_outputs):
        est_audio_list, clean_audio_list = self._extract_audio_lists(generator_outputs)
        metric_score_score = discriminator.batch_metric_score(clean_audio_list, est_audio_list)

        if metric_score_score is not None:
            predict_enhance_metric, predict_max_metric = self._predict_metrics(generator_outputs)
            discrim_loss_metric = self._compute_loss(predict_max_metric, predict_enhance_metric, generator_outputs, metric_score_score)
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def _compute_loss(self, predict_max_metric, predict_enhance_metric, generator_outputs, metric_score_score):
        return (
            F.mse_loss(predict_max_metric.flatten(), generator_outputs["one_labels"]) +
            F.mse_loss(predict_enhance_metric.flatten(), metric_score_score)
        )


    def _predict_metrics(self, generator_outputs):
        predict_enhance_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
        )
        predict_max_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["clean_mag"]
        )
        return predict_enhance_metric, predict_max_metric


    def _extract_audio_lists(self, generator_outputs):
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy())
        return est_audio_list, clean_audio_list


    def _extract_audio_lists(self, generator_outputs):
        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        return est_audio_list, clean_audio_list


    def train_step(self, batch):
        clean_audio, noisy_audio = self._prepare_batch(batch)

        generator_outputs = self._generate_outputs(clean_audio, noisy_audio)
        generator_outputs["one_labels"] = torch.ones(BATCH_SIZE_CONSTANT).to(self.gpu_id)
        generator_outputs["clean"] = clean_audio

        generator_loss = self._calculate_generator_loss(generator_outputs)
        self._update_generator(generator_loss)

        discriminator_loss = self._update_discriminator(generator_outputs)

        return generator_loss.item(), discriminator_loss.item()

    def _update_discriminator(self, generator_outputs):
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])
        return discrim_loss_metric

    def _generate_outputs(self, clean_audio, noisy_audio):
        return self.forward_generator_step(clean_audio, noisy_audio)

    def _calculate_generator_loss(self, generator_outputs):
        return self.calculate_generator_loss(generator_outputs)

    def _prepare_batch(self, batch):
        clean_audio = batch[0].to(self.gpu_id)
        noisy_audio = batch[1].to(self.gpu_id)
        return clean_audio, noisy_audio


    def _update_generator(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    @torch.no_grad()
    def test_step(self, batch):
        clean_audio, noisy_audio = self._prepare_batch(batch)

        generator_outputs = self._generate_outputs(clean_audio, noisy_audio)
        generator_outputs["one_labels"] = torch.ones(BATCH_SIZE_CONSTANT).to(self.gpu_id)
        generator_outputs["clean"] = clean_audio

        generator_loss = self._calculate_generator_loss(generator_outputs)

        discriminator_loss = self._calculate_discriminator_loss(generator_outputs)
        if discriminator_loss is None:
            discriminator_loss = torch.tensor([0.0])

        return generator_loss.item(), discriminator_loss.item()


    def _generate_outputs(self, clean_audio, noisy_audio):
        return self.forward_generator_step(clean_audio, noisy_audio)

    def _calculate_generator_loss(self, generator_outputs):
        return self.calculate_generator_loss(generator_outputs)

    def _calculate_discriminator_loss(self, generator_outputs):
        return self.calculate_discriminator_loss(generator_outputs)

    def _prepare_batch(self, batch):
        clean_audio = batch[0].to(self.gpu_id)
        noisy_audio = batch[1].to(self.gpu_id)
        return clean_audio, noisy_audio


    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total, disc_loss_total = 0.0, 0.0
        total_batches = len(self.test_ds)
    
        for index, batch in enumerate(self.test_ds):
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
    
        gen_loss_avg = gen_loss_total / total_batches
        disc_loss_avg = disc_loss_total / total_batches
    
        self._log_test_results(gen_loss_avg, disc_loss_avg)
    
        return gen_loss_avg

    def _log_test_results(self, gen_loss_avg, disc_loss_avg):
        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))


    def train(self):
        scheduler_G = self._create_scheduler(self.optimizer)
        scheduler_D = self._create_scheduler(self.optimizer_disc)
    
        for epoch in range(EPOCHS_CONSTANT):
            self._set_model_to_train_mode()
            for index, batch in enumerate(self.train_ds):
                loss, disc_loss = self.train_step(batch)
                self._log_training_progress(epoch, index, loss, disc_loss)
        
            gen_loss = self.test()
            self._save_model(epoch, gen_loss)
        
            scheduler_G.step()
            scheduler_D.step()

    def _create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=DECAY_EPOCH_INTERVAL_CONSTANT, gamma=0.5
        )

    def _set_model_to_train_mode(self):
        self.model.train()
        self.discriminator.train()

    def _log_training_progress(self, epoch, step, loss, disc_loss):
        template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
        if (step % LOGS_INTERVAL_CONSTANT) == 0:
            logging.info(template.format(self.gpu_id, epoch, step, loss, disc_loss))

    def _save_model(self, epoch, gen_loss):
        path = os.path.join(
            SAVE_MODEL_DIRECTORY_CONSTANT,
            f"CMGAN_epoch_{epoch}_{gen_loss:.5f}"
        )
        os.makedirs(SAVE_MODEL_DIRECTORY_CONSTANT, exist_ok=True)
        if self.gpu_id == 0:
            torch.save(self.model.module.state_dict(), path)



def main():
    available_gpus = get_available_gpus()
    print(available_gpus)
    
    train_ds, test_ds = load_dataset()
    
    gpu_id = 0
    trainer = Trainer(train_ds, test_ds, gpu_id)
    trainer.train()

def get_available_gpus():
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

def load_dataset():
    train_ds, test_ds = dataloader.load_data(
        TRAINING_DATA_DIRECTORY_CONSTANT, BATCH_SIZE_CONSTANT, 2, CUT_LENGTH_CONSTANT
    )
    return train_ds, test_ds

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(world_size)
    main()
