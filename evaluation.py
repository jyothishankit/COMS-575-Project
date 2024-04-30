import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
from constants import *

@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    # Preprocess audio
    noisy = preprocess_audio(noisy)
    length = noisy.size(-1)

    # Perform enhancement
    est_audio, length = perform_enhancement(model, noisy, n_fft, hop, length)

    # Save enhanced audio if required
    if save_tracks:
        save_audio(saved_dir, name, est_audio, sr)

    return est_audio, length

def preprocess_audio(audio):
    c = torch.sqrt(audio.size(-1) / torch.sum((audio**2.0), dim=-1))
    audio = torch.transpose(audio, 0, 1)
    audio = torch.transpose(audio * c, 0, 1)
    return audio

def perform_enhancement(model, audio, n_fft, hop, length):
    c = torch.sqrt(audio.size(-1) / torch.sum((audio**2.0), dim=-1))
    # Audio processing
    noisy_spec = torch.stft(
        audio, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )
    # Model inference
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
    # Post-processing
    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length
    return est_audio, length

def save_audio(directory, name, audio, sr):
    saved_path = os.path.join(directory, name)
    sf.write(saved_path, audio, sr)

def evaluate(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)

    audio_list = natsorted(os.listdir(noisy_dir))
    num = len(audio_list)
    metrics_total = np.zeros(6)

    for audio in audio_list:
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)

        est_audio, length = enhance_one_track(
            model, noisy_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks
        )

        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics_total += np.array(metrics)

    metrics_avg = metrics_total / num
    print("ssnr: ", metrics_avg[0], "stoi: ", metrics_avg[1])



if __name__ == "__main__":
    noisy_dir = os.path.join(EVALUATION_TEST_DIRECTORY, "noisy")
    clean_dir = os.path.join(EVALUATION_TEST_DIRECTORY, "clean")
    evaluate(EVALUATION_MODEL_PATH, noisy_dir, clean_dir, EVALUATION_SAVE_GENERATED_TRACKS_FLAG, EVALUATION_SAVE_GENERATED_TRACKS_DIRECTORY)
