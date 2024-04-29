from pydub import AudioSegment
import os

def convert_sample_rate(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000)# Resample to 16,000 Hz
            audio.export(output_path, format="wav")# Export to output folder with format WAV

if __name__ == "__main__":
    input_folder = "/work/classtmp/hritikz/VCTK-DEMAND/train/noisy"
    output_folder = "/work/classtmp/hritikz/VCTK-CONVERTED/train/noisy"
    convert_sample_rate(input_folder, output_folder)
    convert_sample_rate("/work/classtmp/hritikz/VCTK-DEMAND/test/clean", "/work/classtmp/hritikz/VCTK-CONVERTED/test/clean")
    convert_sample_rate("/work/classtmp/hritikz/VCTK-DEMAND/test/noisy", "/work/classtmp/hritikz/VCTK-CONVERTED/test/noisy")
