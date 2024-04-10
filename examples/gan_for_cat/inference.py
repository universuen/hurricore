import _path_setup  # noqa: F401
import torch
import cv2
from tqdm import tqdm

from hurricane.utils import import_config

from models import Generator

config = import_config('configs.for_256px')

CKPT_PATH = config.PathConfig().checkpoints / 'ckpt_step_45000'
STEP = 72
NUM_CATS = 5

if __name__ == '__main__':
    # load model
    generator = Generator(**config.GeneratorConfig()).to('cpu')
    model_path = CKPT_PATH / 'pytorch_model.bin'
    generator.load_state_dict(torch.load(model_path, map_location='cpu'))
    generator.eval()

    # Set up video writer
    video_filename = config.PathConfig().data / 'interpolation.mp4'
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_filename), fourcc, 24.0, (256, 256))
    
    # create interpolation
    cnt = 0
    start_z = torch.randn(1, config.GeneratorConfig().z_dim).to('cpu')
    end_z = torch.randn(1, config.GeneratorConfig().z_dim).to('cpu')
    while cnt < NUM_CATS:
        z_diff = (end_z - start_z) / STEP
        for i in tqdm(range(STEP)):
            z = start_z + z_diff * i
            with torch.no_grad():
                img = generator(z).squeeze().permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            frame = (img * 255).astype('uint8')
            # Convert frame from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        cnt += 1
        start_z = end_z
        end_z = torch.randn(1, config.GeneratorConfig().z_dim).to('cpu')

    # Release the video writer
    video_writer.release()
    print(f'Interpolation video saved at {video_filename}')
