import _path_setup

from pathlib import Path
import cv2
import os

from configs.for_64px import PathConfig

def create_video_from_images(img_folder_path: Path, saving_path: Path):
    images = [img for img in os.listdir(img_folder_path) if img.endswith((".png", ".jpg", ".jpeg"))]
    # Ensure there are images to process
    if not images:
        raise ValueError("No images found in the folder.")
    # Read the first image to set video properties
    frame = cv2.imread(str(img_folder_path / images[0]))
    height, width, layers = frame.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24  # Frames per second
    out = cv2.VideoWriter(str(saving_path), fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(str(img_folder_path / image))
        out.write(frame)  # Write out frame to video
    # Release the video writer
    out.release()  

if __name__ == '__main__':
    create_video_from_images(PathConfig().peek_images, PathConfig().data / 'training_process.mp4')
