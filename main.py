import torch
import numpy as np
from utility import runner
from PIL import Image
import os
from preload import preloader

def process():
    current_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        default="./",
        help="path to image",
    )

    parser.add_argument(
        "--seg_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights of segmentation",
    )

    parser.add_argument(
        "--lpr_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights of segmentation",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to save output image",
    )



    parser.add_argument(
        "--debug",
        action="store_true",
        help=" ",
    )

    args = parser.parse_args()

    
    model = preloader()
    with torch.no_grad():

        im = Image.open(args.image)
        frame = np.array(im)
        array_image, data_dictionary = runner(frame,model,cfg)
        final_image = Image.fromarray(array_image, "RGB")
        final_image.save(f"{current_path}/{img}")  # for Separate Use


if __name__ == "__main__":
    process()