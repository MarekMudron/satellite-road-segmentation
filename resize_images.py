import argparse
import glob
import os
from PIL import Image


def parseargs():
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('-e', '--input-extension', required=True)
    parser.add_argument('-t', '--type', required=True, help='sat or mask')
    args = parser.parse_args()
    return args

def main():
    args = parseargs()

    if args.type == 'sat':
        output_suffix = '_sat.jpg'
        filtering = Image.BILINEAR
    elif args.type == 'mask':
        output_suffix = '_mask.png'
        filtering = Image.NEAREST
    else:
        raise ValueError("--type must be sat or mask")

    img_paths = glob.glob(os.path.join(args.input_dir, "*" + args.input_extension))

    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.resize((1024, 1024), filtering)
        filename, _ = os.path.splitext(os.path.basename(img_path))
        out_file = os.path.join(args.output_dir, filename + output_suffix)
        img.save(out_file)

if __name__ == '__main__':
    main()