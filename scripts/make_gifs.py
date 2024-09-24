#! /usr/bin/env python3
import argparse
import os

from PIL import Image


def create_animation(input_dirs, output_dir, num_colors, frame_duration, quality):
    gif_output_dir = os.path.join("plots/animations", output_dir, "gif")
    apng_output_dir = os.path.join("plots/animations", output_dir, "apng")
    os.makedirs(gif_output_dir, exist_ok=True)
    os.makedirs(apng_output_dir, exist_ok=True)

    # Get the number of PNG files in each directory
    num_files = len(
        [name for name in os.listdir(input_dirs[0]) if name.endswith(".png")]
    )
    print(f"Number of files: {num_files}")

    for i in range(num_files):
        frames = []
        for dir in input_dirs:
            img_path = os.path.join(dir, f"{i:02d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)

                # # Convert to P mode with a high-quality palette
                # img = img.convert("RGB").convert(
                #     "P", palette=Image.ADAPTIVE, colors=num_colors
                # )

                frames.append(img)

        if frames:
            gif_output_path = os.path.join(gif_output_dir, f"{i:02d}.gif")
            apng_output_path = os.path.join(apng_output_dir, f"{i:02d}.png")
            frames[0].save(
                gif_output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,  # Turn off optimize
                duration=frame_duration,
                loop=0,
                quality=quality,  # Add quality parameter
                disposal=2,
            )  # Add disposal method
            print(f"Created {gif_output_path}")

            frames[0].save(
                apng_output_path,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
                disposal=2,
                format="PNG",
            )
            print(f"Created {apng_output_path}")
        else:
            print(f"No images found for index {i}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG sequences to high-quality animated GIFs and APNGs."
    )
    parser.add_argument(
        "input_dirs", nargs="+", help="Input directories containing PNG sequences"
    )
    parser.add_argument("--output_dir", "-o", help="Output directory for animations")
    parser.add_argument(
        "--num_colors",
        "-c",
        type=int,
        default=256,
        help="Number of colors in the output GIF",
    )
    parser.add_argument(
        "--frame_duration",
        "-f",
        type=int,
        default=100,
        help="Duration of each frame in milliseconds",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=100,
        help="Quality of GIF (1-100, higher is better)",
    )

    args = parser.parse_args()

    create_animation(
        args.input_dirs,
        args.output_dir,
        args.num_colors,
        args.frame_duration,
        args.quality,
    )


if __name__ == "__main__":
    main()
