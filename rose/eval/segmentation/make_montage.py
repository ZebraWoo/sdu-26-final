import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _collect_sample_ids(input_dir: Path):
    ids = set()
    for p in input_dir.glob("*_overlay.png"):
        ids.add(p.stem.replace("_overlay", ""))
    return sorted(ids)


def _load_or_blank(path: Path, size):
    if path.exists():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", size, (20, 20, 20))


def main():
    parser = argparse.ArgumentParser("Make segmentation visualization montage")
    parser.add_argument("--input-dir", required=True, help="Directory with *_img/*_pred/*_gt/*_overlay images")
    parser.add_argument("--output", required=True, help="Output montage image path")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--font-size", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sample_ids = _collect_sample_ids(input_dir)
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
    if not sample_ids:
        raise ValueError(f"No '*_overlay.png' files found in {input_dir}")

    first_img = Image.open(input_dir / f"{sample_ids[0]}_img.png").convert("RGB")
    w, h = first_img.size

    cols = ["img", "pred", "gt", "overlay"]
    header_h = args.font_size + 12
    out_w = w * len(cols)
    out_h = header_h + h * len(sample_ids)
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for j, name in enumerate(cols):
        draw.text((j * w + 8, 6), name, fill=(255, 255, 255))

    for i, sid in enumerate(sample_ids):
        y = header_h + i * h
        imgs = [
            _load_or_blank(input_dir / f"{sid}_img.png", (w, h)),
            _load_or_blank(input_dir / f"{sid}_pred.png", (w, h)),
            _load_or_blank(input_dir / f"{sid}_gt.png", (w, h)),
            _load_or_blank(input_dir / f"{sid}_overlay.png", (w, h)),
        ]
        for j, im in enumerate(imgs):
            canvas.paste(im, (j * w, y))
        draw.text((6, y + 6), sid, fill=(255, 255, 0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved montage: {output_path}")


if __name__ == "__main__":
    main()
