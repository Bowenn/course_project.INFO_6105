import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_args
from dataset import TestDataset, get_val_transforms
from model import build_model


LABEL_INV = {0: "Negative", 1: "Positive"}


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = build_model(args).to(device)
    assert args.checkpoint, "--checkpoint is required for prediction"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (val_acc={ckpt['val_acc']:.4f})")

    # data
    test_ds = TestDataset(args.test_dir, transform=get_val_transforms(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Test images: {len(test_ds)}")

    ids, preds = [], []
    with torch.no_grad():
        for imgs, image_ids in tqdm(test_loader, desc="predicting"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred_classes = outputs.argmax(1).cpu().tolist()
            ids.extend(image_ids)
            preds.extend(LABEL_INV[p] for p in pred_classes)

    df = pd.DataFrame({"image_id": ids, "predicted_class": preds})
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
