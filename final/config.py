import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Surface Crack Detection CNN")

    # data
    parser.add_argument("--data_dir", type=str, default="data/train",
                        help="root dir containing Positive/ and Negative/ subdirs")
    parser.add_argument("--test_dir", type=str, default="data/test",
                        help="dir containing test images")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--img_size", type=int, default=227)

    # model
    parser.add_argument("--model", type=str, default="custom_cnn",
                        choices=["custom_cnn", "resnet18", "resnet34", "resnet50",
                                 "vgg16"],
                        help="model architecture")
    parser.add_argument("--num_conv_blocks", type=int, default=4,
                        help="number of conv blocks for custom_cnn")
    parser.add_argument("--base_filters", type=int, default=32,
                        help="filters in the first conv layer (doubles each block)")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pretrained", action="store_true",
                        help="use pretrained weights for resnet/vgg")

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"])
    parser.add_argument("--step_size", type=int, default=7,
                        help="step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="gamma for StepLR scheduler")

    # reproducibility & I/O
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="name for this run (used in checkpoint filename)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a checkpoint to resume or predict from")

    # bias-variance analysis
    parser.add_argument("--bv_runs", type=int, default=0,
                        help="number of bootstrap runs for bias-variance decomp "
                             "(0 = skip)")
    parser.add_argument("--bv_train_ratio", type=float, default=0.5,
                        help="fraction of data used per bootstrap sample")

    return parser.parse_args()
