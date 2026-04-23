"""
Export trained UNet++ checkpoint to ONNX for Roboflow upload.

Usage:
  python export_onnx.py --checkpoint runs/s2_20260227_231556/best_model.pth
"""

import argparse
import torch
import segmentation_models_pytorch as smp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="unetpp_s2.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]

    num_classes = 1   # binary segmentation
    model = smp.UnetPlusPlus(
        encoder_name=ckpt_args["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    ).eval()
    model.load_state_dict(ckpt["model_state"])

    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported: {args.out}")
    print(f"  mode={ckpt_args['mode']}  classes={num_classes}  encoder={ckpt_args['encoder']}")
    print(f"  val IoU={ckpt['val_iou']:.4f}  epoch={ckpt['epoch']}")


if __name__ == "__main__":
    main()