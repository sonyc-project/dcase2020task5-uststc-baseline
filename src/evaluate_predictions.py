import argparse
import json
import os
import oyaml as yaml
from metrics import compute_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluation script for Urban Sound Tagging task for the DCASE 2019 Challenge.

        See `metrics.py` for more information about the metrics.
        """)

    parser.add_argument('prediction_path', type=str,
                        help='Path to prediction CSV file.')
    parser.add_argument('annotation_path', type=str,
                        help='Path to dataset annotation CSV file.')
    parser.add_argument('yaml_path', type=str,
                        help='Path to dataset taxonomy YAML file.')
    parser.add_argument('--eval-split', type=str, choices=["validate", "test"],
                        default="validate",
                        help='Split with which to evaluate model.')
    parser.add_argument('--target-mode', type=str, choices=["verified", "sonyc_annotator_agreement"],
                        default="verified",
                        help="Method for determining ground truth targets from annotations."
                             "'verified' uses the final annotations verified by the SONYC team"
                             "after disagreement resolution. 'sonyc_annotator_agreement' produces"
                             "positives only if both SONYC annotators agree on the presence of a tag.")

    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        taxonomy = yaml.load(f)

    metrics = {}
    for mode in ("fine", "coarse"):
        metrics[mode] = compute_metrics(args.prediction_path,
                                        args.annotation_path,
                                        args.yaml_path,
                                        mode,
                                        eval_split=args.eval_split,
                                        target_mode=args.target_mode)

        print("{} level evaluation:".format(mode.capitalize()))
        print("======================")
        print(" * Micro AUPRC:           {}".format(metrics[mode]["micro_auprc"]))
        print(" * Micro F1-score (@0.5): {}".format(metrics[mode]["micro_f1"]))
        print(" * Macro AUPRC:           {}".format(metrics[mode]["macro_auprc"]))
        print(" * Coarse Tag AUPRC:")
        for coarse_name, auprc in metrics[mode]["class_auprc"].items():
            print("      - {}: {}".format(coarse_name, auprc))
        print(" * lwlrap:                {}".format(metrics[mode]["lwlrap"]))
        print(" * Per-Tag lwlrap:")
        for tag_name, lwlrap in metrics[mode]["class_lwlrap"].items():
            weight = metrics[mode]["class_lwlrap_weight"]
            print("      - {}: {}".format(tag_name, lwlrap))
            print("        (Weight: {})".format(weight))

    prediction_fname = os.path.splitext(os.path.basename(args.prediction_path))[0]
    eval_fname = "evaluation_{}.json".format(prediction_fname)
    eval_path = os.path.join(os.path.dirname(args.prediction_path), eval_fname)
    with open(eval_path, 'w') as f:
        json.dump(metrics, f)

