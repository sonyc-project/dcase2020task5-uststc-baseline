import warnings
import numpy as np
import oyaml as yaml
import pandas as pd


def parse_ground_truth(annotation_path, yaml_path):
    """
    Parse ground truth annotations from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are coarse
    IDs of the form 1, 2, 3 etc.


    Parameters
    ----------
    annotation_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing coarse taxonomy.


    Returns
    -------
    gt_df: DataFrame
        Ground truth.
    """
    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.Loader)

    # Load CSV file into a Pandas DataFrame.
    ann_df = pd.read_csv(annotation_path)

    # Restrict to ground truth ("annotator zero").
    gt_df = ann_df[
        (ann_df["annotator_id"]==0) & (ann_df["split"]=="validate")]

    # Rename coarse columns.
    coarse_dict = yaml_dict["coarse"]
    coarse_renaming = {
        "_".join([str(c), coarse_dict[c], "presence"]): str(c)
        for c in coarse_dict}
    gt_df = gt_df.rename(columns=coarse_renaming)

    # Collect tag names as strings and map them to mixed (coarse-fine) ID pairs.
    # The "mixed key" is a hyphenation of the coarse ID and fine ID.
    fine_dict = {}
    for coarse_id in yaml_dict["fine"]:
        for fine_id in yaml_dict["fine"][coarse_id]:
            mixed_key = "-".join([str(coarse_id), str(fine_id)])
            fine_dict[mixed_key] = yaml_dict["fine"][coarse_id][fine_id]

    # Rename fine columns.
    fine_renaming = {"_".join([k, fine_dict[k], "presence"]): k
        for k in fine_dict}
    gt_df = gt_df.rename(columns=fine_renaming)

    # Loop over coarse tags.
    n_samples = len(gt_df)
    coarse_dict = yaml_dict["coarse"]
    for coarse_id in yaml_dict["coarse"]:
        # Construct incomplete fine tag by appending -X to the coarse tag.
        incomplete_tag = str(coarse_id) + "-X"

        # If the incomplete tag is not in the prediction, append a column of zeros.
        # This is the case e.g. for coarse ID 7 ("dogs") which has a single
        # fine-level tag ("7-1_dog-barking-whining") and thus no incomplete
        # tag 7-X.
        if incomplete_tag not in gt_df.columns:
            gt_df[incomplete_tag] = np.zeros((n_samples,)).astype('int')

    # Return output in DataFrame format.
    return gt_df.sort_values('audio_filename')


def parse_fine_prediction(pred_csv_path, yaml_path):
    """
    Parse fine-level predictions from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are mixed (coarse-fine)
    IDs of the form 1-1, 1-2, 1-3, ..., 1-X, 2-1, 2-2, 2-3, ... 2-X, 3-1, etc.


    Parameters
    ----------
    pred_csv_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing fine taxonomy.


    Returns
    -------
    pred_fine_df: DataFrame
        Fine-level complete predictions.
    """

    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.Loader)

    # Collect tag names as strings and map them to mixed (coarse-fine) ID pairs.
    # The "mixed key" is a hyphenation of the coarse ID and fine ID.
    fine_dict = {}
    for coarse_id in yaml_dict["fine"]:
        for fine_id in yaml_dict["fine"][coarse_id]:
            mixed_key = "-".join([str(coarse_id), str(fine_id)])
            fine_dict[mixed_key] = "_".join([
                mixed_key, yaml_dict["fine"][coarse_id][fine_id]])

    # Invert the key-value relationship between mixed key and tag.
    # Now, tags are the keys, and mixed keys (coarse-fine IDs) are the values.
    # This is possible because tags are unique.
    rev_fine_dict = {fine_dict[k]: k for k in fine_dict}

    # Read comma-separated values with the Pandas library
    pred_df = pd.read_csv(pred_csv_path)

    # Assign a predicted column to each mixed key, by using the tag as an
    # intermediate hashing step.
    pred_fine_dict = {}
    for f in sorted(rev_fine_dict.keys()):
        if f in pred_df:
            pred_fine_dict[rev_fine_dict[f]] = pred_df[f]
        else:
            pred_fine_dict[rev_fine_dict[f]] = np.zeros((len(pred_df),))
            warnings.warn("Column not found: " + f)

    # Loop over coarse tags.
    n_samples = len(pred_df)
    coarse_dict = yaml_dict["coarse"]
    for coarse_id in yaml_dict["coarse"]:
        # Construct incomplete fine tag by appending -X to the coarse tag.
        incomplete_tag = str(coarse_id) + "-X"

        # If the incomplete tag is not in the prediction, append a column of zeros.
        # This is the case e.g. for coarse ID 7 ("dogs") which has a single
        # fine-level tag ("7-1_dog-barking-whining") and thus no incomplete
        # tag 7-X.
        if incomplete_tag not in fine_dict.keys():
            pred_fine_dict[incomplete_tag] =\
                np.zeros((n_samples,)).astype('int')


    # Copy over the audio filename strings corresponding to each sample.
    pred_fine_dict["audio_filename"] = pred_df["audio_filename"]

    # Build a new Pandas DataFrame with mixed keys as column names.
    pred_fine_df = pd.DataFrame.from_dict(pred_fine_dict)

    # Return output in DataFrame format.
    # Column names are 1-1, 1-2, 1-3 ... 1-X, 2-1, 2-2, 2-3 ... 2-X, 3-1, etc.
    return pred_fine_df.sort_values('audio_filename')


def parse_coarse_prediction(pred_csv_path, yaml_path):
    """
    Parse coarse-level predictions from a CSV file containing both fine-level
    and coarse-level predictions (and possibly additional metadata).
    Returns a Pandas DataFrame in which the column names are coarse
    IDs of the form 1, 2, 3 etc.


    Parameters
    ----------
    pred_csv_path: string
        Path to the CSV file containing predictions.

    yaml_path: string
        Path to the YAML file containing coarse taxonomy.


    Returns
    -------
    pred_coarse_df: DataFrame
        Coarse-level complete predictions.
    """

    # Create dictionary to parse tags
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.Loader)

    # Collect tag names as strings and map them to coarse ID pairs.
    rev_coarse_dict = {"_".join([str(k), yaml_dict["coarse"][k]]): k
        for k in yaml_dict["coarse"]}

    # Read comma-separated values with the Pandas library
    pred_df = pd.read_csv(pred_csv_path)

    # Assign a predicted column to each coarse key, by using the tag as an
    # intermediate hashing step.
    pred_coarse_dict = {}
    for c in rev_coarse_dict:
        if c in pred_df:
            pred_coarse_dict[str(rev_coarse_dict[c])] = pred_df[c]
        else:
            pred_coarse_dict[str(rev_coarse_dict[c])] = np.zeros((len(pred_df),))
            warnings.warn("Column not found: " + c)

    # Copy over the audio filename strings corresponding to each sample.
    pred_coarse_dict["audio_filename"] = pred_df["audio_filename"]

    # Build a new Pandas DataFrame with coarse keys as column names.
    pred_coarse_df = pd.DataFrame.from_dict(pred_coarse_dict)

    # Return output in DataFrame format.
    # The column names are of the form 1, 2, 3, etc.
    return pred_coarse_df.sort_values('audio_filename')
