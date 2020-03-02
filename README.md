# DCASE 2020 Challenge: Task 5 - Urban Sound Tagging with Spatiotemporal Context

This repository contains code to reproduce the baseline results and evaluate system outputs for [Task 5 (Urban Sound Tagging with Spatiotemporal Context)](http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context) of the the [DCASE 2020 Challenge](http://dcase.community/challenge2020). We encourage participants to use this code as a starting point for manipulating the dataset and for evaluating their system outputs.

## Installation
You'll need [Python 3](https://www.python.org/download/releases/3.0/) and [Anaconda](https://www.anaconda.com/distribution/) installed, and will need a bash terminal environment.

Before doing anything else, clone this repository and enter it:

```shell
git clone https://github.com/sonyc-project/dcase2020task5-uststc-baseline
cd dcase2020task5-uststc-baseline
```

### Quick Start

To get started quickly, simply run:

```shell
# Replace with your preferred directory:
export SONYC_UST_PATH=~/sonyc-ust-stc
./setup.sh
```

### Setup Guide

If you want to go through the motions of setting up the environment, you can follow this guide.

First, set up some environment variables to make things easier for yourself. Feel free to change these to a directory that works better for you.

```shell
export SONYC_UST_PATH=~/sonyc-ust-stc
```

Then set up your Python environment:

```shell
conda create -n sonyc-ust-stc python=3.6
source activate sonyc-ust-stc
pip install -r requirements.txt
```

Now, download the dataset from [Zenodo](https://zenodo.org/record/3693077) and decompress the audio files:
```shell
mkdir -p $SONYC_UST_PATH/data
pushd $SONYC_UST_PATH/data
wget https://zenodo.org/record/3693077/files/annotations.csv
wget https://zenodo.org/record/3693077/files/audio.tar.gz
wget https://zenodo.org/record/3693077/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/3693077/files/README.md
tar xf audio.tar.gz
rm audio.tar.gz
popd
```

Your environment is now set up!


## Replicating baseline
### Quick Start

To get started immediately (assuming you've set up your environment), you can just run:

```shell
# Replace with your preferred directory:
export SONYC_UST_PATH=~/sonyc-ust-stc
./baseline_example.sh
```

### Baseline Guide

First, activate your conda environment (if it isn't already activated).

```shell
source activate sonyc-ust-stc
```

Then, set up some environment variables to make things easier. Feel free to change these to a directory that works better for you.

```shell
export SONYC_UST_PATH=~/sonyc-ust-stc
```

Enter the source code directory within the repository:

```shell
cd src
```

Extract embeddings from the SONYC-UST data using [OpenL3](https://github.com/marl/openl3):

```shell
openl3 audio $SONYC_UST_PATH/data/audio --output-dir $SONYC_UST_PATH/features --input-repr mel256 --content-type env --audio-embedding-size 512 --audio-hop-size 1.0 --audio-batch-size 16
```

Now, train a fine-level model and produce predictions:

```shell
python classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/embeddings $SONYC_UST_PATH/output baseline_fine --label_mode fine
```

Evaluate the fine-level model output file (using frame-averaged clip predictions) on AUPRC:

```shell
python evaluate_predictions.py $SONYC_UST_PATH/output/baseline_fine/*/output.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml
```

Now, train a coarse-level model and produce predictions:

```shell
python classify.py $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml $SONYC_UST_PATH/embeddings $SONYC_UST_PATH/output baseline_coarse --label_mode coarse
```

Evaluate the coarse-level model output file (using frame-averaged clip predictions) on AUPRC:

```shell
python evaluate_predictions.py $SONYC_UST_PATH/output/baseline_coarse/*/output.csv $SONYC_UST_PATH/data/annotations.csv $SONYC_UST_PATH/data/dcase-ust-taxonomy.yaml
```

## Baseline Description

For the baseline model, we use a multi-label multi-layer perceptron model, using a single hidden layer of size 128 (with ReLU non-linearities), and using [AutoPool](https://github.com/marl/autopool) to aggregate frame level predictions.  The model takes in as input:
 * Audio content, via OpenL3 embeddings (`content_type="env"`, `input_repr="mel256"`, and `embedding_size=512`), using a window size and hop size of 1.0 second (with centered windows), giving us 11 512-dimensional embeddings for each clip in our dataset.
 * Spatial context, via latitude and longitude values, giving us 2 values for each clip in our dataset.
 * Temporal context, via hour of the day, day of the week, and week of the year, each encoded as a one hot vector, giving us 24 + 7 + 52 = 83 values for each clip in our dataset.
 
We Z-score normalize the embeddings, latitude, and longitude values, and concatenate all of the inputs (at each time step), resulting in an input size of 512 + 2 + 83 = 597.

We use the weak tags for each audio clip as the targets for each clip. For the `train` data (which has no verified target), we count a positive for a tag if at least one annotator has labeled the audio clip with that tag (i.e. minority vote). Note that while some of the audio clips in the `train` set have verified annotations, we only use the crowdsourced annotations. For audio clips in the `validate` set, we only use annotations that have been manually verified.

We train the model using stochastic gradient descent to minimize binary cross-entropy loss, using L2 regularization (A.K.A. weight decay) with a factor of 1e-5. For training models to predict tags at the fine level, we modify the loss such that if "unknown/other" is annotated for a particular coarse tag, the loss for the fine tags corresponding to this coarse tag are masked out. We train for up to 100 epochs; to mitigate overfitting, we use early stopping with a patience of 20 epochs using loss on the `validate` set. We train one model to predict fine-level tags, with coarse-level tag predictions obtained by taking the maximum probability over fine-tags predictions within a coarse category. We train another model only to predict coarse-level tags.


## Metrics Description

The Urban Sound Tagging challenge is a task of multilabel classification. To evaluate and rank participants, we ask them to submit a CSV file following a similar layout as the publicly available CSV file of the development set: in it, each row should represent a different ten-second snippet, and each column should represent an urban sound tag.

The area under the precision-recall curve (AUPRC) is the classification metric that we employ to rank participants. To compute this curve, we threshold the confidence of every tag in every snippet by some fixed threshold tau, thus resulting in a one-hot encoding of predicted tags. Then, we count the total number of true positives (TP), false positives (FP), and false negatives (FN) between prediction and consensus ground truth over the entire evaluation dataset.

The Urban Sound Tagging challenge provides two leaderboards of participants, according to two distinct metric: fine-grained AUPRC and coarse-grained AUPRC. In each of the two levels of granularity, we vary tau between 0 and 1 and compute TP, FP, and FN for each coarse category. Then, we compute micro-averaged precision P = TP / (TP + FP) and recall R = TP / (TP + TN), giving an equal importance to every sample. We repeat the same operation for all values of tau in the interval [0, 1] that result in different values of P and R. Lastly, we use the trapezoidal rule to estimate the AUPRC.

The computations can be summarized by the following expressions defined for each coarse category, where `t_0` and `y_0` correspond to the presence of an incomplete tag in the ground truth and prediction (respectively), and `t_k` and `y_k` (for `k = 1, ..., K`) correspond to the presence of fine tag `k` in the ground truth and prediction (respectively).

![Mathematical expressions for computing true positives, false positives, and false negatives at the coarse level.](./figs/coarse_metrics.png)

For samples with complete ground truth (i.e., in the absence of the incomplete fine tag in the ground truth for the coarse category at hand), evaluating urban sound tagging at a fine level of granularity is also relatively straightforward. Indeed, for samples with complete ground truth, the computation of TP, FP, and FN amounts to pairwise conjunctions between predicted fine tags and corresponding ground truth fine tags, without any coarsening. Each fine tag produces either one TP (if it is present and predicted), one FP (if it it absent yet predicted), or one FN (if it is absent yet not predicted). Then, we apply one-hot integer encoding to these boolean values, and sum them up at the level of coarse categories before micro-averaging across coarse categories over the entire evaluation dataset. In this case, the sum (TP+FP+FN) is equal to the number of tags in the fine-grained taxonomy, i.e. 23. Furthermore, the sum (TP+FN) is equal to the number of truly present tags in the sample at hand.

The situation becomes considerably more complex when the incomplete fine tag is present in the ground truth, because this presence hinders the possibility of precisely counting the number of false alarms in the coarse category at hand. We propose a pragmatic solution to this problem; the guiding idea behind our solution is to evaluate the prediction at the fine level only when possible, and fall back to the coarse level if necessary.

For example, if a small engine is present in the ground truth and absent in the prediction but an "other/unknown" engine is predicted, then it's a true positive in the coarse-grained sense, but a false negative in the fine-grained sense. However, if a small engine is absent in the ground truth and present in the prediction, then the outcome of the evaluation will depend on the completeness of the ground truth for the coarse category of engines. If this coarse category is complete (i.e. if the tag "engine of uncertain size" is absent from the ground truth), then we may evaluate the small engine tag at the fine level, and count it as a false positive. Conversely, if the coarse category of engines is incomplete (i.e. the tag "engine of uncertain size" is present in the ground truth), then we fall back to coarse-level evaluation for the sample at hand, and count the small engine prediction as a true positive, in aggregation with potential predictions of medium engines and large engines.

The computations can be summarized by the following expressions defined for each coarse category, where `t_0` and `y_0` correspond to the presence of an incomplete tag in the ground truth and prediction (respectively), and `t_k` and `y_k` (for `k = 1, ..., K`) correspond to the presence of fine tag `k` in the ground truth and prediction (respectively).

![Mathematical expressions for computing true positives, false positives, and false negatives at the fine level.](./figs/fine_metrics.png)

As a secondary metric, we report the micro-averaged F-score of the system, after fixing the value of the threshold to 0.5. This score is the harmonic mean between precision and recall: F = 2\*P\*R / (P + R). We only provide the F-score metric for purposes of post-hoc error analysis and do not use it at the time of producing the official leaderboard.


## Baseline Results

### Fine-level model

#### Fine-level evaluation:

* Micro AUPRC: 0.7329
* Micro F1-score (@0.5): 0.6149
* Macro AUPRC: 0.5278
* Coarse Tag AUPRC:
                                                         
    | Coarse Tag Name | AUPRC |
    | :--- | :--- |
    | engine | 0.6429 |
    | machinery-impact | 0.5098 |
    | non-machinery-impact | 0.4474 |
    | powered-saw | 0.5194 |
    | alert-signal | 0.8238 |
    | music | 0.3151 |
    | human-voice | 0.9073 |
    | dog | 0.0568 |

#### Coarse-level evaluation:

* Micro AUPRC: 0.8391
* Micro F1-score (@0.5): 0.6736
* Macro AUPRC: 0.6370

* Coarse Tag AUPRC:                                      
                                                         
    | Coarse Tag Name | AUPRC |
    | :--- | :--- |
    | engine | 0.8191 |
    | machinery-impact | 0.6502 |
    | non-machinery-impact | 0.4474 |
    | powered-saw | 0.7960 |
    | alert-signal | 0.8837 |
    | music | 0.4720 |
    | human-voice | 0.9711 |
    | dog | 0.0568 |

### Coarse-level model

#### Coarse-level evaluation:
* Micro AUPRC: 0.8352
* Micro F1-score (@0.5): 0.7389
* Macro AUPRC: 0.6323
* Coarse Tag AUPRC:
                                                         
    | Coarse Tag Name | AUPRC |
    | :--- | :--- |
    | engine | 0.8500 |
    | machinery-impact | 0.6021 |
    | non-machinery-impact | 0.4192 |
    | powered-saw | 0.7200 |
    | alert-signal | 0.8518 |
    | music | 0.6145 |
    | human-voice | 0.9593 |            
    | dog | 0.0463 |

### Appendix: taxonomy of SONYC urban sound tags

We reproduce the classification taxonomy of the DCASE Urban Sound Challenge in the diagram below. Rectangular and round boxes respectively denote coarse and fine complete tags. For the sake of brevity, we do not explicitly show the incomplete fine tag in each coarse category.

![Taxonomy of SONYC urban sound tags.](./figs/dcase_ust_taxonomy.png)
