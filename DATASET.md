# ExtraSensory Dataset Guide

## Overview

This project uses UC San Diego's [ExtraSensory dataset](https://extrasensory.ucsd.edu/), a collection of smartphone and smartwatch sensor measurements paired with self-reported activity and context labels.

The local dataset contains:

- 60 gzip-compressed CSV files, one for each participant
- 377,346 observations in total
- 278 columns in each file
- 225 sensor-derived input features
- 51 activity and context labels

The files are stored under:

```text
data/
└── ExtraSensory.per_uuid_features_labels/
    ├── <participant-uuid>.features_labels.csv.gz
    └── ...
```

The participant UUID is part of the filename and is not included as a column in the CSV. The `data/` directory is intentionally excluded from Git because the dataset is large.

## What One Row Represents

Each row represents one observation window for one participant. Observations were typically collected about once per minute. During an observation, the device recorded sensor measurements and converted them into summary features.

These files contain precomputed features, not the original high-frequency sensor streams. For example, accelerometer columns contain statistics such as means, standard deviations, percentiles, spectral energy, and autocorrelation.

A row has the following general layout:

| Section | Purpose | Example |
| --- | --- | --- |
| Timestamp | Time of the observation | `timestamp` |
| Sensor features | Model inputs derived from phone or watch sensors | `raw_acc:magnitude_stats:mean` |
| Context labels | Model targets indicating activities or circumstances | `label:SITTING` |
| Label metadata | Information about how labels were obtained | `label_source` |

## Column Groups

The beginning of a column name identifies its source:

| Prefix | Meaning | Local column count |
| --- | --- | ---: |
| `raw_acc:` | Phone accelerometer features | 26 |
| `proc_gyro:` | Processed phone gyroscope features | 26 |
| `raw_magnet:` | Phone magnetometer features | 31 |
| `location:` | Location and movement-range features | 11 |
| `location_quick_features:` | Quick relative-location features | 6 |
| `audio_naive:` | Audio-derived MFCC statistics, not audio recordings | 26 |
| `audio_properties:` | Audio magnitude and normalization information | 2 |
| `discrete:` | Phone state, battery, Wi-Fi, and time-of-day indicators | 34 |
| `lf_measurements:` | Low-frequency readings such as light and pressure | 8 |
| `watch_acceleration:` | Smartwatch accelerometer features | 46 |
| `watch_heading:` | Smartwatch compass/heading features | 9 |
| `label:` | Activity and context targets | 51 |

The project primarily studies common smartphone sensors, so smartwatch columns can be excluded from the main experiments.

## Labels and Missing Values

This is a **multilabel classification** dataset. Several labels can be true for the same row. For example, one observation could simultaneously have the following labels:

```text
SITTING + COMPUTER_WORK + LOC_home + OR_indoors
```

Label columns use these values:

| Value | Meaning |
| --- | --- |
| `1` | The label applies to this observation |
| `0` | The label does not apply |
| `NaN` | The label is unknown or unavailable |

Do not automatically replace missing label values with `0`; unknown labels are not necessarily negative examples.

Sensor features also contain missing values. Sensors were not available on every device or during every observation. A model pipeline will therefore need to select sufficiently available features and handle missing inputs, usually with imputation.

Some label names contain cleaning prefixes:

- `FIX_` indicates a corrected activity label, such as `label:FIX_walking`.
- `LOC_` indicates a location context, such as `label:LOC_home`.
- `OR_` indicates a label produced by combining related original labels, such as `label:OR_indoors`.

## Reading One Participant

Pandas can read `.csv.gz` files directly, so decompression is not required:

```python
from pathlib import Path

import pandas as pd

data_dir = Path("data/ExtraSensory.per_uuid_features_labels")
path = next(data_dir.glob("*.csv.gz"))

df = pd.read_csv(path)
participant_id = path.name.split(".")[0]

df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

print("Participant:", participant_id)
print("Shape:", df.shape)
print(df.head())
```

To separate input features from target labels:

```python
feature_cols = [
    column
    for column in df.columns
    if column != "timestamp"
    and not column.startswith("label:")
    and column != "label_source"
]

label_cols = [
    column
    for column in df.columns
    if column.startswith("label:")
]
```

## Reading All Participants

Add the UUID from each filename before combining the files. Keeping participant identity is necessary for participant-level training and testing splits.

```python
from pathlib import Path

import pandas as pd

data_dir = Path("data/ExtraSensory.per_uuid_features_labels")
frames = []

for path in data_dir.glob("*.csv.gz"):
    participant = pd.read_csv(path)
    participant["participant_id"] = path.name.split(".")[0]
    frames.append(participant)

data = pd.concat(frames, ignore_index=True)

print(data.shape)
```

Loading every column for every participant can use considerably more memory than the 215 MB compressed dataset. During early exploration, load only the required columns:

```python
wanted_columns = [
    "timestamp",
    "raw_acc:magnitude_stats:mean",
    "raw_acc:magnitude_stats:std",
    "label:FIX_walking",
    "label:SITTING",
]

df = pd.read_csv(path, usecols=wanted_columns)
```

## Selecting Sensor Groups

Column prefixes make it straightforward to build sensor-specific feature sets:

```python
accelerometer_cols = [
    column for column in df.columns
    if column.startswith("raw_acc:")
]

gyroscope_cols = [
    column for column in df.columns
    if column.startswith("proc_gyro:")
]

phone_state_cols = [
    column for column in df.columns
    if column.startswith("discrete:")
    or column.startswith("lf_measurements:")
]
```

These lists can be combined to compare accelerometer-only, motion-sensor, location, phone-state, and all-phone-sensor models.

## Unpacking a File Manually

Manual decompression is optional. To create an uncompressed CSV while preserving the original file:

```bash
gunzip -k data/ExtraSensory.per_uuid_features_labels/<participant-uuid>.features_labels.csv.gz
```

Avoid unpacking every file unless an uncompressed copy is specifically needed; pandas can work with the compressed files and saves disk space.

## Modeling Considerations

- Split training and test data by `participant_id`, not by random rows. Otherwise, observations from the same person can appear in both sets and inflate performance estimates.
- Fit imputation and normalization steps using only the training data.
- Exclude `participant_id`, `timestamp`, labels, and `label_source` from model inputs unless an experiment explicitly needs time-derived features.
- Evaluate class imbalance with metrics such as balanced accuracy and F1 rather than plain accuracy alone.
- For each target label, discard rows where that particular target is `NaN` before training or evaluation.

## Additional Resources

- [ExtraSensory dataset and downloads](https://extrasensory.ucsd.edu/)
- [Introduction to the ExtraSensory Dataset](http://extrasensory.ucsd.edu/intro2extrasensory/intro2extrasensory.html)
- [Original ExtraSensory paper](https://arxiv.org/html/1609.06354v4)
