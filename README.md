# CSCI 4622 Group Project: Milestone 1

## SensorScope

**Team members**

- Matt Topham
- Jacob Lehman
- James Vallery

## Problem Space

We want to use machine learning to figure out which smartphone sensors are needed to recognize a person’s activity or context.

Our main research question is: **What is the minimum set of common smartphone sensors needed to accurately predict a person’s activity or context?**

The idea came from an interest in building a context-aware AI assistant. Knowing whether someone is walking, sleeping, or working could help an assistant provide more relevant information. We want to find out whether a small set of sensors can provide useful context without needing every available input.

Machine learning is useful because sensor readings vary between people and depending on how they carry their phones. A classifier could learn patterns that would be difficult to describe with fixed rules.

Previous ExtraSensory research has compared sensors and sensor combinations. We will build on those comparisons to identify smaller sensor sets that perform close to an all-phone-sensor baseline and examine which activities become harder to recognize. Our initial target is balanced accuracy within five percentage points of that baseline.

## Data / Data Plan

We plan to use UC San Diego’s public ExtraSensory dataset. It contains more than 300,000 labeled examples from 60 participants, typically collected at one-minute intervals.

The dataset includes features from accelerometers, gyroscopes, magnetometers, location, audio, and phone state. Audio is provided as numerical features derived from sound. We will use the precomputed features and exclude smartwatch features, participant IDs as model inputs, and the separate absolute-location data.

We will compare combinations such as:

- Accelerometer only
- Accelerometer and gyroscope
- Motion sensors and location
- Motion sensors and phone state
- All selected phone-sensor features

This is supervised multilabel classification: one example can include multiple activities or contexts. Possible labels include walking, sitting, bicycling, sleeping, talking, computer work, at home, and at school. We will narrow the list after checking how much usable data each label has.

We plan to compare logistic regression and random forests, testing on participants who were not included in training. We will use balanced accuracy and F1 scores to compare sensor combinations and identify where fewer sensors work well or lose performance.

The dataset is available for download now. We still need to download and inspect the files, which we will do before Milestone 2. No new data collection is needed.

## References

- [ExtraSensory dataset and documentation](https://extrasensory.ucsd.edu/)
- [Introduction to the ExtraSensory Dataset](http://extrasensory.ucsd.edu/intro2extrasensory/intro2extrasensory.html)
- Vaizman, Ellis, and Lanckriet (2017). [“Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches.”](https://arxiv.org/html/1609.06354v4)
- de Souza et al. (2023). [“A Study on the Influence of Sensors in Frequency and Time Domains on Context Recognition.”](https://doi.org/10.3390/s23125756)

## Presentation Plan

| Slide | What to show and explain | Time |
| --- | --- | --- |
| 1. Problem and research question | The assistant motivation, the question about fewer sensors, and the five-point target. | 40 sec |
| 2. Dataset overview | Source, participants, examples, selected feature count, and what one row represents. | 40 sec |
| 3. Inputs and outcomes | A small real data sample, important sensor features, selected labels, and a brief missing-data observation. | 40 sec |
| 4. Planned experiment | Sensor combinations, logistic regression and random forest, and testing on unseen participants. | 40 sec |
| 5. Contribution and deliverables | How you build on prior work, planned comparison charts, and one stretch goal. | 20 sec |

## Data Section

The main contribution of this work is the emphasis on in-the-wild conditions:

- **Naturally used devices.** Subjects used their own personal phones, and a smartwatch that we provided.
- **Unconstrained device placement.** Subjects were free to carry their phone in any way that was convenient to them.
- **Natural environment.** Subjects collected data in their own regular environment for about a week.
- **Natural behavioral content.** No script or tasks were given. We did not target a specific set of activities. Instead, the context labels that we analyze came from the data, as the subjects engaged in their routine and applied any relevant labels (from the large menu) that fit what they were doing.
