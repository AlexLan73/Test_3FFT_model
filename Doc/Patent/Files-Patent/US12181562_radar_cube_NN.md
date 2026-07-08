# US12181562B2 - Methods for Classifying Objects in Automotive-Grade Radar Signals

**Patent Number:** US12181562B2

**Title:** Methods for Classifying Objects in Automotive-Grade Radar Signals

**Assignee:** Indie Semiconductor Inc

**Inventors:** Tao Yu, Atulya Yellepeddi, Michael Price

**Priority Date:** 2021-03-02

**Filing Date:** 2022-03-01

**Publication Date:** 2024-12-31

**Application Number:** US17/684,022

**Legal Status:** Active, expires 2042-10-03

---

## Abstract

A method includes an operation to collect radar signals reflected from objects in a field of view. A three-dimensional range-angle-velocity cube is formed from the radar signals. The three-dimensional range-angle-velocity cube includes individual bins with radar intensity values characterizing angle and range for a specific velocity. Point-pillar sub-cubes are selected from the three-dimensional range-angle-velocity cube. Each point-pillar sub-cube includes a predefined range surrounding a high energy peak in the range-angle dimensions and an entire range in a velocity vector. The point-pillar sub-cubes are processed to compress, decompress, detect, classify or track objects in the field of view.

---

## Background

An autonomous vehicle (AV) is configured to navigate roadways based upon sensor signals output by sensors of the AV, wherein the AV navigates the roadways without input from a human. The AV is configured to identify and track objects (such as vehicles, pedestrians, bicyclists, static objects, and so forth) based upon the sensor signals output by the sensors of the AV and perform driving maneuvers (such as accelerating, decelerating, turning, stopping, etc.) based upon the identified and tracked objects.

In certain environments, a plurality of different types of sensors for sensing the surroundings of a vehicle are used, such as monoscopic or stereoscopic cameras, light detection and ranging (LiDAR) sensors, and radio detection and ranging (radar) sensors. The different sensor types comprise different characteristics that may be utilized for different tasks.

Embodiments of the present disclosure concern aspects of processing measurement data of radar systems, whereby the computationally heavy fusion of sensor data (e.g., range, angle and velocity) can be mitigated. This is particularly useful when one parameter array needs to be populated before processing another, such as range and velocity.

---

## Technical Field

The present disclosure relates to techniques for classifying objects in automotive-grade radar signals. More particularly, the disclosure describes using point pillars to better classify objects in a radar scene.

The disclosure generally relates to Millimeter Wave Sensing. Specifically, the present method pertains to a sensing technology called Frequency Modulated Continuous Waves (FMCW) RADARS, which is very popular in automotive and industrial segments.

---

## Key Technical Concepts

### FMCW Radar Fundamentals

A chirp is a sinusoid or a sine wave whose frequency increases linearly with time. The chirp is a continuous wave whose frequency is linearly modulated. Hence the term frequency modulated continuous wave or FMCW.

FMCW radar measures the range, velocity, and angle of arrival of objects in front of it. At the heart of an FMCW radar is a signal called a chirp.

The radar operates as follows:
1. A synthesizer generates a chirp
2. The chirp is transmitted by the TX antenna
3. The chirp is reflected off an object
4. The reflected chirp is received at the RX antenna
5. The RX signal and the TX signal are mixed at a mixer
6. The resultant signal is called an intermediate (IF) signal
7. The IF signal is prepared for signal processing by low-pass (LP) filtering and sampled using an analog to digital converter (ADC)

### Range-Angle-Velocity Cube

Radar data form of 3-dimensional, complex-valued array (a.k.a. a radar cube) with dimensions corresponding to azimuth (angle), radial velocity (doppler), and radial distance (range).

The magnitude in each angle-doppler-range bin characterizes how much energy the radar sensor sees coming from that point in space (angle and range) for that radial velocity.

### Problem in the Art

The problem in the art arises from the vast volume of energy data populating the cube. This makes processing the data implausible in a real time environment. Solutions presently found in the art include processing one dimension (i.e., parameter) at a time. However, this is not useful for some applications, such as 3-d cube processing. Furthermore, previous efforts tend to throw away data which can be useful during object classification.

Object classification is typically done on the object list where Doppler information is lost. More rich information is obtained by clustering multiple detections as "objects". "Features" are extracted from the object list and classification is performed on the extracted features. Due to limited bandwidth between the radar unit and the central computer, it has not been feasible to leverage full velocity information for classification tasks.

Thus, there is a need in the art for improved techniques for retaining data during object classification.

---

## Point-Pillar Sub-Cubes

### Sub-Cube Selection

Point-pillars are selected as sub-cubes. Sub-cubes are selected to sufficiently surround the high energy peaks in the range-angle dimensions, while selecting the entire range in the velocity vector.

Each point-pillar sub-cube includes a predefined range surrounding a high energy peak in the range-angle dimensions and an entire range in a velocity vector.

The entire velocity array may be utilized to provide for accurate classification of objects.

### Sub-Threshold Point-Pillars

In one embodiment, sub-threshold point-pillars are used to select pillars. Sub-thresholding is determined and performed in the following manner:

1. Strongest intensity peaks are identified
2. Within the range-angle space, other local maxima can then be identified within a predetermined neighborhood
3. The sub-threshold is a percentage (e.g., 50%) of the intensity of the strongest peak, or a function of the distance from the strongest peak and/or predetermined parameters
4. The sub-threshold point pillar comprises the union of the range-angle of the strongest peak with a contiguous local maximum and a disparate neighbor exhibiting an intensity above the sub-threshold

---

## Classification and Processing

### Point-Pillar Based Classification

The point-pillars are subject to:
- Clustering (points are clustered into common objects)
- Tracking (object trajectory)
- Classification (apply semantic label to object)

### Velocity Signatures

Velocity signatures are characteristic patterns of an object. For example, a tire moving orthogonally to a radar array has a deterministic velocity signature. The center has little doppler velocity, as it moves tangentially; yet the annulus exhibits a distinct array of velocities. Velocity signatures can be used to identify an object by pattern matching these velocities with those predetermined, or they may be learned by a neural network.

### Pointwise Operations

The velocity data can be reduced by the application of a pointwise operator. Pointwise is used to indicate that each value f(x) is subject to some function φ. Pointwise operations apply operations to function values separately for each point.

Pointwise operators can be applied to each range-angle bin. In other embodiments, it can be a convolutional featurization, which is a convolutional kernel applied over a plurality of range, angle and doppler bins.

### Convolutional Neural Networks

CNNs (Convolutional Neural Networks) are regularized versions of multilayer perceptrons. CNNs take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns.

---

## Point-Pillar Based Auto-Encoding

Full velocity vector compression comprises doppler dimensional reduction by applying two layers of pointwise featurization. The latter portion of the pipeline recovers the doppler dimension by applying two layers of feature recovery.

This point-pillar based Autoencoding is useful for compressing data for transmission to a central classifier while preserving fidelity such that interpretability is maximized.

The data in point-pillars are compressed and sent over a bus to another computational structure which performs the actual classification.

---

## Classifications (IPC/CPC)

- G01S13/00 — Systems using the reflection or reradiation of radio waves (radar systems)
- G01S13/583 — Velocity or trajectory determination systems using Doppler effect
- G01S13/584 — Velocity or trajectory determination systems adapted for simultaneous range and velocity measurements
- G01S13/931 — Radar or analogous systems specially adapted for anti-collision purposes of land vehicles
- G01S7/417 — Using analysis of echo signal for target characterisation using neural networks
- G06N3/00 — Computing arrangements based on biological models
- G06N3/02 — Neural networks
- G06N3/04 — Neural network architecture
- G06N3/045 — Auto-encoder networks; Encoder-decoder networks
- G06N3/0464 — Convolutional networks [CNN, ConvNet]

---

## Related Applications

- This application claims priority to U.S. Provisional Patent Application Ser. No. 63/155,508, filed Mar. 2, 2021
- Related to U.S. Provisional Patent Applications No. 63/123,403 entitled "METHOD, APPARATUS AND RADAR SYSTEMS FOR TRACKING OBJECTS" filed on Dec. 9, 2020
- Related to U.S. Provisional Patent Applications No. 63/143,154 entitled "METHOD FOR DETECTING OBJECTS IN AUTOMOTIVE-GRADE RADAR SIGNALS" filed on Jan. 29, 2021

---

## Drawings

The patent includes 13 figures:
- FIGS. 1A and 1B: Exemplary radar chirp as a function of time
- FIG. 2: Exemplary auto-grade radar system
- FIGS. 3A and 3B: Frequency difference in exemplary send and receive radar chirps
- FIG. 4: Exemplary two-dimensional range array being populated
- FIGS. 5A and 5B: Creation of a velocity-range array from a chirp index-range array
- FIG. 6: Exemplary antenna array used to calculate angle
- FIG. 7: Processing method chain in exemplary prior art radar system
- FIG. 8: Processing method chain in exemplary radar system according to embodiment
- FIG. 9: Exemplary sub-cube in a range-angle-velocity radar cube
- FIG. 10: Selecting point-pillars using sub-thresholds in a range-angle-velocity radar cube
- FIG. 11: Exemplary method for point-pillar based classification in automotive-grade radar
- FIG. 12: Exemplary method for point-pillar based auto-encoding in automotive-grade radar
- FIG. 13: Schematic of an exemplary radar system

---

**Source:** US Patent Office, Google Patents
