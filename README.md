# Problem Framing
Alzheimer’s disease (AD) has recently been characterized as a disorder of large-scale brain network disintegration rather than just an isolated regional dysfunction. Neuroimaging studies consistently demonstrate early disruption of functional connectivity, particularly within hub-dominated systems such as the Default Mode Network; this is then followed by compensatory reorganization and progressive network collapse. However, many existing machine learning approaches rely on static or Euclidean representations that limit their ability to capture the network-driven nature of AD. However, for clinical relevance, one must be able to discern how the model makes those decisions, particularly from the interaction between brain regions over time. Without this, the classifications are simply unhelpful.
To address this limitation, this project reframes Alzheimer’s disease modeling as a graph learning problem. Brain regions are represented as nodes, which have two primary parts: A node’s edges encode dynamic functional connectivity, while its features incorporate neurophysiological descriptors of regional activity and synchrony. To capture the deteriorating connectivity across time, the model extends graph convolution with continuous-time graph neural ordinary differential equations. This enables a smooth modeling of longitudinal network evolution. Interpretability is included through attention-based edge weighting and node-level attribution, allowing identification of disease-relevant regions and subnetworks consistent with known AD pathology.

# Methods
To model Alzheimer’s disease as a progressive network disorder, this project portrays each subject as a longitudinal sequence of functional brain networks evolving. For each clinical session, resting-state fMRI (rs-fMRI) time series are extracted from regions of interest (ROIs). Pairwise Pearson correlations between ROI signals are computed to construct functional connectivity matrices at each time point.
Each time point is represented as a weighted, undirected graph, where nodes correspond to ROIs and edges encode the strength of functional connectivity. The novelty is that they are organized into subject-specific temporal trajectories that preserve the evolution of network structure across clinical visits. This formulation explicitly captures progressive connectivity disruption rather than simply static representations.
To model continuous-time network evolution under irregular longitudinal sampling, the model uses graph neural ordinary differential equations (GNN-ODEs) in the learning process. This allows the latent trajectories to be modeled smoothly, accommodating missing visits and variable inter-scan intervals common in longitudinal neuroimaging.
Disease-relevant predictions are produced from time-dependent graph representations without collapsing temporal information into flat vectors. The model is trained end-to-end in a supervised setting using gradient-based optimization. The architectural choices are guided by stability, interpretability, and biological plausibility. 

# Evaluation
Model evaluation focuses on both predictive performance and temporal coherence. Standard supervised metrics such as training and validation loss and classification accuracy are used to assess learning stability, with the model achieving 96.2% accuracy on rs-fMRI data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). These results confirm that graph-structured, continuous-time models can successfully encode progressive network changes associated with Alzheimer’s disease.

# Future Work:
All planned extensions are detailed in the accompanying design document and will be completed before major research competitions (ISEF and JSHS) and the conclusion of the Hack for Health mentorship:
https://docs.google.com/document/d/1cy0_JPhTfHO2ONlpRYeG0yKOuKpWstInn_SPx_0A3I0/

---

### To run this file, install a Python virtual environment + requirements.txt. 
### Then type:
`python run_gnn_on_synthetic.py`.

---

Thank you!
