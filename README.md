# 🧠 MRI Realism using 3D Autoencoder

PIPELINE SUMMARY:
MRI → Dataset Loader → 3D Autoencoder → Reconstruction → Visualization

This project focuses on reconstructing realistic brain MRI volumes using a 3D Convolutional Autoencoder trained on the BraTS 2023 Glioma Dataset. The pipeline includes data loading, preprocessing, model training, validation, and visualization.

⸻

## 🚀 Project Overview
	•	Dataset: BraTS 2023 (Glioma MRI)
	•	Model: 3D Convolutional Autoencoder
	•	Goal: Reconstruct MRI volumes from multi-modal inputs
	•	Output: High-quality reconstructed MRI slices (Axial, Coronal, Sagittal)

NOTE:
This project does NOT:
- Perform segmentation
- Diagnose disease
- Replace medical analysis
It is only for MRI reconstruction research.


## 🖥️ Environment Setup (GPU Server)
🔐 Step 1: Login to Server
ssh craft-guest@craft-1.cs.binghamton.edu

Step 2: Activate Environment
. ~/miniconda3/etc/profile.d/conda.sh
conda activate mri_realism

## ⚙️ Step 3: Verify GPU
nvidia-smi

## 🎯 Step 4:Select GPU
export CUDA_VISIBLE_DEVICES = 2

## 📁 Project Structure

mri-realism-3d-autoencoder/
│
├── docs/
├── scripts/
├── examples/
├── README.md
├── requirements.txt
└── .gitignore

## 📊 Dataset Setup
Training Dataset
export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

Validation Dataset
export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/validation/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData

## 🧠 Scripts Overview

1. dataset_gli.py
Loads MRI data, normalizes intensities, and converts into tensors of shape [4, H, W, D].

2. train_autoencoder3d.py
Trains the 3D autoencoder using L1 loss and saves model checkpoints after each epoch.

3. viz_ae3d_recon.py
Loads trained model and generates reconstruction images for a given patient (3-plane visualization).

4. refiner3d.py
Defines a residual refinement network to enhance reconstructed MRI quality.

5. train_refiner3d.py
Trains the refinement model using autoencoder outputs as input and original MRI as target.

6. viz_ae_refiner_compare.py
Compares original, reconstructed, and refined MRI outputs for visual evaluation.

### INPUT:
BraTS-GLI-00015-000 (4 MRI modalities)
### OUTPUT:
ae3d_recon_BraTS-GLI-00015-000.png
Includes:
- Original MRI
- Reconstructed MRI
- Axial / Coronal / Sagittal views

## 🏋️ Training the Model
Set Paths
cd ~/sjaipurkar/projects/mri_realism
export PYTHONPATH=$(pwd)
export OUTDIR=~/sjaipurkar/projects/mri_realism/runs/ae3d_run_big_v2

Run Training
python scripts/train_autoencoder3d.py

Output
•	Model checkpoints:
checkpoint_epochX.pt

•	Final model:
checkpoint_epoch19.pt

## 📉 Training Results

| Epoch | Loss |
|------|------|
| 1 | 0.028 |
| 2 | 0.012 |
| 3 | 0.008 |
| 4 | 0.007 |
| 5 | 0.006 |

✅ Loss decreases consistently → model successfully learns MRI structure.

### EXPECTED RESULT QUALITY:
- Reconstruction should look smooth and realistic
- Tumor regions should still be visible
- Loss should reduce to ~0.006

If output looks wrong → check troubleshooting section

### OUTPUT LOCATION:
Checkpoints:
~/sjaipurkar/projects/mri_realism/runs/ae3d_run_big_v2/
Images:
examples/ or current working directory


## 🔍 Validation / Inference
Set Variables

cd ~/sjaipurkar/projects/mri_realism
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES=2
export MRI_DATA_ROOT=~/sjaipurkar/data/GLI/validation/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
export OUTDIR=~/sjaipurkar/projects/mri_realism/runs/ae3d_run_big_v2

export CKPT=~/sjaipurkar/projects/mri_realism/runs/ae3d_run_big_v2/checkpoint_epoch19.pt
export PATIENT_ID=BraTS-GLI-00015-000

Run Visualization
python scripts/viz_ae3d_recon.py

Output
ae3d_recon_<patient_id>.png

Displays:
	•	Original MRI
	•	Reconstructed MRI
	•	Axial, Coronal, Sagittal views

## 🧪 Refiner Model (Optional)
Train Refiner
python scripts/train_refiner3d.py

Compare Outputs
python scripts/viz_ae_refiner_compare.py

## 📈 Training Objective
The model uses L1 Reconstruction Loss:
L = || X - X̂ ||₁

Where:
	•	X = Original MRI
	•	X̂ = Reconstructed MRI


## 📸 Example Results
examples/
Includes reconstructed MRI slices from multiple validation patients.

## 🔁 Reproducibility Steps
	1.	Login to server
	2.	Activate conda environment
	3.	Select GPU
	4.	Set dataset path
	5.	Train model
	6.	Load checkpoint
	7.	Run validation
	8.	View reconstructed MRI outputs

## 🧠 Key Contributions
	•	Built a 3D autoencoder for volumetric MRI reconstruction
	•	Processed multi-modal MRI data (T1, T1c, T2, FLAIR)
	•	Designed full training and validation pipeline
	•	Achieved high-quality reconstruction with low loss (~0.006)
	•	Developed visualization tools for medical imaging

## 🔮 Future Work
	•	Variational Autoencoders (VAE)
	•	Diffusion-based MRI generation
	•	Tumor-aware reconstruction
	•	Clinical evaluation metrics

## 📦 Requirements
torch
torchvision
numpy
matplotlib
nibabel
tqdm

## ⚠️ Notes
	•	Dataset is not included due to size
	•	Requires GPU (48GB VRAM recommended)
	•	Use tmux for long training sessions

## MINIMUM REQUIREMENTS:

- GPU: 16 GB (slow but works)
- RAM: 16 GB
- CPU: 4 cores

## RECOMMENDED:

- GPU: 48 GB
- RAM: 32–64 GB
- Multi-core CPU


## EXPECTED RUN TIME:
- Training: 2–5 hours
- Inference: 10–30 seconds per patient

## 👨‍💻 Author

Shantanu Jaipurkar
MS Information Systems | AI


# 🧠 Why 3D Autoencoder for MRI Reconstruction?

### 🔍 Problem Context

Brain MRI data is inherently **3D volumetric** and consists of multiple modalities (T1, T1c, T2, FLAIR). Each scan captures spatial relationships across depth (slices), which are critical for identifying anatomical structures and tumor regions.

Traditional 2D models process each slice independently, which leads to:
- Loss of spatial continuity across slices  
- Poor reconstruction of volumetric structures  
- Inconsistent anatomical representation  

To address this, a **3D model** is required.

---

### 🧠 Why Autoencoder?

An **autoencoder** is a neural network designed to learn a compressed representation of input data and reconstruct it.

It consists of:

- **Encoder** → Compresses input MRI into latent representation  
- **Decoder** → Reconstructs MRI from compressed features  

This makes it ideal for:

- Learning underlying anatomical structure  
- Removing noise while preserving key features  
- Reconstructing high-dimensional medical images  

---

### ⚙️ Why 3D Convolutional Autoencoder?

We specifically use a **3D Convolutional Autoencoder** because:

#### ✅ 1. Preserves Spatial Structure
3D convolutions operate across height, width, and depth:
- Captures relationships between slices  
- Maintains anatomical continuity  

#### ✅ 2. Multi-Modal Learning
Input includes 4 MRI modalities:
- T1  
- T1c  
- T2  
- FLAIR  

The model learns joint representations across all modalities.

#### ✅ 3. Better Feature Extraction
3D kernels capture:
- Tumor shape and volume  
- Tissue boundaries  
- Structural patterns  

---

### 🧩 Model Working (Step-by-Step)

1. **Input Tensor**
   - Shape: `[4, H, W, D]`
   - 4 channels = MRI modalities  

2. **Encoding Phase**
   - Series of 3D convolution layers  
   - Downsampling reduces spatial dimensions  
   - Extracts high-level features  

3. **Latent Representation**
   - Compact representation of MRI  
   - Contains structural + semantic information  

4. **Decoding Phase**
   - Transposed convolutions (upsampling)  
   - Reconstructs original volume  

5. **Output**
   - Reconstructed MRI volume  
   - Same shape as input  

---

### 📉 Why L1 Loss?

We use **L1 Reconstruction Loss**:

L = || X - X̂ ||₁

Where:
- X = Original MRI  
- X̂ = Reconstructed MRI  

#### Advantages:
- Preserves sharp edges  
- Reduces blurring (compared to L2 loss)  
- Better for medical imaging where structure matters  

---

### 📊 Why This Works Well for This Problem

| Requirement | Solution |
|------------|---------|
| Preserve 3D structure | 3D convolutions |
| Multi-modal input | 4-channel input |
| High-dimensional data | Autoencoder compression |
| Structural accuracy | L1 loss |
| Noise reduction | Encoder-decoder learning |

---

### 🚀 Key Advantages of This Approach

- Captures full volumetric context  
- Learns meaningful latent representations  
- Produces smooth and realistic reconstructions  
- Works well with limited labeled data (unsupervised learning)  

---

### ⚠️ Limitations

- High GPU memory requirement (48GB recommended)  
- Training time is longer compared to 2D models  
- Reconstruction quality depends on dataset diversity  

---

### 🔮 Future Improvements

- Variational Autoencoder (VAE) for probabilistic modeling  
- Diffusion models for higher realism  
- Attention mechanisms for tumor-focused reconstruction  
- Hybrid 2D + 3D architectures  

---

### 🧠 Summary

A 3D Convolutional Autoencoder is chosen because it effectively captures **volumetric spatial relationships**, learns **multi-modal MRI features**, and reconstructs images with high structural fidelity — making it well-suited for medical imaging tasks like brain MRI reconstruction.






  
