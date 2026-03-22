# 🧠 MRI Realism using 3D Autoencoder

This project focuses on reconstructing realistic brain MRI volumes using a 3D Convolutional Autoencoder trained on the BraTS 2023 Glioma Dataset. The pipeline includes data loading, preprocessing, model training, validation, and visualization.

⸻

## 🚀 Project Overview
	•	Dataset: BraTS 2023 (Glioma MRI)
	•	Model: 3D Convolutional Autoencoder
	•	Goal: Reconstruct MRI volumes from multi-modal inputs
	•	Output: High-quality reconstructed MRI slices (Axial, Coronal, Sagittal)

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

## 👨‍💻 Author

Shantanu Jaipurkar
MS Information Systems | AI






  
