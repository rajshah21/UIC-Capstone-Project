# UIC-Capstone-Project

Project Overview:
As part of the IDS 560 Capstone Project for the Master’s in Business Analytics program at the University of Illinois Chicago, we collaborated with CCC Intelligent Solutions to design and develop a sophisticated AI model for **Vehicle Damage Image Generation**. Leveraging state-of-the-art machine learning techniques, we implemented Fine-Tuning of the **Stable Diffusion Model (RunwayML V1.5)** utilizing **LoRA (Low-Rank Adaptation)** to achieve precise and scalable image generation results. This project demonstrates advanced capabilities in generative AI for real-world applications in the automotive and insurance industries.

# Implementation

1. Install Dependencies:

   Install necessary libraries:
   ```bash
   pip install requirements.txt
   ```

   Clone **Diffusers** repository
   ```bash
   pip install git+https://github.com/huggingface/diffusers.git
   ```
2. Create Image Dataset for training.

   - Create a **metadata.csv** file in the folder where images are present to fine-tune on.
   - In the **metadata.csv** file, it should follow a format a below.
   - Create an image dataset as followig:
     ```bash
     from datasets import load_dataset
     dataset = load_dataset('imagefolder', data_dir = 'PATH_TO_FOLDER', split='train')
     ```
   - Load the image dataset to HuggingFace Hub.
     ```bash
     from huggingface_hub import login
     login(token='YOUR_HUGGINGFACE_TOKEN')
     dataset.push_to_hub('NAME_OF_DATASET', private=True)
     ```
     
   

