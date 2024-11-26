# UIC-Capstone-Project

Project Overview:
As part of the IDS 560 Capstone Project for the Master’s in Business Analytics program at the University of Illinois Chicago, we collaborated with CCC Intelligent Solutions to design and develop a sophisticated AI model for **Vehicle Damage Image Generation**. Leveraging state-of-the-art machine learning techniques, we implemented Fine-Tuning of the **Stable Diffusion Model (RunwayML V1.5)** utilizing **LoRA (Low-Rank Adaptation)** to achieve precise and scalable image generation results. This project demonstrates advanced capabilities in generative AI for real-world applications in the automotive and insurance industries.

# Data
CarDD dataset: Obtained from a research paper “A New Dataset for Vision-based Car Damage Detection” by Xinkuang Wang, Wenjing Li, and Zhongcheng Wu. 
Link to Download: https://cardd-ustc.github.io/

- Utilized a subset of the above dataset to fine-tune our stable diffusion model.
- Considered only 4 vehicle damages as of now. They are: **Flat Tires**, **Dents**, **Scratches**, and **Glass Shatter**.
- Separated each vehicle damage image to its respective folder.



# Implementation

## Install Dependencies:

   Install necessary libraries:
   ```bash
   pip install requirements.txt
   ```

   Clone **Diffusers** repository
   ```bash
   pip install git+https://github.com/huggingface/diffusers.git
   ```
## Steps to Prepare the Image Dataset for Fine-Tuning

### 1. Run the Notebook
- Execute the `Image Captioning for Damage.ipynb` notebook to prepare the image dataset for fine-tuning.  
- All subsequent steps (Points 2–5) are implemented within this notebook.

### 2. Generate Image Captions
- Using the **Salesforce/blip-image-captioning-large** model from HuggingFace, captions are automatically generated for each image in the dataset.

### 3. Save Captions to `metadata.csv`
- The generated captions are saved in a `metadata.csv` file in the following format:

  | file_name    | text                      |
  |--------------|------------------------------|
  | image1.jpg   | Caption for image1...        |
  | image2.jpg   | Caption for image2...        |

### 4. Create an Image Dataset
- The `datasets` library is used to create an image dataset:

  ```python
  from datasets import load_dataset

  dataset = load_dataset('imagefolder', data_dir='PATH_TO_FOLDER', split='train')
  ```
### 5 Push dataset to HuggingFace
The dataset is uploaded to the HuggingFace Hub:
```python
from huggingface_hub import login
# Log in with your HuggingFace token
login(token='YOUR_HUGGINGFACE_TOKEN')
# Push the dataset to the HuggingFace Hub
dataset.push_to_hub('NAME_OF_DATASET', private=True)
```


     
   

