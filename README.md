# UIC-Capstone-Project
## AI-Driven Image Generation: Leveraging Stable Diffusion and LoRA Fine-Tuning.
![Python](https://img.shields.io/badge/-Python-blue?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-red?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/-Hugging%20Face-FFD54F?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Diffusers](https://img.shields.io/badge/-Diffusers-0096FF?style=flat&logo=huggingface&logoColor=white)

Team Members: 
- **Raj Shah**: shahraj2100@gmail.com
- **Harsh Mehta**: hmehta1597@gmail.com
- **Bhavana Pinni**: pinnibhavana@gmail.com
- **Tomiwa Majekodunmi**: gabrielmajekodunmi@gmail.com
- **Kurshid Shaik**: madinakurshied@gmail.com

Project Overview:
As part of the IDS 560 Capstone Project for the Master’s in Business Analytics program at the University of Illinois Chicago, we collaborated with CCC Intelligent Solutions to design and develop a sophisticated AI model for **Vehicle Damage Image Generation**. Leveraging state-of-the-art machine learning techniques, we implemented Fine-Tuning of the **Stable Diffusion Model (RunwayML V1.5)** utilizing **LoRA (Low-Rank Adaptation)** to achieve precise and scalable image generation results. This project demonstrates advanced capabilities in generative AI for real-world applications in the automotive and insurance industries.

# 📂 Data
- CarDD dataset: Obtained from a research paper “A New Dataset for Vision-based Car Damage Detection” by Xinkuang Wang, Wenjing Li, and Zhongcheng Wu. 
- Link to Download: https://cardd-ustc.github.io/

- Utilized a subset of the above dataset to fine-tune our stable diffusion model.
- Considered only 4 vehicle damages as of now. They are: **Flat Tires**, **Dents**, **Scratches**, and **Glass Shatter**.
- Separated each vehicle damage image to its respective folder.



# 🛠 Implementation

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
- The dataset is uploaded to the HuggingFace Hub:
  ```python
   from huggingface_hub import login
   # Log in with your HuggingFace token
   login(token='YOUR_HUGGINGFACE_TOKEN')
   # Push the dataset to the HuggingFace Hub
   dataset.push_to_hub('NAME_OF_DATASET', private=True)
  ```
### 6. Repeat steps
- Repeat all the above steps in this section to create individual image datasets for each damage category.

## Fine-Tuning Stable Diffusion

### 1. Navigate to training script in Diffusers Repo:
- Once you have cloned the diffusers repo, navigate to `train_text_to_image_lora.py` file.
  ```bash
  cd ./diffusers/examples/text_to_image/
  ```
### 2. Setup Accelerate Environment:
- Set up an accelerate environment.
  ```bash
  accelerate config
  ```
- Configuring accelerate environment for GPU-based training where you can configure the following:
     1) GPUs to be used during training.
     2) Distributed Training
     3) Mixed Precision (fp16 or bp16)
     4) Optimization libraries (dynamo, DeepSpeed)
### 3. Execute Training Script:
- Copy the code from `train.sh` file and execute training.
- Make sure to change necessary parameters like `dataset_name`, `rank`, `num_training_epochs`, `output_dir`, `hub_model_id`, `validation_prompt`

### 4. Saving the Fine-Tuned Model
- Once training is completed, the fine-tuned model will be automatically pushed to your HuggingFace Hub with the `hub_model_id` specified.

### 5. Repeat Steps
- Repeat all the above steps in this section for each damage category.
- At the end, you will have individual LoRA models (Expert LoRAs) for each damage category.


# 🚀 Implementing UI

We have implemented a sample UI for this project using Streamlit. 
Here, a user can enter a `text prompt` and select the `adapter weights` for each damage to generate an image.

### 1. Execute the `app.py` file with the below command to implement the sample UI.
```bash
streamlit run app.py
```

![UI-Example](https://github.com/user-attachments/assets/ae73d809-4732-4ba8-8556-8a0bb3b82f78)



    



     
   

