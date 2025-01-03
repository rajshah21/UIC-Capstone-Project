
accelerate launch train_text_to_image_lora.py `
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ` #DIFFUSION_MODEL_TO_FINETUNE
  --dataset_name="IDS-560-Group-3/Scratches-image-captions-v2" ` #PATH_TO_YOUR_DATASET_ON_HUGGINGFACE
  --dataloader_num_workers=0 `
  --resolution=512 `
  --center_crop `
  --random_flip `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --num_train_epochs=20 `
  --rank=16 `
  --learning_rate=1e-04 `
  --max_grad_norm=1 `
  --lr_scheduler="cosine" `
  --lr_warmup_steps=0 `
  --output_dir="C:\Users\Raj\Desktop\Fall 2024\Capstone Project\Synthetic Vehicle Images Damage Generation\finetune\scratches-v3" ` #PATH_TO_SAVE_LOCALLY
  --push_to_hub `
  --hub_model_id="stable-diffusion-scratches-lora-v3" ` #NAME_AFTER_TRAINING_TO_SAVE_ON_HUGGINGFACE
  --checkpointing_steps=500 `
  --validation_prompt="A car with scratches on side door." ` #VALIDATION_PROMPT
  --seed=21 `
  --mixed_precision='no'
