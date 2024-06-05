# Pretrain & Finetune a LlavaMistral model using Mixtral 8 x 7B generated data

## Steps

### Install git-lfs
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

git lfs install  # checks if the installation is successful
```

### Download datasets
1. COCO
```bash
mkdir coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d LlavaMistral/coco/
```

2. LLaVA-CC3M-Pretrain-595K
```bash
git lfs install
git-lfs clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain/
mkdir images
unzip images.zip -d images
```

3. LLaVA-Instruct-150K
```bash
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
```

### Prepare Mixtral 8 x 7B generated data
1. follow the [instructions here](https://github.com/yueying-teng/generate-language-image-instruction-following-data/tree/main?tab=readme-ov-file#steps)

2. after running the post processing script you will see these new files
    - `generated_data/complex_reasoning_77k.json`
    - `generated_data/conversation_58k.json`
    - `generated_data/detail_23k.json`
    - `generated_data/mistral_generated_llava_instruct_150k.json`

3. `generated_data/mistral_generated_llava_instruct_150k.json` is needed for finetuning and make sure it's stored like this `./LlavaMistral/generated_data/mistral_generated_llava_instruct_150k.json`


### Download Mistral 7B model
```bash
cd checkpoints
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
```

### Set up WandB
1. Create [a wandb account](https://wandb.ai/site)
2. Log in to your wandb account [using command line](https://wandb.ai/site)
3. You will be able to see training metrics and system resource usage under the proejct called huggingface


### Training
1. Pretraining
```bash
nohup bash scripts/v1_5/inhouse_pretrain.sh > inhouse_pretrain.out &
```
2. Finetuing with LoRA
```bash
# finetuning
nohup bash scripts/v1_5/inhouse_finetune.sh > inhouse_finetune.out &
```
3. Merge LoRA weights
```bash
python scripts/merge_lora_weights.py     --model-path ./checkpoints/llava-v1.5-vicuna-7b-v1.3-finetune_lora     --model-base ./checkpoints/vicuna-7b-v1.3     --save-model-path ./checkpoints/merged_llava-v1.5-vicuna-7b-v1.3-finetune_lora
```

