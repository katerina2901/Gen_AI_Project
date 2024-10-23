import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

from transformers import Trainer, TrainingArguments
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import concatenate_datasets

from utils.utils import read_video_decord, prepare_small_dataset


n_frames = 16

MODEL_ID = "llava-hf/LLaVa-NeXT-Video-7b-hf"
REPO_ID = "sm47466863/LLaVa-NeXT-Video-GENAI-FT-Project"

USE_LORA = False
USE_QLORA = True


processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False, cache_dir='./cache_processor/')
processor.tokenizer.padding_side = "right"

# dataset preparation 
datasets_combined = prepare_small_dataset()
dataset_processed = concatenate_datasets(datasets_combined)
dataset_processed = dataset_processed.shuffle(seed=42)
dataset = dataset_processed.train_test_split(test_size=0.2)

train_dataset, test_dataset = dataset['train'].with_format("torch"), dataset['test'].with_format("torch")


class LlavaNextVideoDataCollatorWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": [feat['input_ids'][0] for feat in features],
                "attention_mask": [feat['attention_mask'][0] for feat in features],
            },
            padding=True,
            return_tensors="pt",
        )

        labels = padded_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        padded_inputs["labels"] = labels
        padded_inputs["pixel_values_videos"] = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)

        return padded_inputs
    
    
if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="./model_/"
    )
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

args = TrainingArguments(
    output_dir = "./output/",
    eval_strategy = 'steps',
    eval_steps=20,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-05,
    max_steps = 100,
    lr_scheduler_type = 'cosine',
    warmup_ratio = 0.1,

    logging_steps = 20,
    save_strategy = 'steps',
    save_steps=20,
    save_total_limit = 1,
    fp16 = True,
    fp16_full_eval = True,
    optim = 'adamw_bnb_8bit',
    hub_model_id = REPO_ID,
    push_to_hub = True,

    label_names=["labels"], # pass label because QLORA with peft not save it in signature 
    dataloader_num_workers=4,
)

trainer = Trainer(
    model = model,
    tokenizer = processor,
    data_collator = LlavaNextVideoDataCollatorWithPadding(processor=processor),
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    args=args,
)

trainer.train()

trainer.model.push_to_hub(REPO_ID)