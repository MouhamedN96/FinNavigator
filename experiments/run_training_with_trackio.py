import os
import torch
import json
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import trackio

# Initialize TrackIO run
run = trackio.init(project="FinNavigator-LLM", run_name="Qwen-3-VL-4B-Instruct-SFT")

# Custom Callback to log metrics to TrackIO
class TrackIOCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log all available metrics to TrackIO (e.g. loss, learning_rate, epoch)
            run.log(logs, step=state.global_step)

def format_prompts(examples):
    """Format the JSONL inputs into Alpaca conversational format expected by the model."""
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Must match the prompt style the model was evaluated/trained on
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        texts.append(prompt)
    return {"text": texts}

def main():
    print("🚀 Initializing FinNavigator Training Pipeline with TrackIO...")
    
    # 1. Load Dataset (Baseline + Synthetic Career Data)
    data_files = [
        "experiments/finnav_train.jsonl",
        "experiments/synthetic_career_data.jsonl"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in data_files if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError(f"No training data found in {data_files}")
        
    dataset = load_dataset("json", data_files=existing_files, split="train")
    
    # Split into train and eval (80/20 split)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Format the datasets
    train_dataset = dataset["train"].map(format_prompts, batched=True)
    eval_dataset = dataset["test"].map(format_prompts, batched=True)
    
    print(f"📊 Dataset Loaded: {len(train_dataset)} Train, {len(eval_dataset)} Eval")

    # 2. Setup Unsloth / Model
    # (Note: Unsloth requires Linux/CUDA. If running locally on Windows without CUDA, this will fail.
    #  This script is designed to run in your GPU environment.)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is not installed. Please run this script in an environment with unsloth installed (e.g. Colab or Linux GPU).")

    max_seq_length = 2048
    dtype = None # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-VL-4B-Instruct-bnb-4bit", 
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # Target attention layers
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Configure Trainer
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2, # Bumped to 2 for base experiment
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=args,
        # Inject TrackIO callback here
        callbacks=[TrackIOCallback()]
    )

    # Track model configuration
    run.log_params({
        "model": "FinnavQwen 3.5 4B",
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.num_train_epochs,
        "lora_r": 16,
        "lora_alpha": 16
    })

    # 4. Train Model
    print("🔥 Starting Fine-Tuning...")
    trainer_stats = trainer.train()
    
    # Finalize TrackIO run
    run.finish()
    
    # 5. Save the fine-tuned model
    print("💾 Saving LoRA Adapters...")
    model.save_pretrained("finnav_qwen3.5_4b_lora")
    tokenizer.save_pretrained("finnav_qwen3.5_4b_lora")
    print("✅ Training Complete!")

    # Push to Hub
    print("🚀 Pushing to Hugging Face Hub...")
    model.push_to_hub("MOH749/finnav_qwen3.5_4b_lora")
    tokenizer.push_to_hub("MOH749/finnav_qwen3.5_4b_lora")
    print("✅ Pushed successfully!")

if __name__ == "__main__":
    main()
