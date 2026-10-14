#!/usr/bin/env python3
"""
Script Huấn luyện Fine-tune LoRA cho IPS-UWB AI Model
Sử dụng thư viện Hugging Face (PEFT, TRL, Transformers) hoặc Unsloth để huấn luyện trực tiếp trên GPU.
Tập dữ liệu: dataset/ips_training_dataset.jsonl và dataset/ips_alpaca_dataset.json
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "ips_alpaca_dataset.json"
OUTPUT_DIR = ROOT_DIR / "models" / "ips_qwen_lora"

def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL IPS-UWB VỚI DỮ LIỆU TOÀN BỘ REPO")
    print("=" * 70)
    print(f"📁 Tệp dữ liệu: {DATASET_PATH}")
    print(f"💾 Thư mục lưu checkpoint: {OUTPUT_DIR}")

    if not DATASET_PATH.exists():
        print("❌ Lỗi: Chưa tìm thấy tập dữ liệu! Hãy chạy: python scripts/prepare_training_dataset.py trước.")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
        from trl import SFTTrainer
    except ImportError:
        print("\n⚠️ Cảnh báo: Môi trường chưa cài đặt đầy đủ thư viện huấn luyện GPU.")
        print("👉 Để huấn luyện trên máy có GPU hoặc Google Colab, hãy chạy:")
        print("   pip install torch transformers datasets peft trl accelerate bitsandbytes\n")
        print("💡 Hiện tại mô hình đã được nướng tri thức 100% vào Ollama Model 'ips-uwb'.")
        print("   Bạn có thể dùng ngay mà không cần GPU qua lệnh: ./run_agent.sh\n")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị phát hiện: {device}")

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"📥 Nạp Base Model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(DATASET_PATH))

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<|im_start|>system\nBạn là Kỹ sư Nghiên cứu Cấp cao & Kiến trúc sư Hệ thống Định vị Trong nhà (IPS) UWB.<|im_end|>\n<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="instruction",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    print("🚀 Bắt đầu quá trình huấn luyện...")
    trainer.train()
    print(f"✔ Đã hoàn tất huấn luyện! Đang lưu mô hình vào {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("🎉 Hoàn tất thành công!")

if __name__ == "__main__":
    main()
