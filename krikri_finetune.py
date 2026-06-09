from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

# --- 1. Υπολογισμός & Ρυθμίσεις ---
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("=== Βήμα 1: Φόρτωση Μοντέλου ===")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ilsp/Llama-Krikri-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("=== Βήμα 2: Προσθήκη LoRA Adapters ===")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print("=== Βήμα 3: Φόρτωση & Προετοιμασία Δεδομένων ===")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
)

def format_chat(examples):
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in examples["conversations"]]
    return {"text": texts}

data_path = "unsloth_dataset.json" # Σιγουρέψου ότι αυτό είναι το όνομα του JSON αρχείου σου
dataset = load_dataset("json", data_files={"train": data_path}, split="train")
dataset = dataset.map(format_chat, batched=True)
print(f"Φορτώθηκαν επιτυχώς {len(dataset)} διάλογοι μουσικών προτάσεων!")

print("=== Βήμα 4: Δοκιμή Μοντέλου ΠΡΙΝ την εκπαίδευση ===")
FastLanguageModel.for_inference(model)
messages = [
    {"role": "user", "content": "Θέλω να ανακαλύψω νέους καλλιτέχνες, έχεις κάτι που είναι λίγο έντονο ή δραματικό;"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True)
print("Απάντηση βάσης (Base Model):\n", tokenizer.batch_decode(outputs)[0])

print("=== Βήμα 5: Εκκίνηση Εκπαίδευσης ===")
FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit" if load_in_4bit else "adamw_torch",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "music_training_outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()
print(f"\nΗ εκπαίδευση ολοκληρώθηκε σε {round(trainer_stats.metrics['train_runtime']/60, 2)} λεπτά.")

print("=== Βήμα 6: Αποθήκευση Εκπαιδευμένου Μοντέλου ===")
model.save_pretrained("music_krikri_lora")
tokenizer.save_pretrained("music_krikri_lora")
print("Όλα έτοιμα! Το μοντέλο αποθηκεύτηκε επιτυχώς.")

