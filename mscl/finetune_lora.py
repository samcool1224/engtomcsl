"""QLoRA fine-tuning for the English->MSCL parser (run on your GPU).

Trains LoRA adapters on top of the 4-bit-quantized base model so it learns to:
  1) use the exact 17-type vocabulary,
  2) EMIT CHOICE nodes on ambiguous input (the thing few-shot fails at),
  3) output the JSON format directly from a SHORT prompt (no few-shot, no schema)
     -> inference gets much faster after fine-tuning.

Requirements (Colab):  pip install peft trl datasets bitsandbytes transformers accelerate

Usage:
    python examples/finetune_lora.py                       # uses examples/train.jsonl
    python examples/finetune_lora.py --train big.jsonl --n_extra 1500
        (--n_extra generates extra synthetic samples first; recommended: 120 is small)

Output: adapters in ./mscl_lora/  -> load at inference with
    LocalBackend("Qwen/Qwen2.5-7B-Instruct", quantize="4bit", adapter_path="./mscl_lora")
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# The COMPACT prompt used for fine-tuning AND post-finetune inference.
# Short: instruction + types + object table. The model learns the output format
# from the targets, so no schema dump or few-shot examples are needed.
# ---------------------------------------------------------------------------
FT_INSTRUCTION = (
    "Translate the interior-design instruction into MSCL-SPRING JSON "
    '({"objects":[...],"formula":<node>}). Legal types: chair, couch, potted plant, bed, '
    "mirror, dining table, window, desk, toilet, door, TV, microwave, oven, toaster, sink, "
    "refrigerator, blender. If the English is ambiguous (vague direction like 'by/near', "
    "emphasized distance like 'well to the left', ambiguous reference, or an unsupported "
    "object word), emit a CHOICE node instead of guessing."
)

def ft_prompt(english: str, objects: list) -> str:
    return (FT_INSTRUCTION
            + "\n\nDETECTED OBJECTS: " + json.dumps(objects)
            + "\nINSTRUCTION: " + english
            + "\nJSON:")


def build_dataset(train_path: str, tokenizer, max_len: int = 1600):
    """(prompt, completion) pairs with completion-only loss masking."""
    from datasets import Dataset
    rows = [json.loads(l) for l in open(train_path)]
    examples = []
    for r in rows:
        prompt = ft_prompt(r["english"], r["objects"])
        target = json.dumps({"objects": r["objects"], "formula": r["formula"]})
        msgs_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        full = msgs_prompt + target + tokenizer.eos_token
        # tokenize; mask loss on the prompt part
        ids_full = tokenizer(full, truncation=True, max_length=max_len)["input_ids"]
        ids_prompt = tokenizer(msgs_prompt, truncation=True, max_length=max_len)["input_ids"]
        labels = list(ids_full)
        for i in range(min(len(ids_prompt), len(labels))):
            labels[i] = -100
        examples.append({"input_ids": ids_full, "labels": labels,
                         "attention_mask": [1] * len(ids_full)})
    return Dataset.from_list(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default=os.path.join(os.path.dirname(__file__), "train.jsonl"))
    ap.add_argument("--out", default="./mscl_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--n_extra", type=int, default=0,
                    help="generate this many EXTRA synthetic samples and append (recommended: 1000+)")
    args = ap.parse_args()

    # optionally enlarge the training set first (120 is small for fine-tuning)
    train_path = args.train
    if args.n_extra > 0:
        from mscl.datagen import generate_dataset, to_jsonl
        print(f"generating {args.n_extra} extra samples...")
        extra = generate_dataset(n=args.n_extra, ambiguous_frac=0.5, seed=7)
        big = open(train_path).read().rstrip("\n") + "\n" + to_jsonl(extra)
        train_path = train_path.replace(".jsonl", f"_plus{args.n_extra}.jsonl")
        open(train_path, "w").write(big)
        print(f"wrote {train_path}")

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments, Trainer,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    print("loading base model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("building dataset...")
    ds = build_dataset(train_path, tokenizer)
    print(f"train examples: {len(ds)}")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, report_to="none")

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
    print("training...")
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved LoRA adapters to {args.out}")
    print("Load at inference with:")
    print(f'  LocalBackend("{args.model}", quantize="4bit", adapter_path="{args.out}")')


if __name__ == "__main__":
    main()
