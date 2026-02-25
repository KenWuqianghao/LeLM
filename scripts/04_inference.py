"""Load the fine-tuned adapter and generate NBA hot takes."""

from pathlib import Path

import yaml
from unsloth import FastLanguageModel

CONFIG_PATH = Path("configs/train_config.yaml")

DEMO_PROMPTS = [
    "Give me your hottest LeBron James take.",
    "Is Nikola Jokic the best player in the NBA right now?",
    "Who's the most overrated player in the league?",
    "Give me your boldest Finals prediction.",
    "What's your most controversial take about Michael Jordan's legacy?",
    "Is the 3-point revolution ruining basketball?",
    "Who's going to win MVP this season and why?",
    "What's the worst contract in the NBA right now?",
]


def load_model():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    system_prompt = Path(cfg["data"]["system_prompt_path"]).read_text().strip()

    print(f"Loading model: {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["output"]["dir"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer, system_prompt


def generate(model, tokenizer, system_prompt, user_input, max_new_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)
    return response.strip()


def run_demos(model, tokenizer, system_prompt):
    print("\n" + "=" * 60)
    print("LeLM — NBA Hot Take Generator")
    print("=" * 60)

    for prompt in DEMO_PROMPTS:
        print(f"\n>> {prompt}")
        print("-" * 40)
        response = generate(model, tokenizer, system_prompt, prompt)
        print(response)
        print()


def repl(model, tokenizer, system_prompt):
    print("\n" + "=" * 60)
    print("Interactive mode — type your prompt (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        response = generate(model, tokenizer, system_prompt, user_input)
        print(f"\n{response}")


def main():
    model, tokenizer, system_prompt = load_model()
    run_demos(model, tokenizer, system_prompt)
    repl(model, tokenizer, system_prompt)


if __name__ == "__main__":
    main()
