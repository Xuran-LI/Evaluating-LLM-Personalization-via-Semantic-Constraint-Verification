import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# def get_llm_response(llm, llm_tokenizer, test_prompt):
#     """
#     Passes a logic contract example to Falcon to verify outcomes.
#     """
#     messages = [[{"role": "user", "content": p}] for p in test_prompt ]
#     # messages = [{"role": "user", "content": test_prompt}]
#     inputs = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, padding=True, return_dict=True, return_tensors="pt").to(llm.device)
#     outputs = llm.generate(**inputs, max_new_tokens=1024, pad_token_id=llm_tokenizer.eos_token_id)
#     full_output = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     del inputs
#     del outputs
#     torch.cuda.empty_cache()
#     return full_output


def get_llm_response(llm, llm_tokenizer, test_prompts):
    """
    Passes a batch of prompts to the LLM and returns the generated texts.
    """
    messages = [[{"role": "user", "content": p}] for p in test_prompts]

    # Tokenize the batch
    inputs = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, padding=True, return_dict=True, return_tensors="pt").to(llm.device)
    # Get the length of the input so we can slice out the prompt from the output
    input_lengths = [len(inp) for inp in inputs["input_ids"]]
    # Generate
    with torch.no_grad():  # Ensure gradients aren't calculated during inference
        outputs = llm.generate(**inputs,  max_new_tokens=1024,  pad_token_id=llm_tokenizer.eos_token_id, do_sample=False)

    # Slice the outputs to remove the prompt tokens and decode only the new generation
    generated_texts = []
    for i, output in enumerate(outputs):
        generated_tokens = output[input_lengths[i]:]
        text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(text)
    return generated_texts


if __name__ == "__main__":
    # 加载LLM模型
    # model_name = "tiiuae/Falcon-H1R-7B"
    model_name = "Qwen/Qwen3-8B-Base"
    Qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
    Qwen_tokenizer.padding_side = "left"
    Qwen_model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, )

    # 加载测试数据
    for i in range(5):
        test_data_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i+5}_prompt.json"
        with open(test_data_file) as test_file:
            test_data = json.load(test_file)

        for ii in range(len(test_data)):
            print(f"user {i+5}: {ii}th test item")
            test_item = test_data[ii]
            prompts= test_item['prompts']
            torch.cuda.empty_cache()
            llm_responses = get_llm_response(Qwen_model, Qwen_tokenizer, prompts)
            test_item['llm_responses'] = llm_responses

        output_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i+5}_results.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=4)
        print(f"User {i} generation complete! Saved successfully to: {output_file}")
