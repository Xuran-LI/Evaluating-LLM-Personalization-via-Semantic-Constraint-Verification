import os
import json
import time
import numpy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MAX_RUNTIME_SEC = 3600

class NumpyEncoder(json.JSONEncoder):
    """
    Safely encodes NumPy data types (like float32) into native Python types
    to prevent JSON serialization TypeErrors.
    """

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class MetricLogger:
    def __init__(self):
        self.records = []

    def log(self, method, score, start_time, end_time, prompt_tokens=0, completion_tokens=0):
        self.records.append({
            "method": method,
            "score": score,
            "time_sec": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        })

    def summary(self):
        import pandas as pd
        if not self.records:
            return {"avg_time": 0, "avg_tokens": 0, "total_time": 0, "total_tokens": 0}

        df = pd.DataFrame(self.records)
        return {
            "avg_time": df["time_sec"].mean(),
            "avg_tokens": df["total_tokens"].mean() if "total_tokens" in df else 0,
            "total_time": df["time_sec"].sum(),
            "total_tokens": df["total_tokens"].sum() if "total_tokens" in df else 0
        }


def generate_evaluation_prompt(original_query, ground_truth, user_preference, ai_response):
    """
    Generates the LLM-as-a-judge prompt dynamically.
    """
    prompt = f"""You are an expert AI evaluator. Your task is to objectively grade an AI's response based on two distinct dimensions: Correctness and User-Preference Alignment.

                ### [INPUTS]
                1. Original Query: {original_query}
                2. Answer Ground Truth: {ground_truth}
                3. User Preferences: {user_preference}
                4. AI Response to Evaluate: {ai_response}
                
                ### [EVALUATION RUBRIC]
                Dimension 1: Correctness against Ground Truth (Score 1-5)
                Evaluate the factual accuracy and completeness of the response STRICTLY based on the provided Answer Ground Truth. Do not use outside knowledge.
                - 5: Perfectly aligned. The response contains all core information from the Ground Truth, comprehensively answers the query, and makes no contradictory claims.
                - 3: Partially aligned. The response captures the main idea of the Ground Truth but misses important nuances, includes minor contradictions, or adds unverified external details.
                - 1: Completely unaligned. The response directly contradicts the Ground Truth, is entirely irrelevant, or hallucinates major facts not present in the Ground Truth.
                
                Dimension 2: User-Preference Alignment (Score 1-5)
                Evaluate how well the response is tailored to the specific User Preferences provided. 
                - 5: Perfectly aligned. The tone, formatting, and complexity exactly match the user's preferences. It feels highly personalized.
                - 3: Somewhat aligned. It attempts to meet the preferences but slips into a generic tone, or misses specific formatting constraints.
                - 1: Completely unaligned. The response ignores the user's preferences, uses the wrong tone, or violates specific user constraints.
                
                ### [OUTPUT FORMAT]
                You must output your evaluation strictly in the following JSON format. Provide your reasoning before providing the final scores.
                
                {{
                  "correctness_evaluation": {{
                    "reasoning": "<Step-by-step analysis comparing the response strictly against the Ground Truth>",
                    "score": <1-5>
                  }},
                  "alignment_evaluation": {{
                    "reasoning": "<Step-by-step analysis of how well the response matched the User Preferences>",
                    "score": <1-5>
                  }},
                  "overall_summary": "<A 1-2 sentence summary of the response's overall quality>"
                }}"""

    return prompt


# def get_llm_response(llm, llm_tokenizer, test_prompt):
#     """
#     Passes a logic contract example to the LLM and calculates token consumption.
#     """
#     messages = [[{"role": "user", "content": p}] for p in test_prompt]
#
#     inputs = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, padding=True,
#                                                return_dict=True, return_tensors="pt").to(llm.device)
#
#     # Calculate input prompt tokens (batch size * sequence length)
#     prompt_tokens = inputs.input_ids.numel()
#
#     outputs = llm.generate(**inputs, max_new_tokens=1024, pad_token_id=llm_tokenizer.eos_token_id)
#
#     # Calculate total tokens and derive completion tokens
#     total_tokens = outputs.numel()
#     completion_tokens = total_tokens - prompt_tokens
#
#     full_output = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#     del inputs
#     del outputs
#     torch.cuda.empty_cache()
#
#     # Return the texts AND the token metrics
#     return full_output, prompt_tokens, completion_tokens


def get_llm_response(llm, llm_tokenizer, test_prompts):
    """
    Processes prompts sequentially to avoid Falcon's broken 4D attention mask batch behavior.
    """
    full_outputs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for p in test_prompts:
        messages = [[{"role": "user", "content": p}]]

        # Unbatched text processing: no padding artifacts or mask generation errors
        inputs = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(llm.device)

        prompt_tokens = inputs.input_ids.numel()
        total_prompt_tokens += prompt_tokens

        with torch.no_grad():
            outputs = llm.generate(**inputs, max_new_tokens=1024, pad_token_id=llm_tokenizer.eos_token_id)

        total_tokens = outputs.numel()
        total_completion_tokens += (total_tokens - prompt_tokens)

        decoded = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        full_outputs.extend(decoded)

        del inputs
        del outputs

    torch.cuda.empty_cache()
    return full_outputs, total_prompt_tokens, total_completion_tokens

if __name__ == "__main__":
    import pandas as pd

    test_user_file = "./mmlu_dataset/PG_Benchmark/user_profiles.json"
    with open(test_user_file) as user_file:
        user_data = json.load(user_file)

    model_name = "tiiuae/Falcon-H1R-7B"
    Qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
    Qwen_tokenizer.padding_side = "left"

    if Qwen_tokenizer.pad_token is None:
        Qwen_tokenizer.pad_token = Qwen_tokenizer.eos_token

    Qwen_model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0})
    # Initialize the metric logger for the current model
    model_logger = MetricLogger()
    # Global list to compile final timing data for all models
    overall_timing_records = []

    for i in range(10):
        global_start_time = time.time()
        test_data_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered.json"
        if not os.path.exists(test_data_file):
            continue
        with open(test_data_file) as query_file:
            test_data = json.load(query_file)
        results = []
        for test_item in test_data:
            general_responses = test_item['general_response']
            personal_response = test_item['personalized_response']
            sycophancy_response = test_item['sycophancy_response']
            failure_response = test_item['failure_response']
            user_preference = test_item['user_related_interest']
            answer_sentence = test_item['choices'][test_item['answer']]
            query_sentence = test_item['question']

            p_p = generate_evaluation_prompt(query_sentence, answer_sentence, user_preference, personal_response)
            g_p = generate_evaluation_prompt(query_sentence, answer_sentence, user_preference, general_responses)
            s_p = generate_evaluation_prompt(query_sentence, answer_sentence, user_preference, sycophancy_response)
            f_p = generate_evaluation_prompt(query_sentence, answer_sentence, user_preference, failure_response)
            # Record start time
            start_time = time.time()
            # Call the LLM and unpack the returned metrics
            llm_judge_result, p_tokens, c_tokens = get_llm_response(Qwen_model, Qwen_tokenizer, [p_p, g_p, s_p, f_p])
            # Record end time
            end_time = time.time()
            # Log the performance (setting score to 0 since it is qualitative output)
            model_logger.log( method="LLM_Judge", score=0, start_time=start_time, end_time=end_time, prompt_tokens=p_tokens, completion_tokens=c_tokens)
            # Append results dictionary as before
            test_item["llm_judge_result"] = llm_judge_result
            results.append(test_item)

            current_elapsed_time = time.time() - global_start_time
            if current_elapsed_time > MAX_RUNTIME_SEC:
                print(f"⚠️ Warning: Maximum runtime of {MAX_RUNTIME_SEC}s exceeded.")
                print(f"Stopping at item {i}. Saving current progress...")
                break  # Exits the loop immediately

        out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_Falcon.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        print(f"User {i} LLM Judge check complete! Saved to: {out_path}")

        # Compute summary for this user chunk and append to global metrics
        stats = model_logger.summary()
        overall_timing_records.append({ "model_name": "Qwen_Judge", "avg_time_per_check_sec": stats["avg_time"], "total_execution_time_sec": stats["total_time"], "avg_tokens_per_check": stats["avg_tokens"], "total_tokens": stats["total_tokens"], "total_checks_performed": len(model_logger.records) })

    # Save summary table as a CSV file
    summary_df = pd.DataFrame(overall_timing_records)
    summary_csv_path = "./mmlu_dataset/PG_Benchmark/llm_models_timing_summary_Falcon.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 40)
    print(f"Performance report saved to {summary_csv_path}")
    print(summary_df.to_string(index=False))
    print("=" * 40)








