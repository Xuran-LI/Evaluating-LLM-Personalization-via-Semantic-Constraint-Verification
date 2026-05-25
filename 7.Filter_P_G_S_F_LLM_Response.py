import json
import logging
import os
import re

import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


# def analysis_llm_response(llm_response):
#     personalized_json_string = llm_response[0].split("assistant\n")[-1].strip()
#     personalized_json_string = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', personalized_json_string)
#     personalized_evaluation_data = json.loads(personalized_json_string)
#     personalized_correctness_score = personalized_evaluation_data["correctness_evaluation"]["score"]
#     personalized_alignment_score = personalized_evaluation_data["alignment_evaluation"]["score"]
#     personalized =[personalized_correctness_score, personalized_alignment_score]
#
#     generalized_json_string = llm_response[1].split("assistant\n")[-1].strip()
#     generalized_json_string = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', generalized_json_string)
#     generalized_evaluation_data = json.loads(generalized_json_string)
#     generalized_correctness_score = generalized_evaluation_data["correctness_evaluation"]["score"]
#     generalized_alignment_score = generalized_evaluation_data["alignment_evaluation"]["score"]
#     generalized =[generalized_correctness_score, generalized_alignment_score]
#
#     sycophancy_json_string = llm_response[2].split("assistant\n")[-1].strip()
#     sycophancy_json_string = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', sycophancy_json_string)
#     sycophanted_evaluation_data = json.loads(sycophancy_json_string)
#     sycophancy_correctness_score = sycophanted_evaluation_data["correctness_evaluation"]["score"]
#     sycophancy_alignment_score = sycophanted_evaluation_data["alignment_evaluation"]["score"]
#     sycophancy=[sycophancy_correctness_score, sycophancy_alignment_score]
#
#     failure_json_string = llm_response[3].split("assistant\n")[-1].strip()
#     failure_json_string = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', failure_json_string)
#     failure_evaluation_data = json.loads(failure_json_string)
#     failure_correctness_score = failure_evaluation_data["correctness_evaluation"]["score"]
#     failure_alignment_score = failure_evaluation_data["alignment_evaluation"]["score"]
#     failure =[failure_correctness_score, failure_alignment_score]
#
#     return personalized, generalized,sycophancy,failure

def extract_scores(llm_response_text, default_score=0):
    """
    Helper function to safely extract scores from a single LLM response.
    Catches parsing errors, missing keys, and invalid formats.
    """
    if not isinstance(llm_response_text, str):
        logging.warning("LLM response is not a string. Returning default scores.")
        return [default_score, default_score]

    try:
        # 1. Isolate the JSON
        if "assistant\n" in llm_response_text:
            json_string = llm_response_text.split("assistant\n")[-1].strip()
        else:
            # Fallback in case the LLM didn't output the delimiter but just the JSON
            json_string = llm_response_text.strip()

        # 2. Fix invalid LaTeX/Markdown escapes
        json_string = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_string)

        # 3. Parse JSON safely
        evaluation_data = json.loads(json_string)

        # 4. Safely extract scores using .get() to avoid KeyErrors
        # If the key doesn't exist, it falls back to an empty dict {}, then falls back to default_score
        c_score = evaluation_data.get("correctness_evaluation", {}).get("score", default_score)
        a_score = evaluation_data.get("alignment_evaluation", {}).get("score", default_score)

        return [c_score, a_score]

    except json.decoder.JSONDecodeError as e:
        logging.error(f"JSON Parse Error: {e}. Returning default scores.")
        logging.debug(f"Broken JSON String: {json_string}")
        return [default_score, default_score]
    except Exception as e:
        logging.error(f"Unexpected error extracting scores: {e}. Returning default scores.")
        return [default_score, default_score]


def analysis_llm_response(llm_response):
    """
    Analyzes the 4 distinct evaluation dimensions from the LLM response array.
    """
    # Ensure we actually have 4 responses to prevent IndexError
    if not llm_response or len(llm_response) < 4:
        logging.error(f"Expected 4 LLM responses, got {len(llm_response) if llm_response else 0}.")
        # Return defaults for all if the input array is completely broken
        return [0, 0], [0, 0], [0, 0], [0, 0]

    personalized = extract_scores(llm_response[0])
    generalized = extract_scores(llm_response[1])
    sycophancy = extract_scores(llm_response[2])
    failure = extract_scores(llm_response[3])

    return personalized, generalized, sycophancy, failure

# if __name__ == "__main__":
#         for i in range(10):
#             test_llm_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge.json"
#             if not os.path.exists(test_llm_file):
#                 continue
#             with open(test_llm_file) as llm_response_file:
#                 test_llm_data = json.load(llm_response_file)
#             total_items = len(test_llm_data)
#             if total_items == 0:
#                 continue
#             results = []
#             for test_item in test_llm_data:
#                 llm_judge_result= test_item["llm_judge_result"]
#                 personalized, generalized,sycophancy,failure= analysis_llm_response(llm_judge_result)
#
#                 test_item["general_correct"] = generalized[0]/5
#                 test_item["general_align"] =generalized[1]/5
#                 test_item["personal_correct"] = personalized[0]/5
#                 test_item["personal_align"] = personalized[1]/5
#                 test_item["sycophancy_correct"] = sycophancy[0]/5
#                 test_item["sycophancy_align"] = sycophancy[1]/5
#                 test_item["failure_correct"] = failure[0]/5
#                 test_item["failure_align"] = failure[1]/5
#                 results.append(test_item)
#
#             out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_filter.json"
#             os.makedirs(os.path.dirname(out_path), exist_ok=True)
#             with open(out_path, 'w') as f:
#                 json.dump(results, f, indent=4, cls=NumpyEncoder)
#             print(f"User {i} LLM Judge check complete! Saved to: {out_path}")


def extract_scores_deepseek(raw_text, default_score=0):
    # 1. Safely handle empty text
    if not raw_text or not isinstance(raw_text, str):
        return [default_score, default_score]

    # 2. Clean raw text
    cleaned_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")

    c_score = None
    a_score = None
    json_string = ""  # <--- FIXED: Initializing here prevents the UnboundLocalError

    # ==========================================
    # PHASE 1: Attempt JSON Parsing
    # ==========================================
    match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)

    if match:
        json_string = match.group(1)
        # Fix LLM JSON escaping hallucinations
        json_string = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_string)

        try:
            data = json.loads(json_string, strict=False)
            c_val = data.get("correctness_evaluation", {}).get("score")
            a_val = data.get("alignment_evaluation", {}).get("score")
            if c_val is not None: c_score = c_val
            if a_val is not None: a_score = a_val
        except Exception as e:
            # We silently pass here so it can smoothly transition to Phase 2/3
            pass

            # ==========================================
    # PHASE 2: Broken/Truncated JSON Regex
    # ==========================================
    if c_score is None:
        c_match = re.search(r'"correctness_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if c_match: c_score = c_match.group(1)

    if a_score is None:
        a_match = re.search(r'"alignment_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if a_match: a_score = a_match.group(1)

    # ==========================================
    # PHASE 3: Pure Natural Language Regex
    # ==========================================
    if c_score is None:
        c_match = re.search(r'correctness(?:[\s\S]{0,30}?)(?:score|:)[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if c_match: c_score = c_match.group(1)

    if a_score is None:
        a_match = re.search(r'alignment(?:[\s\S]{0,30}?)(?:score|:)[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if a_match: a_score = a_match.group(1)

    # ==========================================
    # Final Validation
    # ==========================================
    if c_score is None: c_score = default_score
    if a_score is None: a_score = default_score

    return [int(c_score), int(a_score)]


def analysis_llm_response_deepseek(llm_response):
    """
    Analyzes the 4 distinct evaluation dimensions from the LLM response array.
    """
    # Ensure we actually have 4 responses to prevent IndexError
    if not llm_response or len(llm_response) < 4:
        logging.error(f"Expected 4 LLM responses, got {len(llm_response) if llm_response else 0}.")
        return [0, 0], [0, 0], [0, 0], [0, 0]

    personalized = extract_scores_deepseek(llm_response[0])
    generalized = extract_scores_deepseek(llm_response[1])
    sycophancy = extract_scores_deepseek(llm_response[2])
    failure = extract_scores_deepseek(llm_response[3])

    return personalized, generalized, sycophancy, failure


if __name__ == "__main__":
    for i in range(10):
        test_llm_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_DeepSeek.json"

        if not os.path.exists(test_llm_file):
            continue

        with open(test_llm_file) as llm_response_file:
            test_llm_data = json.load(llm_response_file)

        total_items = len(test_llm_data)
        if total_items == 0:
            continue

        results = []
        for test_item in test_llm_data:
            llm_judge_result = test_item.get("llm_judge_result", [])
            personalized, generalized, sycophancy, failure = analysis_llm_response_deepseek(llm_judge_result)
            # Because extract_scores_deepseek now safely returns [0, 0] on error,
            # these calculations will never crash with a TypeError
            test_item["general_correct"] = generalized[0] / 5
            test_item["general_align"] = generalized[1] / 5
            test_item["personal_correct"] = personalized[0] / 5
            test_item["personal_align"] = personalized[1] / 5
            test_item["sycophancy_correct"] = sycophancy[0] / 5
            test_item["sycophancy_align"] = sycophancy[1] / 5
            test_item["failure_correct"] = failure[0] / 5
            test_item["failure_align"] = failure[1] / 5
            results.append(test_item)
        out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_filter_DeepSeek.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Note: Ensure NumpyEncoder is defined somewhere above in your actual script
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)  # Add cls=NumpyEncoder back if defined

        print(f"User {i} LLM Judge check complete! Saved to: {out_path}")


def extract_scores_falcon(raw_text, default_score=0):
    # 1. Safely handle empty text
    if not raw_text or not isinstance(raw_text, str):
        return [default_score, default_score]

    # 2. Clean raw BPE decoding errors
    cleaned_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")

    c_score = None
    a_score = None

    # ==========================================
    # ATTEMPT 1: Strict JSON Parsing
    # ==========================================
    match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
    if match:
        json_string = match.group(1)
        json_string = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_string)
        try:
            data = json.loads(json_string, strict=False)
            c_val = data.get("correctness_evaluation", {}).get("score")
            a_val = data.get("alignment_evaluation", {}).get("score")
            if c_val is not None: c_score = c_val
            if a_val is not None: a_score = a_val
        except json.JSONDecodeError:
            pass  # JSON failed (likely truncated). Move to Attempt 2.

    # ==========================================
    # ATTEMPT 2: Regex Fallback in JSON Structure
    # ==========================================
    # If JSON parsing failed, hunt for the numbers directly near the JSON keys.
    if c_score is None:
        c_match = re.search(r'"correctness_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if c_match: c_score = c_match.group(1)

    if a_score is None:
        a_match = re.search(r'"alignment_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if a_match: a_score = a_match.group(1)

    # ==========================================
    # ATTEMPT 3: Natural Language Fallback (<think> block text)
    # ==========================================
    # If the JSON was completely truncated before the scores were written,
    # we look for text like "correctness score 1" or "alignment score is 3".

    if c_score is None:
        # Matches "correctness score", ignores non-numbers, captures the first 1-5 it hits
        c_matches = re.findall(r'correctness\s+score[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if c_matches:
            # Take the LAST match, as the LLM's final conclusion is usually at the end
            c_score = c_matches[-1]

    if a_score is None:
        a_matches = re.findall(r'alignment\s+score[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if a_matches:
            # Take the LAST match
            a_score = a_matches[-1]

    # ==========================================
    # Final Validation
    # ==========================================
    if c_score is None: c_score = default_score
    if a_score is None: a_score = default_score

    return [int(c_score), int(a_score)]


def analysis_llm_response_falcon(llm_response):
    """
    Analyzes the 4 distinct evaluation dimensions from the LLM response array.
    """
    # Ensure we actually have 4 responses to prevent IndexError
    if not llm_response or len(llm_response) < 4:
        logging.error(f"Expected 4 LLM responses, got {len(llm_response) if llm_response else 0}.")
        return [0, 0], [0, 0], [0, 0], [0, 0]

    personalized = extract_scores_falcon(llm_response[0])
    generalized = extract_scores_falcon(llm_response[1])
    sycophancy = extract_scores_falcon(llm_response[2])
    failure = extract_scores_falcon(llm_response[3])

    return personalized, generalized, sycophancy, failure


# if __name__ == "__main__":
#     for i in range(10):
#         test_llm_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_Falcon.json"
#
#         if not os.path.exists(test_llm_file):
#             continue
#
#         with open(test_llm_file) as llm_response_file:
#             test_llm_data = json.load(llm_response_file)
#
#         total_items = len(test_llm_data)
#         if total_items == 0:
#             continue
#
#         results = []
#         for test_item in test_llm_data:
#             llm_judge_result = test_item.get("llm_judge_result", [])
#             personalized, generalized, sycophancy, failure = analysis_llm_response_falcon(llm_judge_result)
#             # Because extract_scores_deepseek now safely returns [0, 0] on error,
#             # these calculations will never crash with a TypeError
#             test_item["general_correct"] = generalized[0] / 5
#             test_item["general_align"] = generalized[1] / 5
#             test_item["personal_correct"] = personalized[0] / 5
#             test_item["personal_align"] = personalized[1] / 5
#             test_item["sycophancy_correct"] = sycophancy[0] / 5
#             test_item["sycophancy_align"] = sycophancy[1] / 5
#             test_item["failure_correct"] = failure[0] / 5
#             test_item["failure_align"] = failure[1] / 5
#             results.append(test_item)
#         out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_filter_Falcon.json"
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         # Note: Ensure NumpyEncoder is defined somewhere above in your actual script
#         with open(out_path, 'w') as f:
#             json.dump(results, f, indent=4)  # Add cls=NumpyEncoder back if defined
#
#         print(f"User {i} LLM Judge check complete! Saved to: {out_path}")


def extract_scores_Llama(raw_text, default_score=0):
    """
    A bulletproof parser that handles perfect JSON, broken JSON, truncated JSON,
    missing backticks, and pure natural language prose.
    """
    if not raw_text or not isinstance(raw_text, str):
        return [default_score, default_score]

    cleaned_text = raw_text.replace("Ġ", " ").replace("Ċ", "\n")

    c_score = None
    a_score = None

    # ==========================================
    # PHASE 1: Attempt JSON Parsing
    # ==========================================
    # Matches outermost brackets whether backticks are present or not
    match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
    if match:
        json_string = match.group(1)
        # Sanitize hallucinated backslashes
        json_string = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', json_string)
        try:
            data = json.loads(json_string, strict=False)
            c_val = data.get("correctness_evaluation", {}).get("score")
            a_val = data.get("alignment_evaluation", {}).get("score")
            if c_val is not None: c_score = c_val
            if a_val is not None: a_score = a_val
        except json.JSONDecodeError:
            pass  # Move to Phase 2

    # ==========================================
    # PHASE 2: Broken/Truncated JSON Regex
    # ==========================================
    # If the JSON was cut off, hunt for the keys directly.
    if c_score is None:
        c_match = re.search(r'"correctness_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if c_match: c_score = c_match.group(1)

    if a_score is None:
        a_match = re.search(r'"alignment_evaluation"[\s\S]*?"score"\s*:\s*(\d+)', cleaned_text)
        if a_match: a_score = a_match.group(1)

    # ==========================================
    # PHASE 3: Pure Natural Language Regex (For LIMA style prose)
    # ==========================================
    # If the model ignored JSON entirely, hunt for natural language declarations.
    if c_score is None:
        # Matches "Correctness score a X" or "Correctness: X/5"
        c_match = re.search(r'correctness(?:[\s\S]{0,30}?)(?:score|:)[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if c_match: c_score = c_match.group(1)

    if a_score is None:
        # Matches "Alignment score a X" or "Alignment: X/5"
        a_match = re.search(r'alignment(?:[\s\S]{0,30}?)(?:score|:)[^0-9]*?([1-5])', cleaned_text, re.IGNORECASE)
        if a_match: a_score = a_match.group(1)

    # ==========================================
    # Final Validation
    # ==========================================
    if c_score is None: c_score = default_score
    if a_score is None: a_score = default_score

    return [int(c_score), int(a_score)]


def analysis_llm_response_Llama(llm_response):
    """
    Analyzes the 4 distinct evaluation dimensions from the LIMA LLM response array.
    """
    if not llm_response or len(llm_response) < 4:
        logging.error(f"Expected 4 LLM responses, got {len(llm_response) if llm_response else 0}.")
        return [0, 0], [0, 0], [0, 0], [0, 0]

    personalized = extract_scores_Llama(llm_response[0])
    generalized = extract_scores_Llama(llm_response[1])
    sycophancy = extract_scores_Llama(llm_response[2])
    failure = extract_scores_Llama(llm_response[3])

    return personalized, generalized, sycophancy, failure


# if __name__ == "__main__":
#     # Your standard loop integration
#     for i in range(10):
#         test_llm_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_Llama.json"
#
#         if not os.path.exists(test_llm_file):
#             continue
#
#         with open(test_llm_file) as llm_response_file:
#             test_llm_data = json.load(llm_response_file)
#
#         if len(test_llm_data) == 0:
#             continue
#
#         results = []
#         for test_item in test_llm_data:
#             llm_judge_result = test_item.get("llm_judge_result", [])
#             personalized, generalized, sycophancy, failure = analysis_llm_response_Llama(llm_judge_result)
#
#             test_item["general_correct"] = generalized[0] / 5
#             test_item["general_align"] = generalized[1] / 5
#             test_item["personal_correct"] = personalized[0] / 5
#             test_item["personal_align"] = personalized[1] / 5
#             test_item["sycophancy_correct"] = sycophancy[0] / 5
#             test_item["sycophancy_align"] = sycophancy[1] / 5
#             test_item["failure_correct"] = failure[0] / 5
#             test_item["failure_align"] = failure[1] / 5
#             results.append(test_item)
#
#         out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_LLM_Judge_filter_Llama.json"
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         with open(out_path, 'w') as f:
#             json.dump(results, f, indent=4)
#
#         print(f"User {i} LIMA Judge check complete! Saved to: {out_path}")