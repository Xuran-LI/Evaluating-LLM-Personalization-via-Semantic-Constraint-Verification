import os
import re
import json
import time
import numpy
import scipy.special
from sentence_transformers import CrossEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        self.records.append(
            {"method": method, "score": score, "time_sec": end_time - start_time, "prompt_tokens": prompt_tokens,
             "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens})

    def summary(self):
        import pandas as pd
        df = pd.DataFrame(self.records)
        return {"avg_time": df["time_sec"].mean(),
                "avg_tokens": df["total_tokens"].mean() if "total_tokens" in df else 0,
                "total_time": df["time_sec"].sum(),
                "total_tokens": df["total_tokens"].sum() if "total_tokens" in df else 0}


def timed_call(func, *args, method_name="method", logger=None):
    start = time.time()
    score = func(*args)
    end = time.time()
    if logger:
        logger.log(method_name, score, start, end)
    return score


class SemanticNLIExplainer:
    def __init__(self, model_name):
        """
        使用NLI模型进行语义逻辑验证
        """
        print(f"Loading CrossEncoder '{model_name}' for Semantic Ablation...")
        self.nli_verifier = CrossEncoder(model_name)
        self.entail_idx = self.nli_verifier.model.config.label2id.get('entailment')
        if self.entail_idx is None:
            for k, v in self.nli_verifier.model.config.label2id.items():
                if 'entail' in k.lower():
                    self.entail_idx = v
                    break
        if self.entail_idx is None:
            self.entail_idx = 0

    def get_entailment_prob(self, premise, hypothesis):
        """计算premise和hypothesis间entailment关系"""
        if not premise.strip() or not hypothesis.strip():
            return 0.0
        logits = self.nli_verifier.predict([(premise, hypothesis)])
        probs = scipy.special.softmax(logits, axis=1)
        print(probs[0][self.entail_idx])
        return probs[0][self.entail_idx]

    def get_critical_premise(self, premise, hypothesis, critical_threshold=0.5):
        "计算critical premise集合"
        base_score = self.get_entailment_prob(premise, hypothesis)
        base_entailment = base_score > critical_threshold
        p_sentences = [s.strip() for s in re.split(r'[.]', premise) if s.strip()]
        critical_P_sent = []
        none_cri_P_ids = []
        if len(p_sentences) > 1:
            for j, p_sent in enumerate(p_sentences):
                ablated_p = ". ".join(
                    [p_sentences[idx] for idx in range(len(p_sentences)) if (idx != j and idx not in none_cri_P_ids)])
                p_score1 = self.get_entailment_prob(ablated_p, hypothesis)
                if p_score1 > critical_threshold and base_entailment:
                    none_cri_P_ids.append(j)
                elif p_score1 <= critical_threshold and base_entailment:
                    critical_P_sent.append(p_sent)
        else:
            critical_P_sent = p_sentences
        # THE FIX: Prevent empty lists
        if not critical_P_sent:
            critical_P_sent = [premise]
        return critical_P_sent

    def get_critical_hypothesis(self, premise, hypothesis, critical_threshold=0.5):
        "计算critical hypothesis"
        base_score = self.get_entailment_prob(hypothesis, hypothesis)
        base_entailment = base_score > critical_threshold
        h_sentences = [s.strip() for s in re.split(r'[.]', hypothesis) if s.strip()]
        critical_H_sent = []
        none_cri_H_ids = []
        if len(h_sentences) > 1:
            for j, h_sent in enumerate(h_sentences):
                ablated_h = ". ".join(
                    [h_sentences[idx] for idx in range(len(h_sentences)) if (idx != j and idx not in none_cri_H_ids)])
                h_score = self.get_entailment_prob(ablated_h, hypothesis)
                if h_score <= critical_threshold and base_entailment:
                    critical_H_sent.append(h_sent)
                elif h_score > critical_threshold and base_entailment:
                    none_cri_H_ids.append(j)
        else:
            critical_H_sent = h_sentences
        # THE FIX: Prevent empty lists
        if not critical_H_sent:
            critical_H_sent = [hypothesis]
        return critical_H_sent

    def analyze_semantic_interaction(self, premise_list, hypothesis_list):
        """
        Performs 2D Semantic Ablation (Propositional Masking).
        Masks one Premise sentence and one Hypothesis sentence simultaneously to measure
        the drop in entailment, mapping the logical bridges.
        """
        num_p = len(premise_list)
        num_h = len(hypothesis_list)
        print(f"Detected the relationship between {num_p} Premise sentences and {num_h} Hypothesis sentences.")
        # 1. Calculate the baseline score (No ablation)
        base_score = self.get_entailment_prob(".".join([item for item in premise_list]),
                                              ".".join([item for item in hypothesis_list]))
        print(f"Baseline Entailment Score: {base_score:.4f}")
        # 2. Initialize the Interaction Matrix: (M+1) x (N+1)
        interaction_matrix = numpy.zeros((num_h + 1, num_p + 1))
        print(f"Running (N+1)*(M+1) Cross-Ablation ({(num_h + 1) * (num_p + 1)} passes)...")
        # --- Populate 1D Marginal Drops ---
        # Cell [0, 0]: Baseline vs Baseline (Drop is exactly 0)
        interaction_matrix[0, 0] = base_score
        # Row 0: Mask Premise ONLY (Hypothesis is fully intact)
        for j, p_sent in enumerate(premise_list):
            ablated_p = ".".join([s for idx, s in enumerate(premise_list) if idx != j])
            score = self.get_entailment_prob(ablated_p, ".".join([item for item in hypothesis_list]))
            interaction_matrix[0, j + 1] = score
        # Column 0: Mask Hypothesis ONLY (Premise is fully intact)
        for i, h_sent in enumerate(hypothesis_list):
            ablated_h = " ".join([s for idx, s in enumerate(hypothesis_list) if idx != i])
            score = self.get_entailment_prob(".".join([item for item in premise_list]), ablated_h)
            interaction_matrix[i + 1, 0] = score
        # --- Populate 2D Interaction Drops ---
        if num_p > 1 and num_h > 1:
            interaction_matrix = numpy.zeros((num_h + 1, num_p + 1))
            interaction_matrix[0, 0] = base_score
            # Row 0: Mask Premise ONLY (Hypothesis is fully intact)
            for j, p_sent in enumerate(premise_list):
                ablated_p = ".".join([s for idx, s in enumerate(premise_list) if idx != j])
                score = self.get_entailment_prob(ablated_p, ".".join([item for item in hypothesis_list]))
                interaction_matrix[0, j + 1] = score
            # Column 0: Mask Hypothesis ONLY (Premise is fully intact)
            for i, h_sent in enumerate(hypothesis_list):
                ablated_h = " ".join([s for idx, s in enumerate(hypothesis_list) if idx != i])
                score = self.get_entailment_prob(".".join([item for item in premise_list]), ablated_h)
                interaction_matrix[i + 1, 0] = score
            for ii, h_sent in enumerate(hypothesis_list):
                ablated_h = ".".join([s for idx, s in enumerate(hypothesis_list) if idx != ii])
                for jj, p_sent in enumerate(premise_list):
                    ablated_p = ".".join([s for idx, s in enumerate(premise_list) if idx != jj])
                    score = self.get_entailment_prob(ablated_p, ablated_h)
                    interaction_matrix[ii + 1, jj + 1] = score
            return interaction_matrix
        elif num_p == 1 and num_h > 1:
            interaction_matrix = numpy.zeros((num_h + 1, num_p))
            interaction_matrix[0, 0] = base_score
            for i, h_sent in enumerate(hypothesis_list):
                ablated_h = " ".join([s for idx, s in enumerate(hypothesis_list) if idx != i])
                score = self.get_entailment_prob(".".join([item for item in premise_list]), ablated_h)
                interaction_matrix[i + 1, 0] = score
            return interaction_matrix
        elif num_p > 1 and num_h == 1:
            interaction_matrix = numpy.zeros((num_h, num_p + 1))
            interaction_matrix[0, 0] = base_score
            for j, p_sent in enumerate(premise_list):
                ablated_p = ".".join([s for idx, s in enumerate(premise_list) if idx != j])
                score = self.get_entailment_prob(ablated_p, ".".join([item for item in hypothesis_list]))
                interaction_matrix[0, j + 1] = score
            return interaction_matrix
        elif num_p == 1 and num_h == 1:
            interaction_matrix = numpy.zeros((num_h, num_p))
            interaction_matrix[0, 0] = base_score
            return interaction_matrix

    def analysis_premise_hypothesis_relationship(self, premise_list, hypothesis_list, interaction_matrix,
                                                 critical_threshold=0.5):
        """
        Performs 2D Semantic Ablation (Propositional Masking).
        Masks one Premise sentence and one Hypothesis sentence simultaneously to measure
        the drop in entailment, mapping the logical bridges.
        """
        premise_hypothesis_pairs = []
        if interaction_matrix.shape[0] > 1 and interaction_matrix.shape[1] > 1:
            for i in range(interaction_matrix.shape[0] - 1):
                for j in range(interaction_matrix.shape[1] - 1):
                    if interaction_matrix[i + 1, j + 1] >= critical_threshold:
                        premise_hypothesis_pairs.append({"premise": premise_list[j], "hypothesis": hypothesis_list[i]})

        elif interaction_matrix.shape[0] == 1 and interaction_matrix.shape[1] > 1:
            for jj in range(interaction_matrix.shape[1] - 1):
                if interaction_matrix[0, jj + 1] >= critical_threshold:
                    premise_hypothesis_pairs.append({"premise": premise_list[jj], "hypothesis": hypothesis_list[0]})
            if len(premise_hypothesis_pairs) == 0:
                premise_hypothesis_pairs.append(
                    {"premise": ".".join([item for item in premise_list]), "hypothesis": hypothesis_list[0]})

        elif interaction_matrix.shape[0] > 1 and interaction_matrix.shape[1] == 1:
            for ii in range(interaction_matrix.shape[0] - 1):
                if interaction_matrix[ii + 1, 0] >= critical_threshold:
                    premise_hypothesis_pairs.append({"premise": premise_list[0], "hypothesis": hypothesis_list[ii]})
            if len(premise_hypothesis_pairs) == 0:
                premise_hypothesis_pairs.append(
                    {"premise": premise_list[0], "hypothesis": ".".join([item for item in hypothesis_list])})
        elif interaction_matrix.shape[0] == 1 and interaction_matrix.shape[1] == 1:
            premise_hypothesis_pairs.append({"premise": premise_list[0], "hypothesis": hypothesis_list[0]})

        return premise_hypothesis_pairs


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


if __name__ == "__main__":
    import pandas as pd

    test_user_file = "./mmlu_dataset/PG_Benchmark/user_profiles.json"
    with open(test_user_file) as user_file:
        user_data = json.load(user_file)
    model_names = ["cross-encoder/nli-deberta-v3-small", "cross-encoder/nli-MiniLM2-L6-H768", "cross-encoder/nli-deberta-v3-large", "cross-encoder/nli-distilroberta-base", "cross-encoder/nli-roberta-base"]
    overall_timing_records = []
    for name in model_names:
        file_suffix = name.split('/')[-1].lower()
        explainer = SemanticNLIExplainer(name)
        # Initialize the metric logger for the current model
        model_logger = MetricLogger()
        for i in range(10):
            test_data_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered.json"
            # test_data_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_{file_suffix}.json"
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
                query_subject = test_item["subject"]

                # Input predicates checking
                interest_premise = user_data[i]["description"]
                interest_hypothesis = f"The user is interested in {user_preference}."
                interest_predicate_score = timed_call(explainer.get_entailment_prob, interest_premise,
                                                      interest_hypothesis, method_name=name, logger=model_logger)

                query_premise = f"The query is {query_sentence}. The query subjet is {query_subject}."
                query_hypothesis = f"The user is interested in {user_preference}. The query subject,{user_preference}, is related to user interest area, {query_subject}."
                related_predicate_score = timed_call(explainer.get_entailment_prob, query_premise, query_hypothesis,
                                                     method_name=name, logger=model_logger)

                # Output predicates checking
                correct_hypothesis = f"The content of the answer is about {answer_sentence}."
                align_hypothesis = f"The content of the answer is about {user_preference}."

                general_responses_premise = f"The answer is {general_responses}."
                general_response_correct_score = timed_call(explainer.get_entailment_prob, general_responses_premise,
                                                            correct_hypothesis, method_name=name, logger=model_logger)
                general_response_align_score = timed_call(explainer.get_entailment_prob, general_responses_premise,
                                                          align_hypothesis, method_name=name, logger=model_logger)

                personal_response_premise = f"The answer is {personal_response}."
                personal_response_correct_score = timed_call(explainer.get_entailment_prob, personal_response_premise,
                                                             correct_hypothesis, method_name=name, logger=model_logger)
                personal_response_align_score = timed_call(explainer.get_entailment_prob, personal_response_premise,
                                                           align_hypothesis, method_name=name, logger=model_logger)

                sycophancy_response_premise = f"The answer is {sycophancy_response}."
                sycophancy_response_correct_score = timed_call(explainer.get_entailment_prob,
                                                               sycophancy_response_premise, correct_hypothesis,
                                                               method_name=name, logger=model_logger)
                sycophancy_response_align_score = timed_call(explainer.get_entailment_prob, sycophancy_response_premise,
                                                             align_hypothesis, method_name=name, logger=model_logger)

                failure_response_premise = f"The answer is {failure_response}."
                failure_response_correct_score = timed_call(explainer.get_entailment_prob, failure_response_premise,
                                                            correct_hypothesis, method_name=name, logger=model_logger)
                failure_response_align_score = timed_call(explainer.get_entailment_prob, failure_response_premise,
                                                          align_hypothesis, method_name=name, logger=model_logger)

                # Append results dictionary as before
                test_item["interest_scores"] = {"premise": interest_premise, "hypothesis": interest_hypothesis,
                                                "score": interest_predicate_score}
                test_item["related_scores"] = {"premise": query_premise, "hypothesis": query_hypothesis,
                                               "score": related_predicate_score}
                test_item["general_responses_correct_score"] = {"premise": general_responses_premise,
                                                                "hypothesis": correct_hypothesis,
                                                                "score": general_response_correct_score}
                test_item["general_responses_align_score"] = {"premise": general_responses_premise,
                                                              "hypothesis": align_hypothesis,
                                                              "score": general_response_align_score}
                test_item["personal_response_correct_score"] = {"premise": personal_response_premise,
                                                                "hypothesis": correct_hypothesis,
                                                                "score": personal_response_correct_score}
                test_item["personal_response_align_score"] = {"premise": personal_response_premise,
                                                              "hypothesis": align_hypothesis,
                                                              "score": personal_response_align_score}
                test_item["sycophancy_response_correct_score"] = {"premise": sycophancy_response_premise,
                                                                  "hypothesis": correct_hypothesis,
                                                                  "score": sycophancy_response_correct_score}
                test_item["sycophancy_response_align_score"] = {"premise": sycophancy_response_premise,
                                                                "hypothesis": align_hypothesis,
                                                                "score": sycophancy_response_align_score}
                test_item["failure_response_correct_score"] = {"premise": failure_response_premise,
                                                               "hypothesis": correct_hypothesis,
                                                               "score": failure_response_correct_score}
                test_item["failure_response_align_score"] = {"premise": failure_response_premise,
                                                             "hypothesis": align_hypothesis,
                                                             "score": failure_response_align_score}
                results.append(test_item)

            out_path = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered_{file_suffix}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            print(f"User {i} {file_suffix} check complete! Saved to: {out_path}")

        # Compute summary for this model and append to global metrics
        stats = model_logger.summary()
        overall_timing_records.append({ "model_name": name, "avg_time_per_check_sec": stats["avg_time"], "total_execution_time_sec": stats["total_time"], "total_checks_performed": len(model_logger.records)})

    # Save summary table as a CSV file
    summary_df = pd.DataFrame(overall_timing_records)
    summary_csv_path = "./mmlu_dataset/PG_Benchmark/nli_models_timing_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 40)
    print(f"Performance report saved to {summary_csv_path}")
    print(summary_df.to_string(index=False))
    print("=" * 40)



