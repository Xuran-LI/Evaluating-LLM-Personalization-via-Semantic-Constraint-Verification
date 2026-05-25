import os
import re
import json


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def analysis_personalized_and_generalized_response(raw_response):
    raw_response = raw_response.split(' \nassistant\n')[-1]
    core_def_pattern = r'Core Definition[^\n]*\n(.*?)(?=\n[#\s\d\.\-\*]*The Bridge)'
    perspective_pattern = r'Perspective Expansion[^\n]*\n(.*?)(?=\n[#\s\d\.\-\*]*Tone & Format)'

    # Search and extract Core Definition
    core_def_match = re.search(core_def_pattern, raw_response, re.DOTALL)
    core_definition = core_def_match.group(1).strip() if core_def_match else "Section not found."

    # Search and extract Perspective Expansion
    perspective_match = re.search(perspective_pattern, raw_response, re.DOTALL)
    perspective_expansion = perspective_match.group(1).strip() if perspective_match else "Section not found."

    return core_definition, perspective_expansion


if __name__ == '__main__':
    for i in range(10):
        test_data_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results.json"
        with open(test_data_file) as test_file:
            test_data = json.load(test_file)
        result=[]
        for ii in range(len(test_data)):
            test_item = test_data[ii]
            prompts= test_item['prompts']
            llm_responses = test_item['llm_responses']
            general_responses, personalized_response = analysis_personalized_and_generalized_response(llm_responses[0])
            failure_response, sycophancy_response = analysis_personalized_and_generalized_response(llm_responses[1])
            test_item['general_response'] = general_responses
            test_item['personalized_response'] = personalized_response
            test_item['sycophancy_response'] = sycophancy_response
            test_item['failure_response'] = failure_response
            if general_responses!="Section not found." and personalized_response!="Section not found." and sycophancy_response!="Section not found." and failure_response!="Section not found.":
                result.append(test_item)

        output_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}_results_filtered.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"User {i} generation complete! Success filter response number {len(result)}. Saved successfully to: {output_file}")

#
# class UserInterestAnswerFilter:
#     def __init__(self):
#         print("Loading DeBERTa-v3 NLI Verifier...")
#         # Cross-encoders are mathematically rigorous for dual-sentence entailment
#         self.nli_verifier = CrossEncoder('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
#
#     def get_entailment_score(self, premise1, hypothesis1, premise2, hypothesis2):
#         """
#         Checks if A -> B1 AND A -> B2  A is the personalized response and B1 is the general answer and B2 is the user interest description.
#         """
#         # Prepare pairs: (Original, Candidate) and (Candidate, Original)
#         pairs = [(premise1, hypothesis1), (premise2, hypothesis2)]
#
#         # Predict
#         logits = self.nli_verifier.predict(pairs)
#         probs = softmax(logits, axis=1)
#
#         # Handle different config structures
#         if hasattr(self.nli_verifier, 'model'):
#             config = self.nli_verifier.Qwen_model.config
#         else:
#             config = self.nli_verifier.config
#
#         # Find Entailment ID (usually 0 or 1 depending on model)
#         entail_idx = config.label2id.get('entailment')
#         if entail_idx is None:
#             entail_idx = 0  # Default fallback if not found in label2id
#
#         NLI_prob1 = probs[0][entail_idx]
#         NLI_prob2 = probs[1][entail_idx]
#
#         return NLI_prob1, NLI_prob2
#
#
#     def filter(self, test_item, candidate_dict):
#         """
#         Runs the logical truth-condition gate over the generated candidate.
#         """
#         # 1. PREMISE: The generated personalized response
#         personalized_response = candidate_dict.get("personalized_response", "")
#         # 2. HYPOTHESIS 1 (Factual Correctness)
#         # Rebuild the standard answer into a clean factual statement without "0:", "1:", etc.
#         choices = test_item.get("choices", [])
#         answer_idx = test_item.get("answer")
#         if choices[answer_idx].lower() == "all of the above" or "all three" in choices[answer_idx].lower():
#             # Combine the actual facts cleanly
#             clean_facts = [c for i, c in enumerate(choices) if i != answer_idx]
#             general_response = " It is true that: " + "; and ".join(clean_facts) + "."
#         else:
#             general_response = f"It is true that: {choices[answer_idx]}."
#         # 3. HYPOTHESIS 2 (Thematic Relevance)
#         # Instead of the broad user_interest_analysis, use the specific candidate detail
#         # or build a specific declarative claim about the text.
#         domain = candidate_dict.get("interest_domain", "")
#         candidate_name = candidate_dict.get("candidate_name", "")
#         candidate_detail = candidate_dict.get("candidate_detail", "")
#         # Build a strong relevance hypothesis:
#         interest_hypothesis = f"The text utilizes the concept of {candidate_name} from {domain}. {candidate_detail}"
#         # 4. RUN NLI EVALUATION
#         score_query, score_interest = self.get_entailment_score(premise1=personalized_response, hypothesis1=general_response, premise2=personalized_response, hypothesis2=interest_hypothesis)
#         # Calculate harmonic mean (punishes the score heavily if ONE of them is very low)
#         # or just use standard average. Standard average is fine for now.
#         NLI_score = (score_query + score_interest) / 2.0
#         print(f"[NLI] | Factual (Q-Score): {score_query:.4f} | Relevance (I-Score): {score_interest:.4f} | Final: {NLI_score:.4f}")
#         # Save the detailed metrics back to the dictionary for your dataset
#         candidate_dict["factual_entailment_score"] = float(score_query)
#         candidate_dict["relevance_entailment_score"] = float(score_interest)
#         candidate_dict["final_nli_score"] = float(NLI_score)
#
#         return candidate_dict
#
# if __name__ == "__main__":
#     candidate_file = "/kaggle/input/LaMPDatasets/xuranli1993/personalized-mmlu/user_benchmark_dataset_answer_direction_generation_filter_personalization.json"
#     output_file = "/kaggle/working//user_benchmark_dataset_answer_direction_generation_filter_personalization_verification.json"
#
#     # Load existing progress if the script was stopped and restarted
#     if os.path.exists(candidate_file):
#         with open(candidate_file, 'r') as f:
#             answer_results_matrix = json.load(f)
#
#     # 5. The Execution Loop
#     print("\nInitiating Personalization Verification...")
#     filter_pipeline = UserInterestAnswerFilter()
#     # Using a cleaner iteration approach
#     for user_data in answer_results_matrix:
#         for test_items in user_data.get("test_data", []):
#             # FIX: Check for correct key "answer_candidates"
#             if "answer_candidates" in test_items:
#                 candidate_dicts = test_items["answer_candidates"]
#                 # FIX: Iterate through the dictionary values (which are lists of candidates)
#                 for pair_key, candidate_list in candidate_dicts.items():
#                     for idx, candidate in enumerate(candidate_list):
#                         if "personalized_response" not in candidate:
#                             continue
#                         # Process the candidate and update it in-place
#                         updated_candidate = filter_pipeline.filter(test_items,candidate)
#                         # Reassign the updated dictionary back to the list
#                         candidate_list[idx] = updated_candidate
#         # FIX: Save the entire matrix (answer_results_matrix), not just user_data
#         # Moving this to the outer loop to save frequently, but safely.
#         with open(output_file, 'w') as f:
#             json.dump(answer_results_matrix, f, indent=4)
#
#     print(f"\nNLI Filtering complete! Saved successfully to: {output_file}")
#
#
# # # pip install rank-llm[all]
# # import os
# # # Force PyTorch to manage memory segments dynamically to prevent fragmentation
# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # import json
# # import torch
# # import gc
# # from scipy.special import softmax
# #
# # # ==========================================
# # # THE DEPENDENCY FIX (MONKEY PATCH)
# # # ==========================================
# # import transformers.modeling_utils
# # import transformers.pytorch_utils
# #
# # missing_funcs = [
# #     'find_pruneable_heads_and_indices',
# #     'prune_linear_layer',
# #     'apply_chunking_to_forward',
# #     'Conv1D'
# # ]
# #
# # for func in missing_funcs:
# #     if hasattr(transformers.pytorch_utils, func) and not hasattr(transformers.modeling_utils, func):
# #         setattr(transformers.modeling_utils, func, getattr(transformers.pytorch_utils, func))
# #
# # # Now RankLLM will successfully import without crashing!
# # from rank_llm.data import Request, Candidate, Query
# # from rank_llm.rerank.pointwise.monot5 import MonoT5
# # # ==========================================
# #
# #
# # class CandidateReranker:
# #     def __init__(self):
# #         print("Initializing Official RankLLM Pointwise Reranker...")
# #         self.reranker = MonoT5(model="castorini/monot5-3b-msmarco-10k", batch_size=4, context_size=1024)
# #
# #     def rank_candidates(self, specific_query, valid_candidates):
# #         """
# #         valid_candidates is a list of dicts:
# #         [{'candidate_name': '...', 'candidate_detail': '...', 'candidate_score': 0.92, 'interest_weight': 0.32, 'interest_domain': '...'}]
# #         """
# #         if not valid_candidates:
# #             return []
# #         rankllm_candidates = []
# #         for candidate in valid_candidates:
# #             rankllm_candidates.append(Candidate(docid=candidate['candidate_name'], score=candidate['candidate_score'], doc={"text": candidate['candidate_detail']}))
# #         request = Request(query=Query(text=specific_query, qid="1"), candidates=rankllm_candidates)
# #         # 1. Execute Pointwise Reranking
# #         result = self.reranker.rerank_batch(requests=[request])[0]
# #         # ==========================================
# #         # MEMORY FLUSH: Clear the VRAM immediately
# #         # ==========================================
# #         torch.cuda.empty_cache()
# #         gc.collect()
# #         # 2. Extract Logits and Apply Softmax (The γ Variable)
# #         raw_qa_logits = [c.score for c in result.candidates]
# #         qa_probabilities = softmax(raw_qa_logits)
# #         ranked_results = []
# #         # 3. Calculate Final Multivariate Score
# #         for i, rank_candidate in enumerate(result.candidates):
# #             c_name = rank_candidate.docid
# #             c_intro = rank_candidate.doc["text"]  # Fixed: RankLLM maps this to "text"
# #             c_score = rank_candidate.score
# #             gamma_qa = float(qa_probabilities[i])
# #             # Retrieve original metrics for this specific candidate
# #             original_c = next(item for item in valid_candidates if item["candidate_name"] == c_name)
# #
# #             u_interest = original_c['user_interest']
# #             q_domain = original_c['query_domain']
# #
# #             ranked_results.append({"candidate_name": c_name, "candidate_detail": c_intro, "nli_score": c_score,
# #                                    "relevance_score": gamma_qa,"user_interest":u_interest,"query_domain":q_domain})
# #         # Sort by the final calculated score descending
# #         return ranked_results
# #
# #
# # if __name__ == "__main__":
# #
# #     # 1. Load Data
# #     users_file = "/kaggle/working/user_benchmark_dataset.json"
# #     with open(users_file, "r") as f:
# #         user_data = json.load(f)
# #
# #     candidates_file = "/kaggle/working/mmlu_topic_overlaps_NLI.json"
# #     with open(candidates_file, "r") as f:
# #         overlap_matrix = json.load(f)
# #
# #     reranker = CandidateReranker()
# #
# #     print(f"\nProcessing {len(user_data)} users...")
# #
# #     for user_profile in user_data:
# #         topic_distribution = user_profile["topic_distribution"]
# #         test_data = user_profile["test_data"]
# #
# #         for test_item in test_data:
# #             query_question = test_item["question"]
# #             query_subject = test_item["subject"]
# #             aggregated_candidates = []
# #             # Aggregate all valid candidates from all of the user's interests
# #             for user_interest, interest_weight in topic_distribution.items():
# #                 if query_subject == user_interest:
# #                     continue  # Skip exact matches (we want cross-disciplinary)
# #                 # Construct the dictionary key safely (alphabetical order)
# #                 pair_key = f"{min(query_subject, user_interest)} || {max(query_subject, user_interest)}"
# #                 if pair_key in overlap_matrix:
# #                     pair_data = overlap_matrix[pair_key]
# #                     # Handle the key depending on how it was saved in the previous step
# #                     bc = pair_data.get('bridging_concepts_NLI', {})
# #                     if isinstance(bc, dict) and "candidate_names" in bc:
# #                         for idx in range(len(bc["candidate_names"])):
# #                             aggregated_candidates.append({"candidate_name": bc["candidate_names"][idx],
# #                                                           "candidate_detail": bc["candidate_details"][idx],
# #                                                           "candidate_score": bc["candidate_score"][idx],
# #                                                           "user_interest": user_interest,
# #                                                           "query_domain": query_subject})
# #
# #             if "personalized_answer_directions" not in test_item:
# #                 test_item["personalized_answer_directions"] = []
# #
# #             # 2. Rank all pooled candidates against the specific question
# #             if aggregated_candidates:
# #                 ranked_directions = reranker.rank_candidates(query_question, aggregated_candidates)
# #
# #                 # 3. Connect the new results to the existing list
# #                 test_item["personalized_answer_directions"] += ranked_directions
# #
# #     # 2. Save the final data back to a JSON file
# #     output_file = "/kaggle/working/user_benchmark_dataset_routed.json"
# #     with open(output_file, "w") as f:
# #         json.dump(user_data, f, indent=4)
# #
# #     print(f"\nSuccessfully routed and saved dataset to: {output_file}")






