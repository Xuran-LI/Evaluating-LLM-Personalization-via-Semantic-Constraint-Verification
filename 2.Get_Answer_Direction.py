import copy
import json
from rank_bm25 import BM25Okapi


def get_relevance_by_BM25(query, user_interests, k=1):
    """
    基于BM25获取与query相关的user data from profile
    :return:
    """
    corpus = list(user_interests.keys())
    # Tokenize the corpus (interest names) and the query (the question)
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query.lower().split()
    # Initialize BM25 and get the top N matching interest names
    bm25 = BM25Okapi(tokenized_corpus)
    top_n_items = bm25.get_top_n(tokenized_query, corpus, n=k)
    return top_n_items


if __name__ == "__main__":
    print("\nInitiating User Interest Answer Direction Generation...")

    users_file = "./mmlu_dataset/PG_Benchmark/user_profiles.json"
    with open(users_file, "r") as f:
        user_data = json.load(f)

    data_file = "./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data.json"
    with open(data_file, "r") as f:
        test_data = json.load(f)

    print(f"\nProcessing {len(user_data)} users...")
    for i in range(len(user_data)):
        user_profile = user_data[i]
        user_interests = user_profile["cluster_distribution"]
        # FIX 1: Create a deep copy of the test data so we don't overwrite
        # previous users' data in memory during the loop
        user_specific_test_data = copy.deepcopy(test_data)
        for test_item in user_specific_test_data:
            query_question = test_item["question"]
            # Get the top relevant interest for this specific user
            top_interests = get_relevance_by_BM25(query_question, user_interests, k=1)
            # Safely assign the top interest (or None if the user had no interests)
            test_item["user_related_interest"] = top_interests[0] if top_interests else None
        output_file = f"./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data_user_{i}.json"
        # FIX 2: Dump the user_specific_test_data, not the user_data profiles!
        with open(output_file, 'w') as f:
            json.dump(user_specific_test_data, f, indent=4)
        print(f"User {i} generation complete! Saved successfully to: {output_file}")



# class UserInterestAnswerDirectionGeneration:
#     def __init__(self, n_candidates=1):
#         print("Loading Qwen2.5-7B-Instruct Generator...")
#         self.n_candidates = n_candidates
#         # Using bfloat16 and auto-device mapping to fit the 7B model on modern GPUs
#         self.generator = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
#
#
#     def parse_llm_result(self, llm_output, query, user_interest):
#         """
#         Parses the HuggingFace pipeline output list to extract the query description, user interest, personalized answer direction
#         """
#         if isinstance(llm_output, list):
#             llm_output = llm_output[0]  # Grab the first generated sequence
#
#         if isinstance(llm_output, dict) and "generated_text" in llm_output:
#             text_content = llm_output["generated_text"]
#             # If the pipeline returns the full chat history, grab the last assistant message
#             if isinstance(text_content, list):
#                 llm_output_string = text_content[-1].get("content", "")
#             else:
#                 llm_output_string = text_content
#         else:
#             # Fallback just in case the format is highly unusual
#             llm_output_string = str(llm_output)
#
#             # 2. Now it is a standard string, so we can safely strip and clean it
#         cleaned_string = llm_output_string.strip()
#         # Remove ```json or ``` if the LLM accidentally added them
#         cleaned_string = re.sub(r"^```(?:json)?", "", cleaned_string, flags=re.IGNORECASE)
#         cleaned_string = re.sub(r"```$", "", cleaned_string).strip()
#         # 2. Safely parse the JSON
#         try:
#             response_dict = json.loads(cleaned_string)
#         except json.JSONDecodeError as e:
#             print(f"  [Error] LLM output was not valid JSON for interest '{user_interest}'. Skipping.")
#             return []  # Return an empty list to safely skip this interest
#         # 3. The Hallucination Gate: Did the LLM think a valid connection exists?
#         # (We cast to string to safely handle both 1 and "1")
#         is_possible = str(response_dict.get("is_possible", 0)).strip()
#         query_analysis=response_dict.get("query_analysis", [])
#         user_interest_analysis=response_dict.get("user_interest_analysis", [])
#         relationship_analysis=response_dict.get("relationship_analysis", [])
#         formatted_candidates = []
#         # 4. If a valid connection exists, extract and format the candidates
#         if is_possible == "1":
#             directions = response_dict.get("answer_directions", [])
#             if isinstance(directions, list):
#                 for direction in directions:
#                     # Ensure the LLM didn't skip generating the required keys
#                     if "candidate_name" in direction and "candidate_detail" in direction:
#                         formatted_candidates.append({
#                             "query": query,
#                             "query_analysis": query_analysis,
#                             "interest_domain": user_interest,
#                             "user_interest_analysis": user_interest_analysis,
#                             "relationship_analysis": relationship_analysis,
#                             "candidate_name": direction["candidate_name"],
#                             "candidate_detail": direction["candidate_detail"],
#                         })
#
#             print(f"  [Pass] Found {len(formatted_candidates)} valid direction(s) using '{user_interest}'.")
#         else:
#             print(f"  [Reject] LLM determined '{user_interest}' cannot logically answer this query.")
#
#         return formatted_candidates
#
#
#     def generate_candidates_answer_direction(self, query_question, query_domain, user_interest):
#         """Forces Qwen to generate N diverse candidate answer directions"""
#
#         prompt = f"""
#         You are a formal academic reasoning engine specializing in cross-disciplinary semantics.
#
#         Your objective is to identify TRUE conceptual intersections between a specific test Query (bounded by its Query Domain) and a specific User Interest.
#
#         These intersections must function as valid "answer directions"—meaning the specific Query can be factually, accurately, and completely answered using the analytical lens, vocabulary, or mechanisms of the User Interest.
#
#         <Query>
#         {query_question}
#         </Query>
#
#         <Query_Domain>
#         {query_domain}
#         </Query_Domain>
#
#         <User_Interest>
#         {user_interest}
#         </User_Interest>
#
#         ------------------------------------------------------------
#         Step 1 — Definition & Scope
#         ------------------------------------------------------------
#         1. Query Analysis: Identify the exact factual entities, intent, and core knowledge required to answer the specific Query.
#         2. User Interest Analysis: Define the User Interest strictly in terms of its object of study and primary analytical mechanisms.
#
#         ------------------------------------------------------------
#         Step 2 — Relationship Analysis (The Hallucination Gate)
#         ------------------------------------------------------------
#         Evaluate the direct conceptual relationship between the Query and the User Interest to determine if a factual, non-forced intersection is logically possible.
#         - If answering the Query using the User Interest requires hallucination, logical leaps, or results in a nonsensical connection, the relationship is INVALID.
#         - If the User Interest provides a highly relevant, factually grounded lens to explain the Query, the relationship is VALID.
#         - If VALID, generate up to 5 of the most suitable candidate answer directions.
#
#         ------------------------------------------------------------
#         Step 3 — Output Generation
#         ------------------------------------------------------------
#         Output your final response STRICTLY as a valid JSON object. Do not include any conversational filler, markdown formatting (such as ```json), or text outside of the JSON structure.
#
#         Important formatting rules:
#         - If "is_possible" is 0, leave the "answer_directions" list completely empty [].
#         - If "is_possible" is 1, populate "answer_directions" with up to 5 highly relevant direction objects.
#
#         {{
#             "query_analysis": "[1-2 sentences identifying the core factual entities required to answer the query]",
#             "user_interest_analysis": "[1-2 sentences defining the analytical boundaries of the user interest]",
#             "relationship_analysis": "[1-2 sentences explaining why a factual intersection is or is not logically possible]",
#             "is_possible": [Output strictly 1 if valid, or 0 if invalid/forced],
#             "answer_directions": [
#                 {{
#                     "candidate_name": "[Subconcept Name]",
#                     "candidate_detail": "[1-2 sentences explaining exactly how this concept bridges the User Interest to form a factual answer direction for the specific Query.]"
#                 }}
#             ]
#         }}
#         """
#         messages = [{"role": "system", "content": "You are a highly logical academic reasoning engine."},
#                     {"role": "user", "content": prompt}]
#         # Generate N distinct sequences using high temperature for diversity
#         outputs = self.generator(messages, max_new_tokens=512, max_length=None, temperature=0.8, top_p=0.9, do_sample=True, num_return_sequences=self.n_candidates)
#         candidate_results = self.parse_llm_result(outputs, query_question, user_interest)
#         return candidate_results
#
#
#     def run(self, query, query_domain, user_interest):
#         print(f"\n--- Processing Overlap ---")
#         print(f"Query: {query} | Domain: {query_domain} | Interest: {user_interest}")
#         candidate_results = self.generate_candidates_answer_direction(query, query_domain, user_interest)
#         print(candidate_results)
#         return candidate_results
