import json
import os
import random
import numpy
from datasets import load_dataset


def download_MMLU():
    """
    下载mmlu数据，返回test 和 topic列表
    :return:
    """
    print("Downloading MMLU")
    dataset = load_dataset("cais/mmlu", name="all", cache_dir="./mmlu_dataset/mmlu")
    MMLU_test = dataset['test']
    # Get the unique 57 topics
    unique_topics = numpy.unique(MMLU_test['subject']).tolist()
    clean_topics = [topic.replace("_", " ") for topic in unique_topics]
    return dataset, clean_topics


def optimize_test_distribution(user_interest, model_generalize, tau_min, tau_max):
    """
    Solves the optimization problem:
        min || t - w ||
        s.t. tau_min <= Divergence(t, p) <= tau_max

    Method: Analytical Derivation over the Linear Path.
    Since the objective and constraints are over the probability simplex,
    the optimal solution t_u* lies exactly on the linear interpolation
    path between p_u and w_u.
    """
    # 1. Calculate the Maximum Possible Divergence
    # This is the distance between the User Interest and the Target.
    # Div(w, p) = 0.5 * || w - p ||_1
    max_possible_div = 0.5 * numpy.linalg.norm(model_generalize - user_interest, ord=1)

    # Check Feasibility: If user and target are identical, return the user distribution.
    if max_possible_div < 1e-9:
        return user_interest

    # 2. Calculate Optimal Mixing Coefficient (alpha)
    # We define the path: t(alpha) = (1 - alpha)*p + alpha*w
    # Divergence scales linearly: Div(t(alpha), p) = alpha * max_possible_div
    alpha_upper = tau_max / max_possible_div
    alpha_lower = tau_min / max_possible_div

    # 3. Apply Constraints
    # We want alpha as close to 1.0 as possible to minimize distance to model_generalize
    optimal_alpha = 1.0

    # Enforce upper bound (Don't generalize too much)
    optimal_alpha = min(optimal_alpha, alpha_upper)

    # Enforce lower bound (Ensure minimum generalization)
    optimal_alpha = max(optimal_alpha, alpha_lower)

    # Final safety clip to ensure it's a valid probability mix [0, 1]
    optimal_alpha = numpy.clip(optimal_alpha, 0.0, 1.0)

    # 4. Compute the final distribution
    personal_general_test = (1 - optimal_alpha) * user_interest + optimal_alpha * model_generalize

    return personal_general_test


def generate_benchmark_test_distributions(input_json_path, output_json_path, all_topics=None, tau_min=0.1, tau_max=0.3):
    """
    Reads user profiles, calculates the optimized test distribution, and saves the updated data.
    """
    # Load the generated user profiles
    with open(input_json_path, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    # Setup the dimensional space (N)， Define model_generalize: The Company Requirement (Uniform Distribution over all categories)
    n_categories = len(all_topics)
    model_generalize = numpy.ones(n_categories) / n_categories
    print(f"Processing {len(profiles_data)} profiles over {n_categories} categories...")
    # Generate test distribution for each user
    for profile in profiles_data:
        user_interest_dict = profile['topic_distribution']
        user_interest_dis = numpy.zeros(n_categories)
        for i, category in enumerate(all_topics):
            user_interest_dis[i] = user_interest_dict.get(category, 0.0)
        # Run the analytical optimization
        user_test_dis = optimize_test_distribution(user_interest_dis, model_generalize, tau_min, tau_max)
        test_dist_dict = {}
        for i, category in enumerate(all_topics):
            if user_test_dis[i] > 1e-4:
                test_dist_dict[category] = round(float(user_test_dis[i]), 4)
        profile['test_distribution'] = test_dist_dict
        # coverage
        div = 0.5 * numpy.sum(numpy.abs(user_test_dis - user_interest_dis))
        profile["Divergence"]=div
        print(f"User {profile['user_id']}: Divergence achieved = {div:.4f}")
    # Save the updated profiles back to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(profiles_data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved updated profiles to {output_json_path}")


def sample_MMLU_test_items(dataset, input_json_path, output_json_path, test_size_per_user=100):
    """
    Samples physical test items from the MMLU dataset based on each user's test distribution.

    Args:
        dataset: The loaded HuggingFace MMLU dataset (the full dataset object).
        input_json_path: Path to the JSON containing user profiles and test distributions.
        output_json_path: Path to save the final user profiles with sampled questions.
        test_size_per_user: Total number of questions to sample for each user.
    """
    print(f"Loading user profiles from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)

    # 1. Group MMLU dataset by topic for extremely fast sampling (O(1) lookup)
    print("Indexing MMLU dataset by topic...")
    test_data = dataset['test']
    items_by_topic = {}

    for item in test_data:
        # Match your clean topic format (replace underscores with spaces)
        clean_topic = item['subject'].replace("_", " ")
        if clean_topic not in items_by_topic:
            items_by_topic[clean_topic] = []
        # We store the relevant fields to keep the JSON lightweight
        items_by_topic[clean_topic].append({"question": item["question"], "choices": item["choices"], "answer": item["answer"],  "subject": clean_topic })

    print("Sampling test sets for each user...")

    # 2. Process each user profile
    for profile in profiles_data:
        # Note: Your previous code saved it as 'test_distribution_cluster' even though it's topics.
        # Ensure we are using the right key based on your previous code.
        test_dist = profile.get('test_distribution', {})
        if not test_dist:
            continue
        topics = list(test_dist.keys())
        probs = numpy.array(list(test_dist.values()))
        # Convert probabilities to exact question counts using Multinomial Distribution
        topic_counts = numpy.random.multinomial(test_size_per_user, probs)
        user_test_data = []
        # Sample the actual questions
        for topic, count in zip(topics, topic_counts):
            if count == 0:
                continue
            available_items = items_by_topic.get(topic, [])
            available_count = len(available_items)
            # Edge Case: If the distribution asks for more questions than MMLU has for that topic
            if count > available_count:
                print( f"Warning: User {profile['user_id']} requested {count} items for '{topic}', but only {available_count} exist. Using all available.")
                sampled_items = available_items
            else:
                # Randomly sample without replacement
                sampled_items = random.sample(available_items, count)
            user_test_data.extend(sampled_items)
        # Shuffle the final test set so topics are mixed (realistic test scenario)
        random.shuffle(user_test_data)
        # Attach to the profile
        profile['test_data'] = user_test_data
        print(f"User {profile['user_id']}: Successfully sampled {len(user_test_data)} questions.")
    # Save the final dataset
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(profiles_data, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved final benchmark dataset to {output_json_path}")


def get_personalized_MMLU_test_items(dataset, output_json_path, sample_ratio=0.10):
    """
    Samples a global benchmark by taking exactly 10% of the questions from EACH topic
    in the MMLU dataset, then assigns the same mixed test set to every user.
    """
    test_data = dataset['test']
    # 1. Group MMLU dataset by topic
    print("Indexing MMLU dataset by topic...")
    items_by_topic = {}
    for item in test_data:
        clean_topic = item['subject'].replace("_", " ")
        if clean_topic not in items_by_topic:
            items_by_topic[clean_topic] = []
        items_by_topic[clean_topic].append({ "question": item["question"], "choices": item["choices"], "answer": item["answer"], "subject": clean_topic })

    print(f"Sampling {sample_ratio * 100:.0f}% of questions from each topic...")

    # 2. Sample exactly 10% from EACH topic individually (Stratified Sampling)
    global_benchmark = []
    for topic, items in items_by_topic.items():
        topic_total = len(items)
        # Calculate 10%. We use max(1, ...) to ensure that even if a topic
        # has very few questions, it still gets at least 1 question in the benchmark.
        sample_count = max(1, int(topic_total * sample_ratio))
        sampled_items = random.sample(items, sample_count)
        global_benchmark.extend(sampled_items)
    # 3. Shuffle the final benchmark so topics are mixed (realistic test scenario)
    random.shuffle(global_benchmark)
    print(f"Created a balanced global benchmark of {len(global_benchmark)} total questions.")
    print("Assigning the identical global benchmark to all users...")

    # Save the final dataset
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(global_benchmark, f, indent=4, ensure_ascii=False)

    print(f"Successfully saved standardized benchmark dataset to {output_json_path}")


if __name__ == "__main__":
    # 生成测试分布， 满足公司泛化性要求， 个人的个性化要求
    MMLU_dataset, MMLU_topic = download_MMLU()
    get_personalized_MMLU_test_items(MMLU_dataset, "./mmlu_dataset/PG_Benchmark/sampled_mmlu_test_data.json", sample_ratio=0.10)
    # generate_benchmark_test_distributions("mmlu_dataset/PG_Benchmark/user_profiles.json", "mmlu_dataset/PG_Benchmark/user_profiles_with_tests.json", MMLU_topic)
    # # 基于测试分布生成每个人的测试集
    # sample_MMLU_test_items(MMLU_dataset, "mmlu_dataset/PG_Benchmark/user_profiles_with_tests.json", "mmlu_dataset/PG_Benchmark/user_benchmark_dataset.json", 300)
