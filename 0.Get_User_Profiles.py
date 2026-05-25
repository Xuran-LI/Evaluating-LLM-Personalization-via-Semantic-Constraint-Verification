import json
import os
import random
import numpy
import torch
from matplotlib import pyplot as plt
from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel


def download_MMLU():
    """
    Downloads MMLU data and returns a list of unique, clean topics.
    """
    print("Downloading MMLU...")
    dataset = load_dataset("cais/mmlu", name="all", cache_dir="./mmlu_dataset/mmlu")
    MMLU_test = dataset['test']

    # Get the unique 57 topics
    unique_topics = numpy.unique(MMLU_test['subject']).tolist()
    clean_topics = [topic.replace("_", " ") for topic in unique_topics]
    return clean_topics


def get_topic_embedding_vector(topics):
    """
    Gets the embedding vector representation of each topic in MMLU.
    """
    print("Loading model safely via Transformers...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # 1. Tokenize Sentences
    encoded_input = tokenizer(topics, padding=True, truncation=True, return_tensors='pt')
    # 2. Compute Token Embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # 3. Perform Mean Pooling
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embeddings.numpy()


def cluster_topics(topics, topics_embeddings, n_clusters=10):
    """
    Clusters topics based on semantic embeddings using Hierarchical Clustering.
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    labels = clustering.fit_predict(topics_embeddings)
    # Safe fallback just in case n_clusters is ever changed to > 10
    default_names = ["Social Sciences", "Natural Sciences", "Law and Ethics", "Mathematics", "CS and Engineering", "Global Affairs", "Human Health", "Economics", "History and Geography", "Medicine"]
    cluster_names = {i: default_names[i] if i < len(default_names) else f"Cluster {i}" for i in range(n_clusters)}
    cluster_map = {cluster_names[i]: [] for i in range(n_clusters)}
    topic_to_cluster = {}

    for topic, label in zip(topics, labels):
        c_name = cluster_names[label]
        cluster_map[c_name].append(topic)
        topic_to_cluster[topic] = c_name

    return cluster_map, topic_to_cluster


def visualize_user_radar_chart(user_distributions, save_dir="./mmlu_dataset/pic/"):
    """
    Draws a grid of dynamic Radar Charts for multiple users in one image.
    Categories with a value of 0 are dynamically hidden for each user.
    """
    num_users = len(user_distributions)
    if num_users == 0:
        print("No users to display.")
        return

    # Automatically calculate rows and columns for the grid (max 5 columns wide)
    cols = min(5, num_users)
    rows = (num_users + cols - 1) // cols

    # Create a large figure with a grid of polar plots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw=dict(polar=True))

    # Handle single-user edge case, otherwise flatten the axes array
    axes = [axes] if num_users == 1 else axes.flatten()

    for i, cluster_dist in enumerate(user_distributions):
        ax = axes[i]

        # 1. Dynamically filter out topics with a 0 value for THIS user
        active_interests = {k: v for k, v in cluster_dist.items() if v > 0}
        categories = list(active_interests.keys())
        values = list(active_interests.values())

        N = len(categories)

        # Safety check: if a user has no interests, leave the plot blank
        if N == 0:
            ax.axis('off')
            ax.set_title(f'User {i}\n(No Data)', size=12, y=1.1)
            continue

        # 2. Calculate dynamic angles just for this user's specific shape
        angles = [n / float(N) * 2 * numpy.pi for n in range(N)]

        # 3. Close the loop to connect the shape
        values += values[:1]
        angles += angles[:1]

        # Draw plot
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # Format axes
        ax.set_xticks(angles[:-1])

        # CRITICAL FIX: Use 'categories' (length 3 or 4) instead of 'default_names' (length 10)
        ax.set_xticklabels(categories, fontsize=7)

        y_max = max(0.5, max(values) + 0.1) if max(values) < 1.0 else 1.0
        ax.set_ylim(0, y_max)
        ax.set_yticks([y_max * 0.5])  # Just one gridline to keep it clean
        ax.set_yticklabels([""], color="grey", size=6)

        ax.set_title(f'User {i}', size=12, y=1.1)

    # If there are empty subplots (e.g., 8 users in a 10-slot grid), hide the empty ones
    for j in range(num_users, len(axes)):
        axes[j].axis('off')

    # Automatically adjust spacing between the charts so they don't overlap
    plt.tight_layout(pad=3.0)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"all_{num_users}_users_radar_grid.pdf")
    plt.savefig(save_path)
    plt.close()

    print(f"Dynamic grid saved to: {save_path}")


def generate_user_description(topic_distribution, cluster_distribution, topic_clusters):
    """
    Generates a detailed user description categorized by interest levels.
    """
    interests_1st = []  # p >= 0.5
    interests_2nd = []  # 0.25 <= p < 0.5 (Fixed typo from 2rd to 2nd)
    interests_other = []  # p < 0.25 (but > 0)

    sorted_clusters = sorted(cluster_distribution.items(), key=lambda x: x[1], reverse=True)

    for c_id, prob in sorted_clusters:
        if prob == 0: continue
        entry = (c_id, prob)
        if prob >= 0.5:
            interests_1st.append(entry)
        elif prob >= 0.25:
            interests_2nd.append(entry)
        else:
            interests_other.append(entry)

    def describe_user_interests(interest_clusters_list, cluster_description, topic_description):
        if not interest_clusters_list:
            return ""
        cluster_names = []
        topic_names = []
        for c_name, _ in interest_clusters_list:
            cluster_names.append(c_name)
            possible_topics = set(topic_clusters[c_name])
            user_topics_in_cluster = [(t_name, t_prob) for t_name, t_prob in topic_distribution.items() if t_name in possible_topics and t_prob > 0]
            user_topics_in_cluster.sort(key=lambda x: x[1], reverse=True)
            topic_names.append(", ".join([t[0] for t in user_topics_in_cluster]))
        interest_cluster_dis = cluster_description.format(", ".join(cluster_names))
        interest_topic_details = []
        num_placeholders = topic_description.count("{}")
        for i in range(len(cluster_names)):
            c_name = cluster_names[i]
            t_list = topic_names[i]
            if num_placeholders == 1:
                interest_topic_details.append(topic_description.format(t_list))
            else:
                interest_topic_details.append(topic_description.format(c_name, t_list))
        interest_topic_dis = ", ".join(interest_topic_details)
        return f"{interest_cluster_dis} {interest_topic_dis}."
    description_parts = []
    if interests_1st:
        description_parts.append(
            describe_user_interests(interests_1st, "The user's primary interest is {}, with a particular focus on", "{}"))
    if interests_2nd:
        description_parts.append(
            describe_user_interests(interests_2nd, "The user have some interest in {}, like", "{}"))
    if interests_other:
        description_parts.append(
            describe_user_interests(interests_other, "The user's minor interests include {}, such as", "{}"))

    return " ".join(description_parts)


def get_user_interest_distribution(user_clusters, cluster_map):
    """
    Generates distributions over explicitly selected clusters and their associated topics.

    Args:
        user_clusters (list): The 3-4 specific cluster names selected for this user.
        cluster_map (dict): The mapping of cluster names to their lists of topics
                            (MMLU_cluster_map).
    """
    # 1. Generate probabilities for the selected clusters using Dirichlet
    # We use a slightly higher alpha range (1.0 to 3.0) to avoid extreme zeros among the 3-4 chosen
    alphas = numpy.full(len(user_clusters), numpy.random.uniform(1.0, 3.0))
    phi_clusters = numpy.random.dirichlet(alphas)
    # Create the cluster distribution dictionary
    cluster_distribution = dict(zip(user_clusters, phi_clusters))
    # 2. Distribute those cluster probabilities down to the actual topics
    topic_distribution = {}
    for cluster_name in user_clusters:
        # Get the real topics that belong to this cluster
        topics_in_cluster = cluster_map[cluster_name]
        cluster_weight = cluster_distribution[cluster_name]
        # Use Dirichlet to split the cluster's weight among its specific topics
        if len(topics_in_cluster) > 0:
            phi_topics = numpy.random.dirichlet(numpy.full(len(topics_in_cluster), numpy.random.uniform(0.5, 2.0)))
            for topic, topic_weight in zip(topics_in_cluster, phi_topics):
                # Chain Rule: P(topic | user) = P(topic | cluster) * P(cluster | user)
                topic_distribution[topic] = topic_weight * cluster_weight
    # 3. Final normalization to ensure floating point math perfectly equals 1.0
    total_q = sum(topic_distribution.values())
    topic_distribution = {k: v / total_q for k, v in topic_distribution.items()}

    return topic_distribution, cluster_distribution


def generate_and_save_profiles(num_profiles=10, save_path="./mmlu_dataset/PG_Benchmark/user_profiles.json"):
    """
    Generates user profiles using a corrected exhaustive coverage strategy.
    """
    MMLU_topic = download_MMLU()
    MMLU_embedding = get_topic_embedding_vector(MMLU_topic)
    MMLU_cluster_map, MMLU_topic_to_cluster = cluster_topics(MMLU_topic, MMLU_embedding)
    cluster_name = ["Social Sciences", "Natural Sciences", "Law and Ethics", "Mathematics", "CS and Engineering", "Global Affairs", "Human Health", "Economics", "History and Geography", "Medicine"]
    profiles_data = []
    cluster_dists=[]
    print(f"Generating {num_profiles} profiles to cover {len(cluster_name)} topics...")
    for i in range(num_profiles):
        # 1. Select the directed/mandatory cluster for this user
        # Using modulo (%) ensures that if num_profiles > 10, it safely loops back to the start
        directed_cluster = cluster_name[i % len(cluster_name)]
        # 2. Determine total clusters for this user (between 3 and 4)
        num_total_clusters = random.randint(3, 4)
        # 3. Sample the remaining random clusters (2 or 3 of them)
        remaining_pool = [c for c in cluster_name if c != directed_cluster]
        sampled_clusters = random.sample(remaining_pool, num_total_clusters - 1)
        # 4. Combine the directed cluster with the random ones
        user_clusters = [directed_cluster] + sampled_clusters
        # 5. Generate the distributions
        # Note: We pass MMLU_cluster_map here so the function knows which topics belong to the clusters
        topic_dist, cluster_dist = get_user_interest_distribution(user_clusters, MMLU_cluster_map)
        cluster_dists.append(cluster_dist)
        user_desc = generate_user_description(topic_dist, cluster_dist, MMLU_cluster_map)
        profiles_data.append({"user_id": i, "description": user_desc,
                              "cluster_topic_map": MMLU_cluster_map,
                              "cluster_distribution": {k: v for k, v in cluster_dist.items() if v > 0},
                              "topic_distribution": {k: v for k, v in topic_dist.items() if v > 0}})
    # Final Check
    all_covered = set()
    for p in profiles_data:
        all_covered.update(p["topic_distribution"].keys())
    print(f"Verification: Total unique topics covered: {len(all_covered)}/{len(MMLU_topic)}")
    visualize_user_radar_chart(cluster_dists)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(profiles_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    generate_and_save_profiles()



