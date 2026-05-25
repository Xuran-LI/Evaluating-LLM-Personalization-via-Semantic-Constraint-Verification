# Evaluating LLM Personalization via Semantic Constraint Verification (NLICV)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges here, e.g., arXiv link, Python version, etc. -->

This repository provides the official implementation of **NLICV**, a framework designed to evaluate Large Language Model (LLM) personalization through rigorous semantic constraint verification. 

NLICV assesses personalized responses by verifying whether an LLM can simultaneously satisfy two core constraints:
1. **Factual Correctness:** Accurately addressing the user's query.
2. **Preference Alignment:** Aligning with user-specific profiles and instructions.

## 📊 Behavioral States Evaluation

Based on the fulfillment of these semantic constraints, our framework categorizes LLM responses into four distinct behavioral states. This taxonomy helps diagnose failure modes such as over-accommodation (sycophancy) or ignored instructions.

| Behavioral State | Addresses User Query (Correctness) | Aligns with Preferences (Personalization) |
| :--- | :---: | :---: |
| **Personalization** | ✅ | ✅ |
| **Generalization** | ✅ | ❌ |
| **Sycophancy** | ❌ | ✅ |
| **Failure** | ❌ | ❌ |

---

## 🚀 Pipeline Overview

The NLICV evaluation pipeline is structured into three distinct stages: Data Preparation, Response Generation, and Verification & Evaluation.

### Stage 1: User Profile Construction & Test Data Preparation
The framework initializes by generating synthetic user profiles tailored for multitask language understanding. We sample 10% of the test instances from the [MMLU benchmark](https://github.com/hendrycks/test) to serve as the foundational queries.

**Relevant Scripts:**
* `0.Get_User_Profiles.py`: Constructs diverse user profiles.
* `1.Get_Test_Data.py`: Samples the MMLU test instances.
* `2.Get_Answer_Direction.py`: Maps the benchmark instances to expected correct/incorrect answer directions.

### Stage 2: Response Generation
Using the **Qwen3-8B** model, the framework systematically generates responses representing the four behavioral states (Personalization, Generalization, Sycophancy, Failure). The generation process is strictly conditioned on the correct/incorrect options from MMLU and the specific user preferences defined in Stage 1.

**Relevant Scripts:**
* `3.Get_P_G_S_F_Prompts.py`: Constructs the conditional prompts for each behavioral state.
* `4.Get_P_G_S_F_Responses.py`: Executes the inference using Qwen3-8B.
* `5.Filter_P_G_S_F_Responses.py`: Cleans and filters the raw generated outputs.

### Stage 3: Response Verification & Evaluation
In the final stage, the framework evaluates the generated responses to identify their behavioral category. We benchmark the **NLICV** approach against standard **LLM-as-a-Judge** methods to demonstrate the effectiveness and reliability of semantic constraint verification.

**Relevant Scripts:**
* `6.Verify_P_G_S_F_Response.py`: Applies the verification constraints to classify responses.
* `7.Filter_P_G_S_F_Response.py`: Refines the verified evaluation results.
* `8.Get_P_G_S_F_Response_Result.py`: Aggregates the data and outputs the final comparative performance metrics.

---



