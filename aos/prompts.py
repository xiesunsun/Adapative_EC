from typing import Any, Dict, List, Optional
import base64
import os
import json


def encode_image_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def build_state_prompt(meta: Dict[str, Any], include_images: bool = False) -> List[Dict[str, Any]]:
    """
    Build messages to ask the model to summarize current GA state.
    meta should include keys: algo_config, task_info, operator_info, recent_metrics, image_paths.
    """
    sys = (
        "You are an expert in evolutionary algorithms analyzing GA for feature selection. "
        "Summarize the current algorithm state concisely and factually based on provided metadata."
    )
    user_lines = []
    user_lines.append("[Algorithm Config]\n" + str(meta.get("algo_config", {})))
    user_lines.append("[Task Info]\n" + str(meta.get("task_info", {})))
    user_lines.append("[Operator Info]\n" + str(meta.get("operator_info", {})))
    user_lines.append("[Recent Metrics]\n" + str(meta.get("recent_metrics", {})))
    content = "\n\n".join(user_lines)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": content}]
    if include_images:
        # Attach selected images as base64 blocks appended to user content
        img_paths = meta.get("image_paths", []) or []
        b64_list = []
        for p in img_paths:
            b64 = encode_image_b64(p)
            if b64:
                b64_list.append({"name": os.path.basename(p), "b64": b64})
        if b64_list:
            msgs.append({"role": "user", "content": "[Images as base64]"})
            for item in b64_list:
                msgs.append({"role": "user", "content": f"name={item['name']}: data:image/png;base64,{item['b64'][:12000]}"})
    return msgs


def build_decision_prompt(meta: Dict[str, Any], allowed_ops: Dict[str, Any], include_images: bool = False) -> List[Dict[str, Any]]:
    """
    Ask the model to choose operators and parameters; require JSON output.
    """
    sys = (
        "You are an assistant that outputs ONLY compact JSON for GA operator selection. "
        "Choose selection/crossover/mutation and their parameters, and cxpb/mutpb in [0,1]. "
        "Use only allowed operator names and clamps to safe ranges."
    )
    user_lines = []
    user_lines.append("[Allowed Operators]\n" + str(allowed_ops))
    user_lines.append("[Algorithm Config]\n" + str(meta.get("algo_config", {})))
    user_lines.append("[Task Info]\n" + str(meta.get("task_info", {})))
    user_lines.append("[Current Operators]\n" + str(meta.get("operator_info", {})))
    user_lines.append("[Recent Metrics]\n" + str(meta.get("recent_metrics", {})))
    user_lines.append(
        "Output strict JSON with keys: selection, crossover, mutation, params, cxpb, mutpb, rationale.\n"
        "Use DEAP parameter names exactly as listed in the allowed operators. "
        "cxpb/mutpb are floats in [0,1]."
    )
    content = "\n\n".join(user_lines)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": content}]
    if include_images:
        img_paths = meta.get("image_paths", []) or []
        b64_list = []
        for p in img_paths:
            b64 = encode_image_b64(p)
            if b64:
                b64_list.append({"name": os.path.basename(p), "b64": b64})
        if b64_list:
            msgs.append({"role": "user", "content": "[Images as base64]"})
            for item in b64_list:
                msgs.append({"role": "user", "content": f"name={item['name']}: data:image/png;base64,{item['b64'][:12000]}"})
    return msgs


def build_decision_prompt_v2(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build messages using the user's specified template for operator selection.
    Expected keys in payload:
      dataset_size, num_features, classification_model,
      current_iteration, total_iterations, adjust_interval,
      algorithm_configuration, state_evaluation_text, fitness_function,
      selection, crossover, mutation (operator pools as printable text/list)
    """
    sys = (
        "Role: You are a domain expert specializing in the configuration of evolutionary algorithms.\n"
        "Task: Your task is to devise an appropriate configuration for our evolutionary algorithm based on its current progress in a feature selection task."
    )
    u = []
    u.append("The following provides information about our feature selection task and the evolutionary algorithm being used.")
    u.append("")
    u.append("1. Feature Selection Task Information")
    u.append(f"Dataset Size: {payload.get('dataset_size')}")
    u.append(f"Number of Features: {payload.get('num_features')}")
    u.append(f"Downstream Classification Model: {payload.get('classification_model')}")
    u.append("")
    u.append("2. Evolutionary Algorithm Information")
    u.append("Framework: DEAP (using a simple Genetic Algorithm structure)")
    u.append(f"Current Iteration: {payload.get('current_iteration')}")
    u.append(f"Total Iterations: {payload.get('total_iterations')}")
    u.append(f"Adjustment Interval: {payload.get('adjust_interval')} (The algorithm's configuration is adjusted each time this interval is reached. You are performing this adjustment now.)")
    u.append(f"Previous Algorithm Configuration: {payload.get('algorithm_configuration')}")
    u.append(f"Algorithm State Evaluation Text: {payload.get('state_evaluation_text')}")
    u.append(f"Fitness Function: {payload.get('fitness_function')}")
    u.append("")
    u.append("3. Available Operator Pools")
    u.append(f"Selection Operator Pool: {payload.get('selection')}")
    u.append(f"Crossover Operator Pool: {payload.get('crossover')}")
    u.append(f"Mutation Operator Pool: {payload.get('mutation')}")
    u.append("")
    u.append("Your Instructions")
    u.append("Based on the information provided, please select a suitable combination of operators and determine appropriate probability values for cxpb and mutpb.")
    u.append("cxpb: The probability of mating two individuals.")
    u.append("mutpb: The probability of mutating an individual.")
    u.append(f"The algorithm is currently at iteration {payload.get('current_iteration')} out of a total of {payload.get('total_iterations')}.")
    u.append("")
    u.append("Please note:")
    u.append("The selected operators must come from the operator pools listed above.")
    u.append("Ensure that the operator names and their parameter settings are reasonable for the current stage of the evolutionary process.")
    u.append("Directly output the complete JSON configuration in the format specified below.")
    u.append("")
    u.append("Required Output Format (use only names and parameters from the pools above):")
    example = {
        "cxpb": 0.7,
        "mutpb": 0.3,
        "Selection": {"name": "tournament", "parameter": {"tournsize": 3}},
        "Crossover": {"name": "uniform", "parameter": {"indpb": 0.5}},
        "Mutation": {"name": "flip_bit", "parameter": {"indpb": 0.033}},
    }
    u.append(json.dumps(example))
    content = "\n".join(u)
    return [{"role": "system", "content": sys}, {"role": "user", "content": content}]


def build_state_prompt_v2(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build messages using the user's specified template for state text generation.
    Expected keys in payload include current_generation, total_generations, adjust_interval,
    algorithm_configuration, fitness_function, algorithm_state_pic (path or URL).
    """
    sys = (
        "Role: You are a domain expert proficient in the analysis of evolutionary algorithms.\n"
        "Task: Based on the provided information about an evolutionary algorithm, you will generate a State Text describing its current running state."
    )
    u = []
    u.append("The following information pertains to the current running state of the algorithm:")
    u.append("")
    u.append("1. **Evolutionary Algorithm Information**")
    u.append("   - **Framework:** DEAP (using a standard Genetic Algorithm structure)")
    cur_gen = payload.get('current_generation')
    tot_gen = payload.get('total_generations')
    u.append(f"   - **Current Generation:** {cur_gen} / **Total Generations:** {tot_gen}")
    u.append(f"   - **Adjustment Interval:** {payload.get('adjust_interval')} (The algorithm's configuration is adjusted each time this interval is reached. You are performing this adjustment now.)")
    u.append(f"   - **Previous Algorithm Configuration:** {payload.get('algorithm_configuration')}")
    u.append(f"   - **Fitness Function:** {payload.get('fitness_function')}")
    # cxpb/mutpb explanation and current values (if provided)
    cxpb = payload.get('cxpb')
    mutpb = payload.get('mutpb')
    u.append("   - **Algorithm Parameters:**")
    u.append("     - **cxpb**: The probability of mating (crossover) for each pair of selected individuals in the offspring pool.")
    u.append("     - **mutpb**: The probability of mutating each individual in the offspring pool.")
    u.append(f"     - **Current Values**: cxpb={cxpb}, mutpb={mutpb}")
    u.append("   - **Algorithm State Visualization:** An overview image that includes:")
    u.append("     - **Best Fitness Trend Line Chart:** Tracks the best fitness value over generations.")
    u.append("     - **Population Diversity Line Chart:** Shows changes in population diversity based on gene frequency. This metric calculates the frequency of each allele at each locus across the entire population. A higher value indicates greater allelic variation and thus higher diversity, suggesting the population is exploring a wider range of solutions. A lower value signifies that certain alleles are becoming dominant, leading to decreased diversity and potential premature convergence.")
    u.append("     - **Fitness Improvement Rate Per Generation Chart:** Illustrates the rate of fitness improvement from one generation to the next.")
    u.append("     - **Operator Success Rate and Application Count:** This is determined by tracking whether an individual's fitness improves after crossover or mutation. If the fitness is greater than the parent's, it is counted as a success (1), otherwise as a failure (0). The total number of successes is then divided by the total number of times the respective operator (crossover or mutation) was applied to calculate the success rate.")
    u.append(f"     - **Image:** {payload.get('algorithm_state_pic')}")
    u.append("")
    u.append("Your Instructions")
    u.append(f"Based on the information provided, please analyze the running state of the algorithm at generation {cur_gen} of {tot_gen} total generations.")
    u.append("Please directly output the corresponding state text in the following JSON format, without any additional information:")
    u.append("{" +
             "\n  \"fitness_trend_analysis\": \"...\"," +
             "\n  \"diversity_trend_analysis\": \"...\"," +
             "\n  \"improvement_rate_trend_analysis\": \"...\"," +
             "\n  \"operator_success_analysis\": \"...\"," +
             f"\n  \"current_algorithm_stage\": \"At the current generation {payload.get('current_generation')}...\"," +
             f"\n  \"bottleneck_analysis\": \"Analyzing the current state at generation {payload.get('current_generation')}, ...\"," +
             "\n  \"suggested_adjustments\": {\n    \"cxpb\": \"...\",\n    \"mutpb\": \"...\",\n    \"Selection\": {\n      \"adjust\": \"...\"\n    },\n    \"Crossover\": {\n      \"adjust\": \"...\"\n    },\n    \"Mutation\": {\n      \"adjust\": \"...\"\n    }\n  }\n}")
    content = "\n".join(u)
    return [{"role": "system", "content": sys}, {"role": "user", "content": content}]
