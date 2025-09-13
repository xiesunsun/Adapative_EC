#!/usr/bin/env python3
import json
from aos.prompts import build_decision_prompt_v2, build_state_prompt_v2
from aos.schema import validate_decision


def main():
    decision_payload = {
        "dataset_size": "n=569 (breast_cancer)",
        "num_features": 30,
        "classification_model": "logistic",
        "current_iteration": 20,
        "total_iterations": 50,
        "adjust_interval": 10,
        "algorithm_configuration": {
            "selection": "tournament3",
            "crossover": "two_point",
            "mutation": "flip_bit",
            "cxpb": 0.6,
            "mutpb": 0.2,
        },
        "state_evaluation_text": "IR is flattening; diversity trending down; SR low.",
        "fitness_function": "CV_mean - alpha * (k/d)",
        "selection": ["tournament", "sus", "best"],
        "crossover": ["one_point", "two_point", "uniform"],
        "mutation": ["bit_flip", "uniform_int"],
    }
    msgs_decision = build_decision_prompt_v2(decision_payload)
    print("Decision prompt built. system+user messages:", len(msgs_decision))
    print("Decision user content preview:\n", msgs_decision[1]["content"][:300], "...\n")

    state_payload = {
        "current_generation": 20,
        "total_generations": 50,
        "adjust_interval": 10,
        "algorithm_configuration": {
            "selection": "tournament3",
            "crossover": "two_point",
            "mutation": "flip_bit",
            "cxpb": 0.6,
            "mutpb": 0.2,
        },
        "fitness_function": "CV_mean - alpha * (k/d)",
        "algorithm_state_pic": "ga_results/overview.png",
    }
    msgs_state = build_state_prompt_v2(state_payload)
    print("State prompt built. system+user messages:", len(msgs_state))
    print("State user content preview:\n", msgs_state[1]["content"][:300], "...\n")

    # Test decision validation with the user's JSON format (v2)
    llm_output = {
        "cxpb": 0.7,
        "mutpb": 0.3,
        "Selection": {"name": "tournament", "parameter": {"k": 3}},
        "Crossover": {"name": "one_point", "parameter": {}},
        "Mutation": {"name": "bit_flip", "parameter": {"prob": 0.05}},
        "rationale": "increase exploration with higher mutation"
    }
    norm, warn = validate_decision(llm_output)
    print("Normalized decision (v2):")
    print(json.dumps(norm, indent=2))
    print("Warnings:", warn)

    # Test a slightly invalid case
    bad_dec = {
        "cxpb": 1.2,
        "mutpb": -0.1,
        "Selection": {"name": "roulette", "parameter": {"k": 1}},
        "Crossover": {"name": "unknown"},
        "Mutation": {"name": "shuffle_indexes"},
    }
    norm2, warn2 = validate_decision(bad_dec)
    print("Normalized (invalid) decision v2:")
    print(json.dumps(norm2, indent=2))
    print("Warnings:", warn2)


if __name__ == "__main__":
    main()
