#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath('.'))

from aos.config_loader import load_configs, build_decision_payload_from_configs, build_state_payload_from_configs
from aos.prompts import build_decision_prompt_v2, build_state_prompt_v2


def main():
    cfgs = load_configs("config")
    assert "task_info" in cfgs and "operator_pools" in cfgs and "algo_config" in cfgs, "Missing configs"
    dec_payload = build_decision_payload_from_configs(cfgs, state_text="demo state", current_iteration=10)
    dec_msgs = build_decision_prompt_v2(dec_payload)
    print("[OK] decision prompt lines:", len(dec_msgs))

    st_payload = build_state_payload_from_configs(cfgs, current_generation=10)
    st_msgs = build_state_prompt_v2(st_payload)
    print("[OK] state prompt lines:", len(st_msgs))


if __name__ == "__main__":
    main()

