import json
import numpy as np


def crossCombine(rules_1, rules_2):
	
	idx_rule_exchange_1 = np.random.randint(0, high=len(rules_1))
	idx_rule_exchange_2 = np.random.randint(0, high=len(rules_2))
	rule_exchange_1 = rules_1[idx_rule_exchange_1] 
	rule_exchange_2 = rules_2[idx_rule_exchange_2] 

	rules_1[idx_rule_exchange_1] = rule_exchange_2
	rules_2[idx_rule_exchange_2] = rule_exchange_1

	return rules_1, rules_2


if __name__ == '__main__': 

	# load comparable log
	logifle_path = f'genetic_algorithm/best_configs/config_0.json'
	with open(logifle_path, 'r') as file:
		generated_config = json.load(file)
	# load comparable log
	logifle_path = f'genetic_algorithm/corpus/objective_config.json'
	with open(logifle_path, 'r') as file:
		objective_config = json.load(file)	

	print(generated_config["looping-rules"])
	print(objective_config["looping-rules"])

	rules_1 = generated_config["looping-rules"]
	rules_2 = objective_config["looping-rules"]

	idx_rule_exchange_1 = np.random.randint(0, high=len(rules_1))
	idx_rule_exchange_2 = np.random.randint(0, high=len(rules_2))
	rule_exchange_1 = rules_1[idx_rule_exchange_1] 
	rule_exchange_2 = rules_2[idx_rule_exchange_2] 

	rules_1[idx_rule_exchange_1] = rule_exchange_2
	rules_2[idx_rule_exchange_2] = rule_exchange_1

	print()
	print(rules_1)
	print(rules_2)
