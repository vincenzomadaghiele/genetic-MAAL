import json
import os
import numpy as np


def makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES):
	rule = {}
	rule["rule-name"] = RULE_NAMES[np.random.randint(0, high=len(RULE_NAMES))]
	rule["rule-type"] = XI_VALUES[np.random.randint(0, high=len(XI_VALUES))]
	rule["rule-threshold"] = THRESHOLD_VALUES[np.random.randint(0, high=len(THRESHOLD_VALUES))]
	return rule


if __name__ == '__main__': 



	# INITIALIZE POSSIBLE SYSTEM SETTINGS
	print()
	RULE_NAMES = [
					"Harmonic similarity", "Harmonic movement - C", "Harmonic movement - D",
					"Melodic similarity", "Melodic trajectory - C", "Melodic trajectory - D",
					"Dynamic similarity", "Dynamic changes - C", "Dynamic changes - D",
					"Timbral similarity", "Timbral evolution - C", "Timbral evolution - D",
					"Global spectral overlap", "Frequency range overlap",
					"Rhythmic similarity", "Rhythmic density",
					"Harmonic function similarity", "Harmonic function transitions - C", "Harmonic function transitions - D"
					]
	XI_VALUES = ["more", "less"]
	step = 0.1
	THRESHOLD_VALUES = np.arange(0.0, 1.0+step, step).tolist()

	print("Rule names:")
	print(RULE_NAMES)
	print("Xi values:")
	print(XI_VALUES)
	print("Threshold values:")
	print(THRESHOLD_VALUES)



	# INITIALIZE BASIC CONFIG FILE
	configs_filepath = './genetic_algorithm/best_configs/'
	best_configs_paths = os.listdir(configs_filepath) 
	print(best_configs_paths)

	for config_filepath in best_configs_paths:
		# open basic JSON config file
		with open(f'{configs_filepath}/{config_filepath}', 'r') as file:
		    config_file = json.load(file)
		#print(config_file)

	rules = config_file["looping-rules"]
	print(rules)

	N_MAX_RULES = 5
	N_MIN_RULES = 1
	
	def MutationAddRandomRule(rules):
		# add a random rule to a random loop track
		idx_rule_to_change = np.random.randint(0, high=len(rules))
		if len(rules[idx_rule_to_change]) < N_MAX_RULES:
			rules[idx_rule_to_change].append(makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES))
		return rules

	new_rules = MutationAddRandomRule(rules)
	print(new_rules)

	def MutationRemoveRandomRule(rules):
		# remove a random rule from a random loop track
		idx_rule_to_change = np.random.randint(0, high=len(rules))
		if len(rules[idx_rule_to_change]) > N_MIN_RULES:
			idx_rule_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
			rules.remove(rules[idx_rule_to_change])
		return rules

	new_rules = MutationRemoveRandomRule(rules)
	print(new_rules)


	#def MutationSubstituteRandomRule(rules):
		# substitute a rule with another random rule

	def MutationIncreaseThreshold(rules):
		# increase the value of a threshold of a random rule element
		idx_rule_to_change = np.random.randint(0, high=len(rules))
		idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
		if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
			rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] += 0.1
		return rules

	new_rules = MutationIncreaseThreshold(rules)
	print(new_rules)


	def MutationDecreaseThreshold(rules):
		# increase the value of a threshold of a random rule element
		idx_rule_to_change = np.random.randint(0, high=len(rules))
		idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
		if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
			rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] -= 0.1
		return rules

	new_rules = MutationDecreaseThreshold(rules)
	print(new_rules)


	NUM_MUTATION_TYPES = 4
	def RandomMutate(rules, num_mutations=1):
		new_rules = rules.copy()
		for _ in range(num_mutations):
			mutation_type = np.random.randint(0, high=NUM_MUTATION_TYPES)
			if mutation_type == 0:
				new_rules = MutationAddRandomRule(rules)
				print(new_rules)
			elif mutation_type == 1:
				new_rules = MutationRemoveRandomRule(rules)
				print(new_rules)
			elif mutation_type == 2:
				new_rules = MutationIncreaseThreshold(rules)
				print(new_rules)
			elif mutation_type == 3:
				new_rules = MutationDecreaseThreshold(rules)
				print(new_rules)
		return new_rules

	new_rules = RandomMutate(rules)
	print(new_rules)



