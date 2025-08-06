import json
import random
import numpy as np
from constants import RULE_NAMES, XI_VALUES, THRESHOLD_VALUES, N_MAX_RULES, N_MIN_RULES

def crossCombine(rules_1, rules_2):

	idx_rule_exchange_1 = np.random.randint(0, high=len(rules_1))
	idx_rule_exchange_2 = np.random.randint(0, high=len(rules_2))

	rule_components_names_1 = [rule_component["rule-name"] for rule_component in rules_1[idx_rule_exchange_1]]
	rule_components_types_1 = [rule_component["rule-type"] for rule_component in rules_1[idx_rule_exchange_1]]
	rule_components_names_2 = [rule_component["rule-name"] for rule_component in rules_2[idx_rule_exchange_2]]
	rule_components_types_2 = [rule_component["rule-type"] for rule_component in rules_2[idx_rule_exchange_2]]

	rule_components_types_1 = sorted(rule_components_names_1, key=lambda x: rule_components_names_1.index(x))
	rule_components_names_1 = sorted(rule_components_names_1)
	rule_components_types_2 = sorted(rule_components_names_2, key=lambda x: rule_components_names_2.index(x))
	rule_components_names_2 = sorted(rule_components_names_2)

	if rule_components_names_1 != rule_components_names_2:
		if rule_components_types_1 != rule_components_types_2:
			# cross-combine only if rules are different (not all same name and type)
			rule_exchange_1 = rules_1[idx_rule_exchange_1] 
			rule_exchange_2 = rules_2[idx_rule_exchange_2] 
			rules_1[idx_rule_exchange_1] = rule_exchange_2
			rules_2[idx_rule_exchange_2] = rule_exchange_1

	return rules_1, rules_2


# utils
def makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES):
	rule = {}
	rule["rule-name"] = RULE_NAMES[np.random.randint(0, high=len(RULE_NAMES))]
	rule["rule-type"] = XI_VALUES[np.random.randint(0, high=len(XI_VALUES))]
	rule["rule-threshold"] = THRESHOLD_VALUES[np.random.randint(0, high=len(THRESHOLD_VALUES))]
	return rule


N_MUTATION_TYPES = 6 # for computation of mutations

# mutation function
def MutationAddRandomRule(rules):
	# add a random rule to a random loop track
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	if len(rules[idx_rule_to_change]) < N_MAX_RULES:
		rules[idx_rule_to_change].append(makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES))
	return rules

def MutationRemoveRandomRule(rules):
	# remove a random rule from a random loop track
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if len(rules[idx_rule_to_change]) > N_MIN_RULES:
		rules[idx_rule_to_change].remove(rules[idx_rule_to_change][idx_rule_component_to_change])
	return rules

def MutationSubstituteRandomRule(rules):
	# substitute a rule with another random rule
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	rules[idx_rule_to_change][idx_rule_component_to_change] = makeRandomRule(RULE_NAMES, XI_VALUES, THRESHOLD_VALUES)
	return rules

def MutationIncreaseThreshold(rules):
	# increase the value of a threshold of a random rule element
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] += 0.1
	return rules

def MutationDecreaseThreshold(rules):
	# increase the value of a threshold of a random rule element
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] > 0 and rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] < 1:
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-threshold"] -= 0.1
	return rules

def MutationReverseXi(rules):
	# increase the value of a threshold of a random rule element
	idx_rule_to_change = np.random.randint(0, high=len(rules))
	idx_rule_component_to_change = np.random.randint(0, high=len(rules[idx_rule_to_change]))
	if rules[idx_rule_to_change][idx_rule_component_to_change]["rule-type"] == "less":
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-type"] = "more"
	else:
		rules[idx_rule_to_change][idx_rule_component_to_change]["rule-type"] = "less"
	return rules


def mutationsCheck(rules):
	for loop_track_rules in rules:
		if len(loop_track_rules) > 1:
			for rule in loop_track_rules:
				if rule["rule-type"] == "less" and rule["rule-threshold"] == 0:
					# remove rule
					loop_track_rules.remove(rule)
				if rule["rule-type"] == "more" and rule["rule-threshold"] == 1:
					# remove rule
					loop_track_rules.remove(rule)
		else:
			# only one rule in this loop track
			if loop_track_rules[0]["rule-type"] =="less" and loop_track_rules[0]["rule-threshold"] == 0:
				loop_track_rules[0]["rule-threshold"] = 0.5
			if loop_track_rules[0]["rule-type"] =="more" and loop_track_rules[0]["rule-threshold"] == 1:
				loop_track_rules[0]["rule-threshold"] = 0.5
	return rules


def RandomMutate(rules, n_mutations=1):
	# probability of selecting rules:...
	new_rules = rules.copy()
	possible_mutations = list(range(N_MUTATION_TYPES))
	mutations_weights = [40, 40, 40, 60, 60, 30]
	for _ in range(n_mutations):
		mutation_type = random.choices(possible_mutations, weights=mutations_weights, k=1)
		#mutation_type = np.random.randint(0, high=N_MUTATION_TYPES)
		if mutation_type[0] == 0:
			new_rules = MutationAddRandomRule(rules)
			new_rules = mutationsCheck(new_rules)
		elif mutation_type[0] == 1:
			new_rules = MutationRemoveRandomRule(rules)
			new_rules = mutationsCheck(new_rules)
		elif mutation_type[0] == 2:
			new_rules = MutationSubstituteRandomRule(rules)
			new_rules = mutationsCheck(new_rules)
		elif mutation_type[0] == 3:
			new_rules = MutationIncreaseThreshold(rules)
			new_rules = mutationsCheck(new_rules)
		elif mutation_type[0] == 4:
			new_rules = MutationDecreaseThreshold(rules)
			new_rules = mutationsCheck(new_rules)
		elif mutation_type[0] == 5:
			new_rules = MutationReverseXi(rules)
			new_rules = mutationsCheck(new_rules)
	return new_rules



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


