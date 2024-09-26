import copy

import numpy as np
import editdistance
import copy

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared import utils
from textattack.shared.validators import transformation_consists_of_word_swaps


class ImmuneAlgorithm(PopulationBasedSearch):
    def __init__(self, pop_size=40, max_iters=30):
        self._search_over = False
        self.pop_size = pop_size
        self.max_iters = max_iters

        self.p_m = 0.7
        self.p_c = 0.5
        self.p_v = 0.5
        self.n_c = 5

        self.deltas = 3
        self._lambda = 0.5

    def _perturb(self, current_result, original_result, neighbours, w_select_probs):
        current_text = current_result.attacked_text
        original_text = original_result.attacked_text
        original_words = np.array(original_text.words)
        current_words = np.array(current_text.words)

        x_len = len(w_select_probs)
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        # Indexes that have not been modified are preferentially selected for mutation
        while current_text.words[rand_idx] != original_text.words[rand_idx] and np.sum(
                original_words != current_words) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neighbours[rand_idx]
        return self._select_best_replacement(rand_idx, current_result, original_result, replace_list)

    def _select_best_replacement(self, rand_idx, current_result, original_result, replace_list):
        current_text = current_result.attacked_text
        word_to_replace = original_result.attacked_text.words[rand_idx]
        transformed_texts = []
        for r in replace_list:
            if r != word_to_replace:
                transformed_text = current_text.replace_word_at_index(rand_idx, r)
                transformed_texts.append(transformed_text)

        neighbour_results, self._search_over = self.get_goal_results(transformed_texts)
        if not neighbour_results:
            return current_result
        else:
            neighbor_scores = np.array([r.score for r in neighbour_results])
            score_diff = neighbor_scores - current_result.score
            if np.max(score_diff) <= 0:
                return current_result

            best_result = max(neighbour_results, key=lambda x: x.score)
            return best_result

    def _initialize_population(self, initial_result, pop_size, neighbours, w_select_probs):
        population = []
        for _ in range(pop_size):
            # Mutation step
            random_result = self._perturb(initial_result, initial_result, neighbours, w_select_probs)
            population.append(
                PopulationMember(random_result.attacked_text, random_result)
            )
        return population

    def _get_index_select_probs(self, initial_text, indices_to_order):
        len_text = len(initial_text.words)
        idx_select_probs = np.zeros(len_text)

        leave_one_texts = [
            initial_text.replace_word_at_index(i, '[UNK]')
            for i in indices_to_order
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        index_scores = np.array([result.score for result in leave_one_results])

        for i, idx in enumerate(indices_to_order):
            idx_select_probs[idx] = index_scores[i]

        return idx_select_probs

    def vaccination(self, source_text, target_text, pop_size):
        pop_members = []
        len_text = len(source_text.words)
        for _ in range(pop_size):
            indices_to_replace = []
            words_to_replace = []
            for i in range(len_text):
                if np.random.uniform() < self.p_v:
                    indices_to_replace.append(i)
                    words_to_replace.append(target_text.words[i])
            new_text = source_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            pop_members.append(PopulationMember(new_text))

        return pop_members

    def perform_search(self, initial_result):
        self._search_over = False
        len_text = len(initial_result.attacked_text.words)
        neighbours = [[] for _ in range(len_text)]
        indices_to_order = set()

        # Get candidates of each index
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            indices_to_order.add(diff_idx)
            neighbours[diff_idx].append(transformed_text.words[diff_idx])

        indices_to_order = sorted(list(indices_to_order))
        w_select_probs = self._get_index_select_probs(initial_result.attacked_text, indices_to_order)
        w_select_probs = normalize(w_select_probs)

        # Initilization
        population = self._initialize_population(initial_result, self.pop_size, neighbours, w_select_probs)

        for pop in population:
            pop.__setattr__('stimulation', pop.score)

        population = sorted(population, key=lambda x: x.stimulation, reverse=True)
        global_elite = max(population, key=lambda x: x.stimulation)
        assert global_elite == population[0]

        # single mutation check
        if (
                self._search_over
                or global_elite.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ):
            return global_elite.result

        num_clone = int(self.pop_size * self.p_c)
        for i in range(self.max_iters):
            # clone selection
            population = population[:num_clone]
            for j in range(num_clone):
                orig_pop = population[j]
                cloned_pops = [orig_pop for _ in range(self.n_c)]
                # mutation
                for k in range(1, self.n_c):
                    if np.random.uniform() < self.p_m:
                        clone_result = self._perturb(cloned_pops[k], initial_result, neighbours, w_select_probs)
                        cloned_pops[k] = PopulationMember(clone_result.attacked_text, clone_result)

                clone_affinity = np.array([pop.score for pop in cloned_pops])
                clone_density = self.get_density(cloned_pops)
                clone_stimulation = self.get_stimulation(clone_affinity, clone_density)
                clone_index = np.argsort(clone_stimulation)[::-1]
                # mutation selection
                population[j] = cloned_pops[clone_index[0]]

            # vaccination
            reset_pops = self.vaccination(initial_result.attacked_text, global_elite.attacked_text,
                                          self.pop_size - num_clone)
            population.extend(reset_pops)
            
            # Query for the results of adversarial examples
            pop_results, self._search_over = self.get_goal_results(
                [pop.attacked_text for pop in population]
            )
            # maximum queries check
            if self._search_over:
                population = population[: len(pop_results)]
                if not population:
                    return initial_result

            for k in range(len(pop_results)):
                population[k].result = pop_results[k]
            
            # Update the best example (vaccination)
            top_member = max(population, key=lambda x: x.score)
            if (
                    self._search_over
                    or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            if top_member.score > global_elite.score:
                global_elite = copy.copy(top_member)
            
            # Compute the quality of examples and sort
            affinity = np.array([pop.score for pop in population])
            density = self.get_density(population)
            stimulation = self.get_stimulation(affinity, density)

            for k in range(len(population)):
                population[k].__setattr__('stimulation', stimulation[k])

            population = sorted(population, key=lambda x: x.stimulation, reverse=True)

        return global_elite.result

    def get_density(self, pop_members):
        density = np.zeros(len(pop_members))
        for i in range(len(pop_members)):
            for j in range(len(pop_members)):
                edit_distance = editdistance.eval(pop_members[i].attacked_text.words,
                                                  pop_members[j].attacked_text.words)
                if edit_distance > self.deltas:
                    density[i] += 0
                else:
                    density[i] += 1

            density[i] /= len(pop_members)

        return density

    def get_stimulation(self, affinity, density):
        return affinity - self._lambda * density

    def check_transformation_compatibility(self, transformation):
        """The iummune algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["pop_size", "max_iters"]


def normalize(n):
    n = np.array(n)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    else:
        return n / s
