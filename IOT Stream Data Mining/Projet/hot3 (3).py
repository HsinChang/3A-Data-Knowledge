#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:11:32 2020

@author: cherilyn
"""

import copy
import textwrap
from abc import ABCMeta
from operator import attrgetter, itemgetter

import numpy as np

from skmultiflow.utils.utils import get_dimensions, normalize_values_in_dict, calculate_object_size
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_class_observer_null import AttributeClassObserverNull
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.numeric_attribute_binary_test import NumericAttributeBinaryTest
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.instance_conditional_test import InstanceConditionalTest

MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'


class HoeffdingOptTree(BaseSKMObject, ClassifierMixin):
    class FoundNode(object):
        def __init__(self, node=None, parent=None, parent_branch=None):
            self.node = node
            self.parent = parent
            self.parent_branch = parent_branch

    class Node(metaclass=ABCMeta):
        def __init__(self, class_observations=None):
            """ Node class constructor. """
            if class_observations is None:
                class_observations = {}  # Dictionary (class_value, weight)
            self._observed_class_distribution = class_observations

        def is_leaf(self):
            return True

        def filter_instance_to_leaves(self, X, parent, parent_branch, update_splitter_counts):
            nodes = []
            self._filter_instance_to_leaves(X, parent, parent_branch, nodes, update_splitter_counts)
            return nodes

        def _filter_instance_to_leaves(self, X, parent, parent_branch, nodes, update_splitter_counts):
            nodes.append(HoeffdingOptTree.FoundNode(self, parent, parent_branch))

        def get_observed_class_distribution(self):
            return self._observed_class_distribution

        def set_observed_class_distribution(self, observed_class_distribution):
            self._observed_class_distribution = observed_class_distribution

        def get_class_votes(self, X, hot):
            return self._observed_class_distribution

        def subtree_depth(self):
            return 0  # 0 if leaf

        def calculate_promise(self):
            total_seen = sum(self._observed_class_distribution.values())
            if total_seen > 0:
                return total_seen - max(self._observed_class_distribution.values())
            else:
                return 0

        def describe_subtree(self, hot, buffer, indent=0):
            buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)
            try:
                class_val = max(self._observed_class_distribution, key=self._observed_class_distribution.get)
                buffer[0] += 'Class {} | {}\n'.format(class_val, self._observed_class_distribution)
            except ValueError:  # Regression problems
                buffer[0] += 'Statistics {}\n'.format(self._observed_class_distribution)

    class SplitNode(Node):
        def __init__(self, split_test, class_observations, next_option=None):
            """ SplitNode class constructor."""
            super().__init__(class_observations)
            self._split_test = split_test
            # Dict of tuples (branch, child)
            self._children = {}
            self.next_option = next_option
            self.option_count = 0

        def num_children(self):
            return len(self._children)

        def get_split_test(self):
            return self._split_test

        def set_child(self, index, node):
            if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
                raise IndexError
            self._children[index] = node

        def get_child(self, index):
            if index in self._children:
                return self._children[index]
            else:
                return None

        def instance_child_index(self, X):
            return self._split_test.branch_for_instance(X)

        @staticmethod
        def is_leaf():
            return False

        def __filter_instance_to_leaves(self, X, y, weight, parent, parent_branch, nodes, update_splitter_counts):
            if (update_splitter_counts):
                try:
                    self._observed_class_distribution[y] += weight
                except KeyError:
                    self._observed_class_distribution[y] = weight
                    self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))
            print(dict(sorted(self._observed_class_distribution.items())))
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    return child._filter_instance_to_leaves(X, self, child_index, nodes, update_splitter_counts)
                    return nodes.append(HoeffdingOptTree.FoundNode(None, self, child_index))
            if self.next_option is not None:
                self.next_option._filter_instance_to_leaves(X, self, -999, nodes, update_splitter_counts)

        def describe_subtree(self, hot, buffer, indent=0):
            for branch_idx in range(self.num_children()):
                child = self.get_child(branch_idx)
                if child is not None:
                    buffer[0] += textwrap.indent('if ', ' ' * indent)
                    buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                    buffer[0] += ':\n'
                    child.describe_subtree(hot, buffer, indent + 2)

        def subtree_depth(self):
            max_child_depth = 0
            for child in self._children:
                if child is not None:
                    print(type(child))
                    depth = child.subtree_depth()
                    if depth > max_child_depth:
                        max_child_depth = depth
            return max_child_depth + 1

        def compute_merit_of_existing_split(self, split_criterion, pre_split_dist):
            post_split_dist = []
            for i in range(len(self._children)):
                post_split_dist[i] = self._children[i].get_observed_class_distribution()
            return split_criterion.get_merit_of_split(pre_split_dist, post_split_dist)

        def update_option_count(self, source, hot):
            if self.option_count == -999:
                self.parent.update_option_count(source, hot)
            else:
                max_child_count = -999
                curr = self
                while curr is not None:
                    for child in curr._children:
                        if isinstance(child, self.SplitNode):
                            split_child = self.SplitNode(child)
                            if (split_child.option_count > max_child_count):
                                max_child_count = split_child.option_count
                    if curr.next_option is not None and isinstance(curr.next_option, self.SplitNode):
                        curr = self.SplitNode(curr.next_option)
                    else:
                        curr = None
                if max_child_count > self.option_count:
                    delta = max_child_count - self.option_count
                    self.option_count = max_child_count
                    if self.option_count >= 5:
                        self.kill_option_leaf(hot)
                    curr = self
                    while curr is not None:
                        for child in curr._children:
                            if isinstance(child, self.SplitNode):
                                split_child = self.SplitNode(child)
                                if split_child is not source:
                                    split_child.update_option_count_below(delta, hot)
                        if curr.next_option is not None and isinstance(curr.next_option, self.SplitNode):
                            curr = self.SplitNode(curr.next_option)
                        else:
                            curr = None
                    if self.parent is not None:
                        self.parent.update_option_count(self, hot)

        def update_option_count_below(self, delta, hot):
            if self.option_count != -999:
                self.option_count += delta
                if self.option_count >= 5:
                    self.kill_option_leaf(hot)
            for child in self._children:
                split_child = self.SplitNode(child)
                split_child.update_option_count_below(delta, hot)
            if isinstance(self.next_option, self.SplitNode):
                self.splitNode(self.next_option).update_option_count_below(delta, hot)

        def kill_option_leaf(self, hot):
            if isinstance(self.next_option, self.SplitNode):
                self.splitNode(self.next_option).kill_option_leaf(hot)
            elif isinstance(self.next_option, self.ActiveLearningNode):
                self.next_option = None
                hot.active_leaf_node_count -= 1
            elif isinstance(self.next_option, self.InactiveLearningNode):
                self.next_option = None
                hot.inactive_leaf_node_count -= 1

    class LearningNode(Node):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, hot):
            pass

    class InactiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, hot):
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
                self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

    class ActiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
            self._attribute_observers = {}

        def learn_from_instance(self, X, y, weight, hot):
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
                self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if hot.nominal_attributes is not None and i in hot.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = NumericAttributeClassObserverGaussian()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            return sum(self._observed_class_distribution.values())

        def get_weight_seen_at_last_split_evaluation(self):
            return self._weight_seen_at_last_split_evaluation

        def set_weight_seen_at_last_split_evaluation(self, weight):
            self._weight_seen_at_last_split_evaluation = weight

        def get_best_split_suggestions(self, criterion, hot):
            best_suggestions = []
            pre_split_dist = self._observed_class_distribution
            null_split = AttributeSplitSuggestion(None, [{}],
                                                  criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
            best_suggestions.append(null_split)
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, hot.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
            return best_suggestions

        def disable_attribute(self, att_idx):
            if att_idx in self._attribute_observers:
                self._attribute_observers[att_idx] = AttributeClassObserverNull()

    class LearningNodeNB(ActiveLearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, hot):
            if self.get_weight_seen() >= 0:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, hot)

        def disable_attribute(self, att_index):
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, hot):
            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, hot)

        def get_class_votes(self, X, hot):
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    def __init__(self,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 secondary_split_confidence=0.1,
                 tie_threshold=0.05,
                 binary_split=False,
                 remove_poor_atts=False,
                 leaf_prediction='nba',
                 nominal_attributes=None,
                 ):
        """ HoeffdingOptionTree class constructor."""
        super().__init__()
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.secondary_split_confidence = secondary_split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.remove_poor_atts = remove_poor_atts
        self.leaf_prediction = leaf_prediction
        self.nominal_attributes = nominal_attributes

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._max_prediction_path = 0
        self._train_weight_seen_by_model = 0.0
        self.classes = None

    @property
    def grace_period(self):
        return self._grace_period

    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period

    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        #if split_criterion != GINI_SPLIT and split_criterion != INFO_GAIN_SPLIT and split_criterion != HELLINGER:
        #    print("Invalid split_criterion option {}', will use default '{}'".format(split_criterion, INFO_GAIN_SPLIT))
        #    self._split_criterion = INFO_GAIN_SPLIT
       # else:
            #self._split_criterion = split_criterion
        self._split_criterion = split_criterion

    @property
    def split_confidence(self):
        return self._split_confidence

    @split_confidence.setter
    def split_confidence(self, split_confidence):
        self._split_confidence = split_confidence

    @property
    def secondary_split_confidence(self):
        return self._secondary_split_confidence

    @secondary_split_confidence.setter
    def secondary_split_confidence(self, secondary_split_confidence):
        self._secondary_split_confidence = secondary_split_confidence

    @property
    def binary_split(self):
        return self._binary_split

    @binary_split.setter
    def binary_split(self, binary_split):
        self._binary_split = binary_split

    @property
    def remove_poor_atts(self):
        return self._remove_poor_atts

    @remove_poor_atts.setter
    def remove_poor_atts(self, remove_poor_atts):
        self._remove_poor_atts = remove_poor_atts

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction != MAJORITY_CLASS and leaf_prediction != NAIVE_BAYES \
                and leaf_prediction != NAIVE_BAYES_ADAPTIVE:
            print("Invalid leaf_prediction option {}', will use default '{}'".format(leaf_prediction,
                                                                                     NAIVE_BAYES_ADAPTIVE))
            self._leaf_prediction = NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, nominal_attributes):
        self._nominal_attributes = nominal_attributes

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        #print("start fit")
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt,
                                                                                                  len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self._partial_fit(X[i], y[i], sample_weight[i])

        return self

    def _partial_fit(self, X, y, sample_weight):
        #print("start fit2")
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_node = self._tree_root.filter_instance_to_leaves(X, None, -1, True)
        for fn in found_node:
            leaf_node = fn.node
            #print("leaf_node:", type(leaf_node))
            #print(leaf_node)
            if leaf_node is None:
                leaf_node = self._new_learning_node()
                fn.parent.set_child(fn.parent_branch, leaf_node)
                self._active_leaf_node_cnt += 1
                print(self._active_leaf_node_cnt)
            if isinstance(leaf_node, self.LearningNode):
                learning_node = leaf_node
                learning_node.learn_from_instance(X, y, sample_weight, self)
                if isinstance(learning_node, self.ActiveLearningNode):
                    active_learning_node = learning_node
                    weight_seen = active_learning_node.get_weight_seen()
                    weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                    if weight_diff >= self.grace_period:
                        print("JFNVAEKDNVKAEBDMNVSALEFMRSLAFME###")
                        self._attempt_to_split(active_learning_node, fn.parent, fn.parent_branch)
                        active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)

    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_nodes = self._tree_root.filter_instance_to_leaves(X, None, -1, 0)
            print("this is the len",len(found_nodes))
            result = []
            for found_node in found_nodes:
                if found_node.parent_branch != -999:
                    leaf_node = found_node.node;
                    if leaf_node is None:
                        leaf_node = found_node.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.append(dist)
            return result
        else:
            return {}

    def predict(self, X):
        #print(self.get_model_measurements()["Tree size (nodes)"])
        #print("start predict")
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)


    def predict_proba(self, X):
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            print("this is i", i)
            votes = self.get_votes_for_instance(X[i])
            print(X[i])
            print(votes)
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                new_votes = dict((key,d[key]) for d in votes for key in d)
                if sum(new_votes.values()) != 0:
                    normalize_values_in_dict(new_votes)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in new_votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        print(predictions)
        return np.array(predictions)



    def measure_tree_depth(self):
        if isinstance(self._tree_root, self.Node):
            return self._tree_root.subtree_depth()
        return 0

    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.ActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations)

    def get_model_description(self):
        if self._tree_root is not None:
            buffer = ['']
            description = ''
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description

    def compute_hoeffding_bound(self, range_val, confidence, n):
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def new_split_node(self, split_test, class_observations):
        """ Create a new split node."""
        return self.SplitNode(split_test, class_observations)

    def _attempt_to_split(self, node: ActiveLearningNode, parent: SplitNode, parent_idx: int):
        split_criterion = InfoGainSplitCriterion()
        best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort(key=attrgetter('merit'))
        should_split = False
        if parent_idx != -999:
            print("enter")
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
                print("step1")
            else:
                print("step2")
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):  # best_suggestion.merit > 1e-10 and \
                    should_split = True
                    print("set should split to true")
                if self.remove_poor_atts is not None and self.remove_poor_atts:
                    poor_atts = set()
                    print("kjnvsaernverjnvfs")
                    # Scan 1 - add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit > hoeffding_bound:
                                    poor_atts.add(int(split_atts[0]))
                    # Scan 2 - remove good attributes from set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            print("WORNG TYPE",type(best_split_suggestions[i].split_test))
                            split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if best_suggestion.merit - best_split_suggestions[i].merit < hoeffding_bound:
                                    poor_atts.remove(int(split_atts[0]))
                    for poor_att in poor_atts:
                      node.disable_attribute(poor_att)

        elif len(best_split_suggestions) > 0:
            print("else statement")
            hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                node.get_observed_class_distribution()), self.secondary_split_confidence, node.get_weight_seen())
            best_suggestion = best_split_suggestions[-1]
            current = parent
            best_previous_merit = float("-inf")
            pre_dist = node.get_observed_class_distribution()
            while (True):
                merit = current.compute_merit_of_existing_split(split_criterion, pre_dist);
                if merit > best_previous_merit:
                    best_previous_merit = merit
                if current.option_count != -999:
                    break
                current = current.parent
            if best_suggestion.merit - best_previous_merit > hoeffding_bound:
                should_split = True

        if should_split:
            print("should")
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                if parent_idx != -999:
                    # Preprune - null wins
                    self._deactivate_learning_node(node, parent, parent_idx)
            else:
                new_split = self.new_split_node(split_decision.split_test,
                                                node.get_observed_class_distribution())
                new_split.parent = parent

                # Add option procedure
                option_head = parent
                if parent is not None:
                    while option_head.option_count == -999:
                        option_head = option_head.parent
                if parent_idx == -999 and parent is not None:
                    # adding a new option
                    new_split.option_count = -999
                    option_head.update_option_count_below(1, self)
                    if option_head.parent is not None:
                        option_head.parent.update_option_count(option_head, self)
                        self.add_to_option_table(split_decision, option_head.parent)
                else:
                    # adding a regular leaf
                    if option_head is None:
                        new_split.option_count = 1
                    else:
                        new_split.option_count = option_head.option_count
                num_option = 1
                if option_head is not None:
                    num_option = option_head.option_count
                if num_option < 5:
                    new_split.next_option = node
                    split_atts = split_decision.split_test.get_atts_test_depends_on()
                    for i in split_atts:
                        node.disable_attribute(i)
                else:
                    self._active_leaf_node_cnt -= 1
                for i in range(split_decision.num_splits()):
                    new_child = self._new_learning_node(split_decision.resulting_class_distribution_from_split(i))
                    new_split.set_child(i, new_child)
                self._decision_node_cnt += 1

                self._active_leaf_node_cnt += split_decision.num_splits()
                if parent is None:
                    self._tree_root = new_split
                    print("PARENT")
                    print(type(self._tree_root))
                else:
                    if parent_idx != -999:
                        parent.set_child(parent_idx, new_split)
                        print("boom")
                    else:
                        parent.next_option = new_split
                        print("groom")
        #return self._active_leaf_node_cnt

    def add_to_option_table(self, best_suggestion: AttributeSplitSuggestion, parent: SplitNode):
        split_atts = best_suggestion.split_test.get_atts_test_depends_on()[0]
        split_val = -1.0
        if isinstance(best_suggestion.split_test, NumericAttributeBinaryTest):
            test = NumericAttributeBinaryTest(best_suggestion.split_test)
            split_val = test.get_split_value()
        tree_depth = 0
        while parent is not None:
            parent = parent.parent
            tree_depth += 1
        print(self._train_weight_seen_by_model + ","
              + tree_depth + "," + split_atts + "," + split_val)

    def deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for i in range(len(learning_nodes)):
            if isinstance(learning_nodes[i], self.ActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)

    def _deactivate_learning_node(self, to_deactivate: ActiveLearningNode, parent: SplitNode, parent_branch: int):
        new_leaf = self.InactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            if parent_branch != -999:
                parent.set_child(parent_branch, new_leaf)
            else:
                parent.next_option = new_leaf
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

dataset = "elec"

stream = FileStream(dataset+".csv", n_targets=1, target_idx=-1)
# 2. Prepare for use
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier
h = [
        HoeffdingOptTree()
     ]

evaluator = EvaluatePrequential(pretrain_size=1000, max_samples=20000, show_plot=True,
                                metrics=['accuracy', 'kappa'], output_file='result_'+dataset+'.csv',
                                batch_size=1)
# 4. Run
evaluator.evaluate(stream=stream, model=h)