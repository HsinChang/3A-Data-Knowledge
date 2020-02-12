from skmultiflow.trees.hoeffding_tree import *
import random
from abc import ABCMeta, abstractmethod
from skmultiflow.trees.numeric_attribute_binary_test import NumericAttributeBinaryTest

SplitNode = HoeffdingTree.SplitNode
FoundNode = HoeffdingTree.FoundNode
Node = HoeffdingTree.Node
ActiveLearningNode = HoeffdingTree.ActiveLearningNode
InactiveLearningNode = HoeffdingTree.InactiveLearningNode
LearningNodeNBAdaptive = HoeffdingTree.LearningNodeNBAdaptive

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'



class HOT(HoeffdingTree):

    class HotNode(Node):
        """
            Abstract Class to create a New Node for HOT
        """
        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                      update_splitter_counters, found_nodes=None):
            if found_nodes is None:
                found_nodes = []
                self.filter_instance_to_leaves(X, y, weight, parent, parent_branch, update_splitter_counters, found_nodes)
                return found_nodes #np.asarray(FoundNode(found_nodes.__len__()))
            else:
                found_nodes.append(FoundNode(self,  parent, parent_branch))

        def get_class_votes(self):
            dist = self._observed_class_distribution
            dist_sum = sum(dist.values())
            if (dist_sum > 0.0):
                factor = 1.0/dist_sum
                dist = {key:value*factor for key, value in dist.iteritems()}
            return dist

    class HotSplitNode(SplitNode, HotNode):

        def __init__(self, split_test, class_observations):
            SplitNode.__init__(self, split_test, class_observations)
            self._option_count = -999
            self._parent = None
            self._next_option = None
            self._children = {}




        def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch, update_splitter_counters, found_nodes):
            if update_splitter_counters:
                self._observed_class_distribution[y] = weight
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    found_nodes = child.filter_instance_to_leaves(X, y, weight, parent, child_index,
                                                                  update_splitter_counters, found_nodes)
                    return found_nodes
                else:
                    found_nodes.append(HOT.FoundNode(self, parent, parent_branch))
            if self._next_option is not None:
                found_nodes = self._next_option.filter_instance_to_leaves(X, y, weight, parent, -999,
                                                                          update_splitter_counters,found_nodes)
                return found_nodes

        def describe_subtree(self, ht, buffer, indent=0):
            buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)

            for branch_idx in range(self.num_children()):
                child = self.get_child(branch_idx)
                if child is not None:
                    buffer[0] += textwrap.indent('if ', ' ' * indent)
                    buffer[0] += self._split_test.describe_condition_for_branch(branch_idx)
                    buffer[0] += ':\n'
                    buffer[0] += '** option count = ' + str(self._option_count)
                    child.describe_subtree(ht, buffer, indent + 2)

#jave line 403
        def compute_merit_of_existing_split(self, split_criterion, pre_dist):
            post_dist = {}
            i = 0
            for child in self._children:
                post_dist[i] = child.get_observed_class_distribution()
            return split_criterion.get_merit_of_split(pre_dist, post_dist)

        def update_option_count(self, source, hot):
            if self._option_count is -999:
                self._parent.update_option_count(source, hot)
            else:
                max_child_count = -999
                curr = self
                while (curr is not None):
                    for child in curr._children:
                        if isinstance(child, HOT.HotSplitNode): #may need to be HotSplitNode
                            if child._option_count > max_child_count:
                                max_child_count = child._option_count
                    if curr._next_option != None and isinstance(curr._next_option, HOT.HotSplitNode):
                        curr = curr._next_option
                    else:
                        curr = None
                if (max_child_count > self._option_count):
                    delta = max_child_count - self._option_count
                    self._option_count = max_child_count
                    if self._option_count >= hot._max_prediction_paths:
                        self.kill_option_leaf(hot)
                    curr = self
                    while (curr is not None):
                        for child in curr._children:
                            if isinstance(child, HOT.HotSplitNode):
                                split_child = child
                                if (split_child != source):
                                    split_child.update_optiopn_count_below(delta, hot)
                        if curr._next_option != None and isinstance(curr._next_option, HOT.HotSplitNode):
                            curr = curr._next_option
                        else:
                            curr = None
                    if self._parent is not None:
                        self._parent.update_option_count(self, hot)

        def update_option_count_below(self, delta, hot):
            if self._option_count is not -999:
                self._option_count += delta
                if (self._option_count>=hot._max_prediction_paths):
                    self.kill_option_leaf(hot)
            for child in self._children:
                if isinstance(child, HOT.HotSplitNode):
                    split_child = child
                    child.update_option_count_below(delta, hot)
            if isinstance(self._next_option, HOT.HotSplitNode):
                self._next_option.update_option_count_below(delta, hot)

        def kill_option_leaf(self, hot):
            if isinstance(self._next_option, HOT.HotSplitNode):
                self._next_option.kill_option_leaf(hot)
                #TODO change after
            elif isinstance(self._next_option, HOT.HotActiveLearningNode):
                self._next_option = None
                hot._active_leaf_node_cnt-=1
            elif isinstance(self._next_option, HOT.HotInactiveLearningNode):
                self._next_option = None
                hot._inactive_leaf_node_cnt-=1

        def get_head_option_count(self):
            sn = self
            while (sn._option_count==-999):
                sn = sn._parent
            return sn._option_count

    class HotActiveLearningNode(ActiveLearningNode):
        pass

    class HotInActiveLearningNode(InactiveLearningNode):
        pass

    def _partial_fit(self, X, y, sample_weight):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_nodes = self._tree_root.filter_instance_to_leaves(X, y, sample_weight, parent=None, parent_branch=-1,
                                                                update_splitter_counters=True, found_nodes=None)
        if found_nodes is None:
            leaf_node = self._new_learning_node()
            self._active_leaf_node_cnt += 1
            if isinstance(leaf_node, self.LearningNode):
                learning_node = leaf_node
                learning_node.learn_from_instance(X, y, sample_weight, self)
                if isinstance(learning_node, self.ActiveLearningNode):
                    active_learning_node = learning_node
                    weight_seen = active_learning_node.get_weight_seen()
                    weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                    if weight_diff >= self.grace_period:
                        self._attempt_to_split(active_learning_node, leaf_node.parent, leaf_node.parent_branch)
                        active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
        else:
            for found_node in found_nodes:
                leaf_node = found_node.node
                if leaf_node is None:
                    leaf_node = self._new_learning_node()
                    self._active_leaf_node_cnt += 1
                if isinstance(leaf_node, self.LearningNode):
                    learning_node = leaf_node
                    learning_node.learn_from_instance(X, y, sample_weight, self)
                    if isinstance(learning_node, self.ActiveLearningNode):
                        active_learning_node = learning_node
                        weight_seen = active_learning_node.get_weight_seen()
                        weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                        if weight_diff >= self.grace_period:
                            self._attempt_to_split(active_learning_node, found_node.parent, found_node.parent_branch)
                            active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)
        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()

    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_nodes = self._tree_root.filter_instance_to_leaves(X, -np.inf, -np.inf, parent=None, parent_branch=-1,
                                                                update_splitter_counters=False, found_nodes=None)
            prediction_paths = 0
            result = {}
            for found_node in found_nodes:
                if found_node.parent_branch != -999:
                    leaf_node = found_node.node
                    if leaf_node is None:
                        leaf_node = found_node.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.update(dist)
                    prediction_paths += 1
            if prediction_paths > self._max_prediction_paths:
                self._max_prediction_paths += 1
            return result
        else:
            return {}

    def _attempt_to_split(self, node: ActiveLearningNode, parent: SplitNode, parent_idx: int):
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            elif self._split_criterion == HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if parent_idx != -999:
                if len(best_split_suggestions) < 2:
                    should_split = len(best_split_suggestions) > 0
                else:
                    hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                        node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                    best_suggestion = best_split_suggestions[-1]
                    second_best_suggestion = best_split_suggestions[-2]
                    if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                            or hoeffding_bound < self.tie_threshold):  # best_suggestion.merit > 1e-10 and \
                        should_split = True
                    if self.remove_poor_atts is not None and self.remove_poor_atts:
                        poor_atts = set()
                        # Scan 1 - add any poor attribute to set
                        for i in range(len(best_split_suggestions)):
                            if best_split_suggestions[i] is not None:
                                split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                                if len(split_atts) == 1:
                                    if best_suggestion.merit - best_split_suggestions[i].merit > hoeffding_bound:
                                        poor_atts.add(int(split_atts[0]))
                        # Scan 2 - remove good ones from set
                        for i in range(len(best_split_suggestions)):
                            if best_split_suggestions[i] is not None:
                                split_atts = best_split_suggestions[i].split_test.get_atts_test_depends_on()
                                if len(split_atts) == 1:
                                    if best_suggestion.merit - best_split_suggestions[i].merit < hoeffding_bound:
                                        poor_atts.remove(int(split_atts[0]))
                        for poor_att in poor_atts:
                            node.disable_attribute(poor_att)
            elif len(best_split_suggestions) > 0:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(),
                                                               self.secondary_split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[len(best_split_suggestions) - 1]
                curr = parent
                best_previous_metric = float('-inf')
                pre_dist = node.get_observed_class_distribution()
                while True:
                    merit = curr.compute_merit_of_existing_split(self, split_criterion, pre_dist)
                    if merit > best_previous_metric:
                        best_previous_metric = merit
                    if curr._option_count != -999:
                        break
                    curr = curr._parent
                if best_suggestion.merit - best_previous_metric > hoeffding_bound:
                    should_split = True
            if should_split:
                split_decision = best_split_suggestions[len(best_split_suggestions) - 1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    if (parent_idx != 999):
                        self._deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self.new_split_node(split_decision.split_test,
                                                    node.get_observed_class_distribution())
                    new_split.parent = parent

                    option_head = parent
                    if parent is not None:
                        while option_head._option_count == -999:
                            option_head = option_head.parent
                    if parent_idx == -999 and parent is not None:
                        new_split._option_count = -999
                        option_head.update_option_count_below(1, self)
                        if option_head.parent is not None:
                            option_head.parent.update_option_count(option_head, self)
                        self.add_to_option_table(split_decision, option_head.parent)
                    else:
                        if option_head is None:
                            new_split._option_count = 1
                        else:
                            new_split._option_count = option_head._option_count
                    num_options = 1
                    if option_head is not None:
                        num_options = option_head._option_count
                    if num_options < self.max_option_paths:
                        new_split._next_option = node
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
                    else:
                        if parent_idx != -999:
                            parent.set_child(parent_idx, new_split)
                        else:
                            parent._next_option = new_split


    def add_to_option_table(self, best_suggestion: AttributeSplitSuggestion, parent: HotSplitNode):
        split_att = best_suggestion.split_test.get_atts_test_depends_on()
        split_val = -1.0
        if isinstance(best_suggestion.split_test, NumericAttributeBinaryTest):
            test = best_suggestion.split_test
            split_val = test.get_split_value()
        tree_depth = 0
        while parent is not None:
            parent = parent._parent
            tree_depth += 1

    def _deactivate_learning_node(self, to_deactivate: ActiveLearningNode, parent: SplitNode, parent_branch: int):
        new_leaf = self.InactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            if parent_branch != -999:
                parent.set_child(parent_branch, new_leaf)
            else:
                parent._next_option = new_leaf
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def _activate_learning_node(self, to_activate: InactiveLearningNode, parent: SplitNode, parent_branch: int):
        new_leaf = self._new_learning_node(to_activate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            if parent_branch != -999:
                parent.set_child(parent_branch, new_leaf)
            else:
                parent._next_option = new_leaf
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def __find_learning_nodes(self, node, parent, parent_branch, found):
        if node is not None:
            if isinstance(node, self.LearningNode):
                found.append(self.FoundNode(node, parent, parent_branch))
            if isinstance(node, self.HotSplitNode):
                split_node = node
                for i in range(split_node.num_children()):
                    self.__find_learning_nodes(split_node.get_child(i), split_node, i, found)
                self.__find_learning_nodes(split_node._next_option, split_node, -999, found)

    class HotLearningNode(LearningNodeNBAdaptive, HotNode):
        pass

    def _new_learning_node(self, initial_class_observations=None):
        return self.HotLearningNode(initial_class_observations)

    def new_split_node(self, split_test, class_observations):
        return self.HotSplitNode(split_test, class_observations)

    def __init__(self,
                 max_byte_size=33554432,
                 max_option_paths = 5,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 secondary_split_confidence = 0.1,
                 nb_threshold=0,
                 nominal_attributes=None):
        super(HOT, self).__init__(max_byte_size=max_byte_size,
                                  memory_estimate_period=memory_estimate_period,
                                  grace_period=grace_period,
                                  split_criterion=split_criterion,
                                  split_confidence=split_confidence,
                                  tie_threshold=tie_threshold,
                                  binary_split=binary_split,
                                  stop_mem_management=stop_mem_management,
                                  remove_poor_atts=remove_poor_atts,
                                  no_preprune=no_preprune,
                                  leaf_prediction=leaf_prediction,
                                  nb_threshold=nb_threshold,
                                  nominal_attributes=nominal_attributes)
        self.max_option_paths = max_option_paths
        self.secondary_split_confidence = secondary_split_confidence
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._train_weight_seen_by_model = 0.0
        self._max_prediction_paths = 0


        @property
        def max_option_paths(self):
            return self._max_option_paths

        @max_option_paths.setter
        def max_option_paths(self, max_option_paths):
            self._max_option_paths = max_option_paths

        @property
        def secondary_split_confidence(self):
            return self._secondary_split_confidence

        @secondary_split_confidence.setter
        def secondary_split_confidence(self, secondary_split_confidence):
            self._secondary_split_confidence = secondary_split_confidence