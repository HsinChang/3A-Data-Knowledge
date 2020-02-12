import json
import hashlib

import block_reader
from transaction import Transaction


class MerkleTree(object):

    def __init__(self):
        self.root = None
        self.nodes = []  # Stores all other nodes except the leaf nodes
        self.leaves = []  # List of leaves of the tree. Leaves contain the transactions

    def add_node(self, node):
        # Add new node
        self.nodes.append(node)

    def add_leaf(self, leaf):
        # Add new leaf node
        self.leaves.append(leaf)

    def generate_nodes(self, list = None):
        temp = []
        if list is None:
            list = self.leaves
        # create the nodes of this layer
        if len(list) > 2:
            for index in range(0, len(list), 2):
                node = MerkleTreeNode()
                node.child_left = list[index].hash
                node.child_right = list[index+1].hash
                node.hash = list[index].parent
                temp.append(node)
        # make binary
            if len(temp) % 2 != 0:
                temp.append(temp[-1])
        # calculate the parents
            for index in range(0, len(temp), 2):
                temp[index].parent = compute_hash(temp[index].hash+temp[index+1].hash)
                temp[index + 1].parent = compute_hash(temp[index].hash + temp[index + 1].hash)
            for node in temp:
                self.nodes.append(node)
            self.generate_nodes(temp)
        # for the root node
        elif len(list)==2 :
            node = MerkleTreeNode()
            node.hash = list[0].parent
            node.child_left = list[0].hash
            node.child_right = list[1].hash
            self.root = node
            self.nodes.append(node)



# You can use the classes for node and leaf or you can create your own.
class MerkleTreeNode(object):

    def __init__(self):
        self.parent = None
        self.child_left = None
        self.child_right = None
        self.hash = None


class MerkleTreeLeaf(object):

    def __init__(self, transaction):
        self.parent = None
        self.transaction = transaction
        self.hash = compute_hash(transaction)


def create_merkle_tree(transactions):
    # Using the Merkle algorithm build the tree from a list of transactions in the block
    # transactions is list of Transaction
    merkle_tree = MerkleTree()
    for transaction in transactions:
        merkle_tree.add_leaf(MerkleTreeLeaf(block_reader.read_transaction(transaction)))
    # make sure that the tree can be binary
    if len(merkle_tree.leaves) % 2 != 0:
        merkle_tree.leaves.append(merkle_tree.leaves[-1])
    # compute the parents of the leaves
    for index in range(0, len(merkle_tree.leaves), 2):
        merkle_tree.leaves[index].parent = compute_hash(merkle_tree.leaves[index].hash + merkle_tree.leaves[index+1].hash)
        merkle_tree.leaves[index+1].parent = compute_hash(merkle_tree.leaves[index].hash + merkle_tree.leaves[index + 1].hash)
    merkle_tree.generate_nodes()
    return merkle_tree

def compute_hash(data):
    if isinstance(data, Transaction):
        to_hash = json.dumps(data.to_dict(), sort_keys=True).encode('utf-8')
    else:
        to_hash = data.encode('utf-8')
    a = type(to_hash)
    return hashlib.sha256(to_hash).hexdigest()