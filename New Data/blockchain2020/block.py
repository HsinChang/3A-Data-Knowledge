import json
import hashlib
from merkle_tree import create_merkle_tree


class Block(object):

    def __init__(self, header, transactions):
        # Store everything internally
        # header is a BlockHeader and transactions is a list of Transaction
        # call create_merkle_tree function to store transactions in Merkle tree
        self.N_STARTING_ZEROS = 4
        self.header = header
        self.transactions = create_merkle_tree(transactions)

    def to_dict(self):
        # Turns the object into a dictionary
        # There are two fields: header and transactions
        # The values are obtained by using the to_dict methods
        return self.__dict__

    def to_json(self):
        # Transforms into a json string
        # use the option sort_key=True to make the representation unique
        return json.dumps(self.to_dict(), sort_keys=True)

    def is_proof_ready(self):
        # Check whether the block is proven
        # For that, make sure the hash begins by N_STARTING_ZEROS
        block_hash = self.header.get_hash()
        return block_hash.startswith('0'*self.N_STARTING_ZEROS,0)

    def make_proof_ready(self):
        # Transforms the block into a proven block
        nonce = 0
        while not self.is_proof_ready():
            nonce += 1
            self.header.set_nonce(nonce)
