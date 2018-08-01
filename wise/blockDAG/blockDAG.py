#!/usr/bin/python
# encoding: utf-8
import hashlib
import json
from datetime import datetime, timezone
from merkleKDtree.merkle_kdtree import MerkleKDTree, square_distance
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from .geo_utils import to_Cartesian

class BlockDAGlocal:

    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.orphan_hashes = []
        self.new_orphan_hashes = []
        self.hash_map_block = {}
        # Create the genesis block
        self.create_block(proof=100)
        self.update_orphan_hashes()
        self.merkle_kd_trees = [None,]


    def create_block(self, proof):
        """
        Create a new Block in the Blockchain
        :param proof: <int> The proof given by the Proof of Work algorithm
        :param previous_hashуs: (Optional) <str> Hashуs of previous Blocks
        :return: <dict> New Block
        """

        if self.current_transactions:
            block = {
                'index': len(self.chain),
                'start_time' : int(min((x['ts'] for x in self.current_transactions))),
                'end_time' : int(max((x['ts'] for x in self.current_transactions))),
                'timestamp': int(max((x['ts'] for x in self.current_transactions))) + 100,
                'transactions': self.current_transactions,
                'orphan_hashes': self.orphan_hashes[:],
                'location_root' : None, # Merkle Patricia-trie
                'nonce' : proof,
                'trx_count' : len(self.current_transactions),
            }
            tree, m_root = self.form_merkle_kdtree(self.current_transactions)
            self.merkle_kd_trees.append(tree)
            block['merkle_space_root'] = m_root

        else:
        #     Genesis block
            creations_ts = int(datetime(2016, 12, 12).replace(tzinfo=timezone.utc).timestamp())
            block = {
                'index': len(self.chain),
                'start_time': 0,
                'end_time': 0,
                'timestamp': creations_ts,
                'transactions': self.current_transactions,
                'orphan_hashes': None,
                'location_root': None,  # Merkle Patricia-trie
                'merkle_space_root': None,  # Merkle KD-tree
                'trx_count' : 0,
                'nonce': proof
            }

        # Reset the current list of transactions
        self.current_transactions = []
        self.chain.append(block)
        hs = self.hash(block)
        self.hash_map_block[hs] = block
        self.new_orphan_hashes.append(hs)
        return hs

    def new_transaction(self, account, lng, lat, ts):
        """
        Creates a new transaction to go into the next mined Block
        :param sender: <str> Address of the Sender
        :param recipient: <str> Address of the Recipient
        :param amount: <int> Amount
        :return: <int> The index of the Block that will hold this transaction
        """
        self.current_transactions.append({
            'account': account,
            'lon' : lng,
            'lat' : lat,
            'ts' : int(ts)
        })

    def update_orphan_hashes(self):
        self.orphan_hashes = self.new_orphan_hashes
        self.new_orphan_hashes = []

    def optimize(self):
        self.merkle_kd_trees = np.array(self.merkle_kd_trees)
        self.chain = np.array(self.chain)

    def super_optimize(self):
        new_kd_trees = [None]
        for i, bh in enumerate(self.chain[1:], 1):
            tree = cKDTree([to_Cartesian(tr['lat'], tr['lon']) for tr in bh['transactions']])  # data[['lon', 'lat']
            new_kd_trees.append(tree)
        del self.merkle_kd_trees
        self.merkle_kd_trees = new_kd_trees
        self.optimize()


    def form_merkle_kdtree(self, current_transactions):
        data = pd.DataFrame(current_transactions)
        data_to_bulk = [(*to_Cartesian(row[0], row[1]), row[2]) for row in data[['lat', 'lon', 'ts']].values]
        tree = MerkleKDTree.construct_from_data(data_to_bulk)
        merkle_root = tree.compute_merkle_root()

        return tree, merkle_root


    @staticmethod
    def form_merkle_patriciatree(current_transactions):
        pass

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        # We must make sure that the Dictionary is Ordered, or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def __str__(self):

        orphan_hashes = self.orphan_hashes
        rows = []
        while orphan_hashes:
            row = '- ' + str((len(orphan_hashes)))
            rows.append(row)
            for hs in orphan_hashes:
                block_header = self.hash_map_block[hs]
                rows.append('\t{}  {}'.format(block_header['start_time'], block_header['end_time']))
            orphan_hashes = block_header['orphan_hashes']


        return '\n'.join(rows)




if __name__ == '__main__':
    block_dag = BlockDAGlocal()
