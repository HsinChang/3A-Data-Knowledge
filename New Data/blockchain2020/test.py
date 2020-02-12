import os
from block_reader import *

#Exercise 14
dir = 'blocks_to_prove'
for file in os.listdir(dir):
    with open(dir + '/' + file) as jsonp:
        print("For block: "+ file.title())
        block = read_block(json.load(jsonp))
        print("Merkle root: "+ block.transactions.root.hash)

#Excercise 21
# dir = 'blocks_to_prove'
# print("block : nonce, hash")
# for file in os.listdir(dir) :
#     with open(dir+"/" +file) as f:
#         block_dict=json.load(f)
#         block=read_block(block_dict)
#         block.make_proof_ready()
#         print(file[:-5]+" : "+str(block.header.nonce)+" , "+block.header.get_hash())

#Exercise 26
# dir = 'blockchain_wallets'
# for file in os.listdir(dir):
#     with open(dir + '/' + file, "r") as f:
#         block_chain = Blockchain()
#         for block in read_chain(f.read()):
#             block_chain.add_block(block)
#         print("For " + file + " :")
#         for (k, v) in block_chain.wallets.items():
#             print(k + ": " + str(v))

#Exercise 28
# dir = 'blockchain_incorrect'
# for file in os.listdir(dir):
#     with open(dir + '/' + file, "r") as f:
#         block_chain = Blockchain()
#         correct = True
#         for block in read_chain(f.read()):
#             if not block_chain.add_block(block):
#                 print(file[:-5] + ' incorrect, first error in block ' + str(block.header.index))
#                 correct = False
#                 break
#         if correct:
#             print(file[:-5] + ' is correct')



