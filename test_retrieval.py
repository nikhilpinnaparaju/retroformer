# from tools import Retriever
# import json


# if __name__ == '__main__':
#     print("started")
#     retriever = Retriever(["this is a test sentence", "random third sentence", "new orleans is in montanao",])
#     print("retro built")
#     ret_val = "location of New Orleans"
    
#     print(retriever.retrieval(
#         ret_val, 2
#     ))

from orig_tools import Retriever

if __name__ == '__main__':
    print("started")
    retriever = Retriever()
    print("retro built")
    ret_val = "location of New Orleans"
    
    print(retriever.retrieval(
       ["new orleans is in montanao", "this is a test sentence", "random third sentence"], ret_val, 2
    ))