import copy
import requests
import calendar
import json
import torch
import wolframalpha
import openai
import datetime
import time
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
)
from typing import List
from operator import truediv, mul, add, sub

# Optional imports
from googleapiclient.discovery import build


"""
retrieval

Uses Carptriever to retrieve sentences before the current context.

input_sentences - List[String], sentences to retrieve from
input_text - String, the input text (e.g. The dog's name is)
k - The number of sentences to retrieve

output - A list of strings, each string is the retrieved sentence, and the sentence after.
"""


class Retriever:
    def __init__(self, input_sentences: List[str]):
        self.model = AutoModel.from_pretrained(
            "CarperAI/carptriever-1", add_pooling_layer=False
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/carptriever-1")
        self.index = None
        self.index_sentences = input_sentences

        self.build_index(input_sentences)
        print("index built")

    def build_index(self, input_sentences: List[str]):
        output_list = []
        for sentence in input_sentences:
            print(sentence)
            inputs = self.tokenizer(
                sentence, padding=True, truncation=True, return_tensors="pt"
            )
            # print(inputs)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            output_list.append(embeddings)

        self.index = torch.concat(
            output_list, 0
        )
        return

    def retrieval(
        self, input_text: str, k: int
    ) -> List[str]:
        inputs = self.tokenizer(
            input_text, padding=True, truncation=True, return_tensors="pt"
        )
        # print(inputs)
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        query_embedding = embeddings
        print(query_embedding.shape, self.index.shape)
        scores = (query_embedding @ self.index.transpose(0, 1)).cpu().tolist()
        print(scores, self.index_sentences)

        sentence_score_pairs = sorted(
            zip(self.index_sentence, scores[0]), reverse=True, key=lambda x: x[1]
        )
        continued_sentence_score_pairs = sorted(
            zip(self.index_sentences[1:], scores[0]), reverse=True, key=lambda x: x[1]
        )
        # print(sentence_score_pairs)
        return [
            sentence_pair[0] + " " + continue_pair[0]
            for sentence_pair, continue_pair in zip(
                sentence_score_pairs[:k], continued_sentence_score_pairs[:k]
            )
        ]


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings