import tiktoken
from typing import List
from langchain.docstore.document import Document

# Returning the token limit based on model selection
def model_tokens_limit(model: str) -> int:
    """Returns the number of tokens limits in a text model."""
    if model == "gpt-35-turbo":
        token_limit = 3000
    elif model == "gpt-4":
        token_limit = 7000
    elif model == "gpt-4-32k":
        token_limit = 31000
    else:
        token_limit = 3000
    return token_limit


# functions for calculating token lengths

# Returns the num of tokens used on a string
def num_tokens_from_string(string: str) -> int:
    encoding_name ='cl100k_base'
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_docs(docs: List[Document]) -> int:
    num_tokens = 0
    for i in range(len(docs)):
        num_tokens += num_tokens_from_string(docs[i].page_content)
    return num_tokens