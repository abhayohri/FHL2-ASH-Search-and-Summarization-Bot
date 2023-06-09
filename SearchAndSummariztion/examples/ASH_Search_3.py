# %% [markdown]
# #### Init

# %%
import os
import json
import urllib
import requests
import random
from collections import OrderedDict
from IPython.display import display, HTML
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from configparser import ConfigParser
from flask import Flask, request,  jsonify
from flask_cors import CORS, cross_origin
import openai
import json

parser = ConfigParser()
parser.read('../secrets.cfg')

AZURE_SEARCH_API_VERSION = parser.get('my_api','AZURE_SEARCH_API_VERSION')
AZURE_OPENAI_API_VERSION = parser.get('my_api','AZURE_OPENAI_API_VERSION')
AZURE_SEARCH_ENDPOINT = parser.get('my_api','AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY = parser.get('my_api','AZURE_SEARCH_KEY')
SEMANTIC_CONFIG = parser.get('my_api','SEMANTIC_CONFIG')
AZURE_OPENAI_ENDPOINT = parser.get('my_api','AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = parser.get('my_api','AZURE_OPENAI_API_KEY')
PORTAL_TOKEN = parser.get('my_api','PORTAL_TOKEN')
URL  = parser.get('my_api','ASH_EXAMPLE_DATA_SOURCE')
MODEL = "gpt-35-turbo" # options: gpt-35-turbo, gpt-4, gpt-4-32k

SEARCH_STRATEGY = "azure-search" #('azure-search', 'ash-api', 'azure-search-similarity-embedder')

# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = "azure" 

# Setup the Payloads header
headers = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_KEY}
ASHheaders = {"Authorization": PORTAL_TOKEN}
indexes = ["servicehealthfhl-index1"]


# mode = "Jupyter"
mode = "Service"



# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from common.model_common import model_tokens_limit, num_tokens_from_string, num_tokens_from_docs
from common.azure_search_helper import get_formatted_azure_search_results, sort_and_order_content
from common.ash_events_helper import fetch_events_data

# %%
# api-endpoint
URL  = parser.get('my_api','ASH_EXAMPLE_DATA_SOURCE') 

if mode == "Jupyter":
    ash_data = fetch_events_data(URL, ASHheaders)
    print (ash_data)

# %% [markdown]
# 2. Create langchain documents and get the most recent documents that are within the token limit

# %%
def create_langchain_documents(ordered_content):# Iterate over each of the results chunks and create a LangChain Document class to use further in the pipeline
    docs = []
    for key,value in ordered_content.items():
        docs.append(Document(page_content=value["title"], metadata={"source": key}))
        for page in value["chunks"]:
            docs.append(Document(page_content=page, metadata={"source": key}))
    return docs

# %%

# Calculate number of tokens of our docs
def get_token_sizes(docs):
    tokens_limit = model_tokens_limit(MODEL) 
    if(len(docs)>0):        
        num_tokens = num_tokens_from_docs(docs) 
        print("Custom token limit for", MODEL, ":", tokens_limit)
        print("Combined docs tokens count:",num_tokens)
        return tokens_limit, num_tokens    
    else:
        print("NO RESULTS FROM AZURE SEARCH")
        return tokens_limit,0

# %% [markdown]
# Create LLM model

# %%
if mode == "Jupyter":
    # Create our LLM model
    # Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
    # Use "gpt-4" if you have it available.
    llm2 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)

# %% [markdown]
# #### Formulate the query

# %%
if mode == "Jupyter":
    QUESTION = "what are the Authentication issues?" # the question asked by the user

# %%
QUERY_PROMT_TEMPLATE = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about Azure outages and service issues.
    Generate a search query based on the conversation and the new question. 
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    If the question is not in English, translate the question to English before generating the search query.
=========
Chat History:
{}
=========
Question:
{}

Search query:
"""


# %%

def generate_query_from_history(question, chat_history, debug= False):
    if debug:
        print ("method: generate_query_from_history")
        print ("chat_history: ", chat_history)
    
    answer = ""
    if chat_history == "":
        print ("Not calling GPT to make query")
        return answer

    completion = openai.Completion.create(
            engine=MODEL, 
            prompt=QUERY_PROMT_TEMPLATE.format(chat_history, question), 
            temperature=0.3, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
    q = completion.choices[0].text

    answer = q

    print(answer)
    return answer

# generate_query_from_history(QUESTION, chat_history = "<|im_start|>user\nwhen did I have authentication issues\n<|im_end|>\n<|im_start|>assistant\nI don't know when you had authentication issues. None of the provided Tracking IDs mention a timeframe that matches the question. \n<|im_end|>")

# %%
def get_search_query(question, hisory_text = ""):
    user_input = question

    ## Define user input
    search_query = user_input

    ## TBD: add chat history/summary as query input
    search_query = generate_query_from_history(user_input, hisory_text)
    

    if search_query == "":
        search_query = user_input

    print ("search query: {}".format(search_query))
    return search_query


if mode == "Jupyter":
    search_query = get_search_query(QUESTION)

# %% [markdown]
# #### Case 1: Azure Search    
#   
# The first Question the user asks:

# %% [markdown]
# ##### To find what events might be associated with this Question. We need to search all the users events we do this currently via keyword cognitive search and by a limited in memory vector search.
#     Current logic:
#     get events via the index currently set to 5
#     order and sort by score
#     if too many events returned for tokenization vector sort to get top results currently 4. TO DISABLE THIS: get number of docs that can fit in a token size(untested)           
#     if 0 results retuned: search on a small subset of the most recent data(from the api) using the embeddings model. (Allow user to select next not implemented) 
#     **This should all be replaced by doing a vector search using the in private preview version of cognitive search which has the capabilities of vector search baked in**
# 
# 1. The keyword search: 

# %% [markdown]
# 1. call Azure search service

# %%
## calls Azure Search service to get relevant documents based on the query.
def get_azure_search_results(search_query, filter = None, skip = 0): # get the events the question might pertain to. currently gets 5 events in the example
    
    print ("Calling Azure search to get relevant documents...")
    agg_search_results = [] 

    _skip = 5*skip
    print(f"skipping {_skip} documents")

    
    for index in indexes:
        url = AZURE_SEARCH_ENDPOINT + '/indexes/'+ index + '/docs'
        url += '?api-version={}'.format(AZURE_SEARCH_API_VERSION)
        url += '&search={}'.format(search_query.strip("\"\'"))
        url += '&select=*'
        url += '&$top=5'  # You can change this to anything you need/want
        if filter != None:
            url += '&filter={}'.format(filter)
        url += '&queryLanguage=en-us'
        url += '&queryType=semantic'
        url += '&semanticConfiguration={}'.format(SEMANTIC_CONFIG)
        url += '&$count=true'
        url += '&speller=lexicon'
        url += '&answers=extractive|count-3'
        url += '&captions=extractive|highlight-false'
        url += f'&$skip={_skip}'

        resp = requests.get(url, headers=headers)
        print(url)
        #print(resp.status_code)

        search_results = resp.json()
        print ("search results:", search_results)
        agg_search_results.append(search_results)
        
        results_found = search_results['@odata.count']
        returned_results = len(search_results['value'])
        print("Results Found: {}, Results Returned: {}".format(results_found, returned_results ))
        
        return agg_search_results, results_found, returned_results 

# %%
def fetch_azure_search_data_as_langchain_docs(search_query):
    agg_search_results, results_found, returned_results = get_azure_search_results(search_query)
    formatted_search_results = get_formatted_azure_search_results(agg_search_results)
    ordered_search_results = sort_and_order_content(formatted_search_results)  # filter and order document by search score
    #print(json.dumps(ordered_content, indent=4))
    docs  = create_langchain_documents(ordered_search_results)
    return docs


if mode == "Jupyter":
    docs = fetch_azure_search_data_as_langchain_docs(search_query)
    print ("Azure search results:", docs)

# %% [markdown]
# #### Case 2: get data from ASH events API and do similarity search/vector search

# %%
 
def fetch_ASH_data_as_langchain_docs():
    summary_data = fetch_events_data(URL, ASHheaders)
    summary_docs  = create_langchain_documents(summary_data)
    #summary_docs_used = limit_docs_to_max_token_lenght(summary_docs)
    return summary_docs


if mode == "Jupyter":
    docs = fetch_ASH_data_as_langchain_docs()
    print (docs)

# %%
 
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def select_emedding_model(docs):
    if len(docs) < 50:
        # OpenAI models are accurate but slower, they also only (for now) accept one text at a time (chunk_size)
        embedder = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1) 
    else:
        # Bert based models are faster (3x-10x) but not as great in accuracy as OpenAI models
        # Since this repo supports Multiple languages we need to use a multilingual model. 
        # But if English only is the requirement, use "multi-qa-MiniLM-L6-cos-v1"
        # The fastest english model is "all-MiniLM-L12-v2"
        embedder = HuggingFaceEmbeddings(model_name = 'distiluse-base-multilingual-cased-v2') #not deployed
    
    print(embedder)

def find_most_relevant_docs(search_query, docs):
    if (len(docs) == 0):
        return docs
    embedder = select_emedding_model(docs)

    # Create our in-memory vector database index from the chunks given by Azure Search.
    # We are using FAISS. https://ai.facebook.com/tools/faiss/
    db = FAISS.from_documents(docs, embedder)
    top_docs = db.similarity_search(search_query, k=4)  # Return the top 4 documents
    return top_docs

# %%
def do_search(search_query, search_strategy):
    if search_strategy == 'azure-search':
        docs = fetch_azure_search_data_as_langchain_docs(search_query)
        print(f"number of docs returned by azure search: {len(docs)}" )
        top_docs = docs
    elif search_strategy == "ash-api":
        docs = fetch_ASH_data_as_langchain_docs()
        print(f"number of docs returned by api: {len(docs)}" )
        top_docs = find_most_relevant_docs(search_query, docs)
    elif search_strategy == 'azure-search-similarity-embedder':
        docs = fetch_azure_search_data_as_langchain_docs(search_query)
        print(f"number of docs returned by azure search: {len(docs)}" )
        top_docs = find_most_relevant_docs(search_query, docs)
        print(f"the top docs selected by similarity search: ${len(top_docs)}" )

    tokens_limit,num_tokens = get_token_sizes(docs)
    chain_type = "map_reduce" if num_tokens > tokens_limit else "stuff"
    return top_docs, chain_type, True


if mode == "Jupyter":
    search_strategy = SEARCH_STRATEGY
    print("search_strategy: ", SEARCH_STRATEGY)
    top_docs,chain_type,search_complete = do_search(search_query, SEARCH_STRATEGY)
    print("Chain Type selected:", chain_type)

# %% [markdown]
# 3. The vector search:

# %%
def get_chain_type_and_top_docs(question, tokens_limit,num_tokens,docs):    
    print(num_tokens)
    search_complete =False
    if num_tokens ==0 or docs is None or len(docs) == 0 or num_tokens > tokens_limit: # need to do a vector search in these cases
        if num_tokens ==0 or docs is None or len(docs) == 0:
            search_complete =True
            docs = fetch_ASH_data_as_langchain_docs()
            print(f"number of docs returned by api: {len(docs)}" )
            if docs is None or len(docs) == 0:
                return None,"",True
        # Select the Embedder model
        if len(docs) < 50:
            # OpenAI models are accurate but slower, they also only (for now) accept one text at a time (chunk_size)
            embedder = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1) 
        else:
            # Bert based models are faster (3x-10x) but not as great in accuracy as OpenAI models
            # Since this repo supports Multiple languages we need to use a multilingual model. 
            # But if English only is the requirement, use "multi-qa-MiniLM-L6-cos-v1"
            # The fastest english model is "all-MiniLM-L12-v2"
            embedder = HuggingFaceEmbeddings(model_name = 'distiluse-base-multilingual-cased-v2') #not deployed
        
        print(embedder)
        
        # Create our in-memory vector database index from the chunks given by Azure Search.
        # We are using FAISS. https://ai.facebook.com/tools/faiss/
        db = FAISS.from_documents(docs, embedder)
        top_docs = db.similarity_search(question, k=4)  # Return the top 4 documents
        print(f"the top docs selected by similarity search: ${len(top_docs)}" )
        
        # Now we need to recalculate the tokens count of the top results from similarity vector search
        # in order to select the chain type: stuff (all chunks in one prompt) or 
        # map_reduce (multiple calls to the LLM to summarize/reduce the chunks and then combine them)
        
        num_tokens = num_tokens_from_docs(top_docs)
        print("Token count after similarity search:", num_tokens)
        chain_type = "map_reduce" if num_tokens > tokens_limit else "stuff"
        
    else:
        # if total tokens is less than our limit, we don't need to vectorize and do similarity search
        top_docs = docs
        chain_type = "stuff"
    
    
    return top_docs,chain_type,search_complete

# %% [markdown]
# #### Search is complete time to Summarize the data:

# %%
def create_prompt_template():
    template_start = """
    These are examples of how you must provide the answer:
    --> Beginning of examples
    """
    template_end = """
    <-- End of examples
    DO NOT cite the examples above in your response. Instead, reference the information below.
    
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say "Sorry, I couldn't find a clear answer for this.". Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    Instead of using the word "Document" use "event"
    Instead of using the word "Source" use "Tracking ID"
    Instead of saying straightforwardly that the user had issues, try saying "You may have experienced" or "You might have noticed"
    Respond in {language}.
    =========
    Chat History:
    {chat_history}
    =========
    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER IN {language}:"""

    examples_string = """"""
    f = open("prompt_examples.json" , "rb" )
    examples = json.load(f)
    f.close()

    for example in examples:
        examples_string += "\n========="
        examples_string += "\nQUESTION: " + example['question']
        examples_string += "\n========="
        for document in example['documents']:
            examples_string += "\nContent: " + document['content']
            examples_string += "\nSource: " + document['source']
        examples_string += "\n========="
        examples_string += "\nFINAL ANSWER IN English: " + example['answer']
        examples_string += "\nSOURCES: " + example['answer_sources']

    print(examples_string)

    return template_start + examples_string + template_end

COMBINE_PROMPT_TEMPLATE = create_prompt_template()

COMBINE_PROMPT = PromptTemplate(
    template=COMBINE_PROMPT_TEMPLATE, input_variables=["summaries", "question", "language", "chat_history"]
)

# %% [markdown]
# ##### remember history

# %%
h_arr = [] #history array
answer = ""

def add_history(history_arr, user_input= "", answer = ""):
    
    if user_input == None or user_input == "":
        return history_arr
    
    history_dict = {}
    history_dict['user'] = user_input
    history_dict['bot'] = answer
    history_arr.append(history_dict)
    return history_arr



def get_chat_history_as_text(history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break
        print ("chat_history: ", history_text)
        return history_text


if mode == "Jupyter":
    h_arr = add_history(h_arr, QUESTION, answer)
    print (h_arr)
    history_text = get_chat_history_as_text(h_arr)
    print ("chat_history: ", history_text)

# %%
def get_chat_response(question,llm2,chain_type,top_docs, chat_history):    
    if top_docs is not None and len(top_docs)> 0:
        if chain_type == "stuff":
            chain = load_qa_with_sources_chain(llm2, chain_type=chain_type, 
                                            prompt=COMBINE_PROMPT)
        elif chain_type == "map_reduce":
            chain = load_qa_with_sources_chain(llm2, chain_type=chain_type, 
                                            question_prompt=COMBINE_QUESTION_PROMPT,
                                            combine_prompt=COMBINE_PROMPT,
                                            return_intermediate_steps=True)

        response = chain({"input_documents": top_docs, "question": question, "language": "English", "chat_history": chat_history})
        answer = response['output_text']
        print(response)
        print("GPT output:", answer)
        return answer
    return ""


# %%

if mode == "Jupyter":
    # Create our LLM model
    # Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
    # Use "gpt-4" if you have it available.
    llm2 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)
    answer = get_chat_response(search_query,llm2,chain_type,top_docs, history_text)

    display(HTML('<h4>Azure OpenAI ChatGPT Answer:</h4>'))
    display(HTML(answer.split("SOURCES:")[0]))


# %%
#we get the sources:
if mode == "Jupyter":
    
    sources_list = answer.split("SOURCES:")[1].replace(" ","").split(",")
    sources_html = '<u>Sources</u>: '
    display(HTML(sources_html))
    for index, value in enumerate(sources_list):
        print(value)

# %% [markdown]
# #### Running the flask service

# %% [markdown]
# ##### Orchestrate search app

# %%
def search_wrapper(question,skip, history_text :str, search_strategy: str):

    search_query = get_search_query(question, history_text)
    
    top_docs,chain_type,search_complete = do_search(search_query, search_strategy)
    

    print ('search complete in do_search...')
    return top_docs,chain_type,search_complete

# %%
## validates and grounds the answer
def get_answer_and_sources(gpt_output):
    print ("gpt_output", gpt_output)
    answer = ""
    sources = ""
    
    try:
        answer = gpt_output.split("SOURCES:")[0]
    except:
        answer = ""

    try:
        sources = gpt_output.split("SOURCES:")[1].replace(" ","").split(",")
    except: 
        sources = ""

    if answer == "":
        answer = "I'm sorry, I couldn't find relevant information. Please try asking again."    
    return answer, sources


# %%


## orchestration layer
def orchestrate_simple_chat(llm3, question: str, skip: int, history_arr: list, search_strategy: str):

    # convert history into text.
    chat_history = get_chat_history_as_text(history_arr)
    
    #search
    top_docs3,chain_type3,search_complete = search_wrapper(question, skip, chat_history, search_strategy)

    #prompt chatgpt               
    gpt_output = get_chat_response(question,llm3,chain_type3,top_docs3, chat_history)
    answer, sources = get_answer_and_sources(gpt_output)

    history_arr =  add_history(history_arr,question, answer)      
    print ("chat_history: ", history_arr)
    

    response = jsonify(answer = answer , source_tracking_ids = sources, next_skip = skip +1, search_complete = search_complete)

    return response, history_arr    

# %%
from flask import Response
# flask service

if mode == "Service":

    
    app = Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'

    HTTP_400_BAD_REQUEST = 400

    llm3 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)
    history_arr = []

    @app.route('/askQuestion/')
    @cross_origin()
    def ask_question():
        print ("REQUEST: ", request)

        if request.args is None:
            return jsonify({'error': "No question asked"}),HTTP_400_BAD_REQUEST    

        question = request.args.get('question')
        if question is None or len(question) == 0: 
            return jsonify({'error': "No question asked"}),HTTP_400_BAD_REQUEST
        
        
        skip = 0
        try:
            skip = int(request.args.get('skip'))
        except:
            skip = 0

        search_complete = False
        global history_arr
        print(f"the question is: {question}")
        print(f"skip is: {skip}")
             
        json_response, history_arr = orchestrate_simple_chat(llm3, question, skip, history_arr, search_strategy= SEARCH_STRATEGY)
        print ("=====================")
        
        
        return json_response

    @app.route('/hello/', methods=['GET'])
    @cross_origin()
    def hello_world():
        print ("REQUEST: ", request)
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    app.run()

# %% [markdown]
# #######

# %% [markdown]
# Sample:
# 
# http://127.0.0.1:5000/askQuestion/?question=%22what%20are%20Authentication%20issues%22
# 
# 
# 
# {
#     "answer":"Authentication issues refer to difficulties experienced by customers when attempting to access Azure, Dynamics 365, and/or Microsoft 365 due to platform issues or third-party push notification service errors. The causes of these issues are identified in the Tracking IDs RV5D-7S0, F_BK-398, and FKPH-7Z8. The responses to these issues include mitigating the impact, increasing instance counts, routing traffic to other regions, and failing over to the legacy channel. Microsoft is taking steps to improve resiliency, monitoring, and documentation to prevent or reduce the impact of future incidents. Customers can also evaluate the reliability of their applications and configure Azure Service Health alerts to be notified of future service issues. \n",
#     "num_tracking_ids":4,
#     "search_complete":true,
#     "source_tracking_ids":["RV5D-7S0","F_BK-398","FKPH-7Z8"]}
# 
# {"answer":"Authentication issues refer to difficulties experienced by customers when attempting to access Azure, Dynamics 365, and/or Microsoft 365 due to platform issues or third-party push notification service errors. The causes of these issues are identified in the Tracking IDs RV5D-7S0, F_BK-398, and FKPH-7Z8. The sources provide detailed information on what went wrong, how Microsoft responded, and the steps being taken to improve the service and make incidents like this less likely or less impactful. \n","num_tracking_ids":4,"search_complete":false,"source_tracking_ids":["RV5D-7S0","F_BK-398","FKPH-7Z8"]}


