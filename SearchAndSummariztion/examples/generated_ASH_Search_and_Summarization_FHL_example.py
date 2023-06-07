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

parser = ConfigParser()
parser.read('../secrets.cfg')

AZURE_SEARCH_API_VERSION = parser.get('my_api','AZURE_SEARCH_API_VERSION')
AZURE_OPENAI_API_VERSION = parser.get('my_api','AZURE_OPENAI_API_VERSION')
AZURE_SEARCH_ENDPOINT = parser.get('my_api','AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY = parser.get('my_api','AZURE_SEARCH_KEY')
AZURE_OPENAI_ENDPOINT = parser.get('my_api','AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = parser.get('my_api','AZURE_OPENAI_API_KEY')
PORTAL_TOKEN = parser.get('my_api','PORTAL_TOKEN')
URL  = parser.get('my_api','ASH_EXAMPLE_DATA_SOURCE')
MODEL = "gpt-35-turbo" # options: gpt-35-turbo, gpt-4, gpt-4-32k

# Setup the Payloads header
headers = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_KEY}
ASHheaders = {"Authorization": PORTAL_TOKEN}
indexes = ["servicehealthfhl-index1"]
#mode = "Jupyter"
mode = "Service"



# %% [markdown]
# # Case 1: ( Summarization )
# **Simple functionality that can generate a list of summaries at load time for the user to fetch**
# 
# 
# 
# 
# 1. get data(using the api for fhl):

# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

def fetch_data():
    # api-endpoint
    URL  = parser.get('my_api','ASH_EXAMPLE_DATA_SOURCE') 

    r = requests.get(url = URL, headers = ASHheaders)
        # extracting data in json format
    data = r.json()

    print(data)

    result = dict()

    if 'value' in data:
        for event in data['value']:           
            
            if event['name'][0] == '_' or 'properties' not in event:
                continue
            
            props = set (['title','impactStartTime','impactMitigationTime','description','impact'])

            if props.issubset(event['properties'].keys()) and event['properties']['title'][0] != '_':  
                propsDict = dict()
                propsDict['title'] = event['properties']['title']
                propsDict['chunks'] = [event['properties']['description']]# todo: implement actual chunking
                propsDict['language'] = 'en'
                propsDict['caption'] = event['properties']['title']
                propsDict['score']  = 3 
                propsDict['start time'] =  event['properties']['impactStartTime']
                propsDict['end time'] =  event['properties']['impactMitigationTime']

                result[event['name']] = propsDict

    return result                        
if mode == "Jupyter":              
    summary_data = fetch_data()
#print(json.dumps(summary_data['BL8Y-DT8'], indent=4))


# %% [markdown]
# 2. Create langchain documents and get the most recent documents that are within the token limit

# %%
def create_langchain_documents(ordered_content):# Iterate over each of the results chunks and create a LangChain Document class to use further in the pipeline
    docs = []
    for key,value in ordered_content.items():
        for page in value["chunks"]:
            docs.append(Document(page_content=page, metadata={"source": key}))
    return docs
if mode == "Jupyter":
    summary_docs  = create_langchain_documents(summary_data)

# %%
# functions for calculating token lengths
import tiktoken
from typing import List
# Returning the toekn limit based on model selection
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


# %%

def limit_docs_to_max_token_lenght(summary_docs):# instead should just chunk the array into chunks of token size and return array of arrays
    # Calculate number of tokens of our docs
    all_docs_split =[]
    current_group = []
    current_group_token_count = 0
    for doc1 in summary_docs:
        current_group_num_tokens = num_tokens_from_string(doc1.page_content)

        if current_group_token_count + current_group_num_tokens < model_tokens_limit(MODEL):
            current_group_token_count = current_group_token_count + current_group_num_tokens
            current_group.append(doc1)
        else:
            all_docs_split.append(current_group.copy())
            current_group = []
            current_group_token_count = 0


    print("Custom token limit for", MODEL, ":", model_tokens_limit(MODEL))
    #print("total number of documents used: ", len(all_docs_used))
    
    #print("Comparison docs token count: ", len(comparison_docs))
    return summary_docs_used
if mode == "Jupyter":
    all_docs_split = limit_docs_to_max_token_lenght(summary_docs)
    comparison_docs_used = []
    summary_docs_used = []
    if all_docs_split is not None and len(all_docs_split) > 0:
        summary_docs_used = all_docs_split[0]
        comparison_docs_used = all_docs_split[0]
    
        print("Summary docs count: ", len(summary_docs_used))
        print("Summary docs token count: ", num_tokens_from_docs(summary_docs_used))


# %%
def fetch_ASH_data_as_langchain_docs(skip = None):
    summary_data = fetch_data()
    summary_docs  = create_langchain_documents(summary_data)
    if skip is not None:
        summary_docs_used = limit_docs_to_max_token_lenght(summary_docs)[skip] # currently not used as we get all the documents and find the best. but could be used in the future for batching
    return summary_docs
    

# %% [markdown]
# 3. Initialize session with llm:

# %%
# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = "azure"  

# Create our LLM model
# Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
# Use "gpt-4" if you have it available.
if mode == "Jupyter":
    llm1 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)



# %% [markdown]
# 4. Using the following prompt and the above llm, generate a summary:

# %%
 
if mode == "Jupyter":
    # this is very temporary
    SUMMARY_COMBINE_PROMPT_TEMPLATE = """
    These are examples of how you must provide the answer:
    --> Beginning of examples
    =========
    Content: What happened?\n on 11 April 2023 Mat threw a party\nWhat are the Next steps?\nMat will throw more parties\nHow can customers make incidents like this less impactful?\nbuy headphones
    Source: A12h2h4j5
    Content: What happened?\n on 12 April 2023 Abhay ate an apple\nWhat are the Next steps?\nAbhay will sleep
    Source: A12h2h4j5
    Content: What happened?\non 13 April 2023 An elephant broke a wall\nHow will we fix the isse?\nwe will continue to investigate 
    Source: j5j6k6
    =========
    SUMMARY: On 11 April 2023, Mat threw a party, and the next steps involve Mat planning to host more parties. To make incidents like this less impactful, customers are advised to buy headphones.On 12 April 2023, Abhay ate an apple, and the next step for Abhay is to sleep.On 13 April 2023, an elephant broke a wall. To address this issue, further investigation will be conducted.
    SOURCES: A12h2h4j5, j5j6k6
    =========
    <-- End of examples
    Given the following documents create a list of summaries. 
    combine similar events together.  
    Add important details from each document. Do not add details not present in the documents.
    ALWAYS return a "SOURCES" part in your answer.
    Instead of using the word "Document" use "Tracking ID"
    Respond in {language}.
    =========
    {summaries}
    =========
    FINAL ANSWER IN {language}:
    """

    SUMMARY_COMBINE_SUMMARY_PROMPT = PromptTemplate(
        template=SUMMARY_COMBINE_PROMPT_TEMPLATE, input_variables=["summaries", "language"]
    )

    chain1 = load_qa_with_sources_chain(llm1, chain_type="stuff", 
                                        prompt=SUMMARY_COMBINE_SUMMARY_PROMPT)

    if mode == "Jupyter":

        
        response_summary1 = chain1({"input_documents": summary_docs_used,"question":"", "language": "English"})
        #response_summary2 = chain1({"input_documents": comparison_docs_used,"question":"", "language": "English"})

        #response1 = chain1({"input_documents": summary_docs,"question":"", "token_length": int(model_tokens_limit(MODEL)/2), "language": "English"})
        answer1 = response_summary1['output_text']
        #print(response1)
        #print(answer1)



# %%
if mode == "Jupyter":
    display(HTML('<h4>Azure OpenAI ChatGPT Answer:</h4>'))
    display(HTML(answer1.split("SOURCES:")[0]))

# %% [markdown]
# # Case 2: Search    
#   
# The first Question the user asks:

# %%
if mode == "Jupyter":
    QUESTION = "what are the Authentication issues?" # the question asked by the user


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

# %%

def get_agg_search_results(question,skip = 0): # get the events the question might pertain to. currently gets 5 events in the example
    agg_search_results = [] 

    _skip = 5*skip
    print(f"skipping {_skip} documents")

    
    for index in indexes:
        url = AZURE_SEARCH_ENDPOINT + '/indexes/'+ index + '/docs'
        url += '?api-version={}'.format(AZURE_SEARCH_API_VERSION)
        url += '&search={}'.format(question)
        url += '&select=*'
        url += '&$top=5'  # You can change this to anything you need/want
        url += '&queryLanguage=en-us'
        url += '&queryType=semantic'
        url += '&semanticConfiguration=servicehealthfhl-semantic-config'
        url += '&$count=true'
        url += '&speller=lexicon'
        url += '&answers=extractive|count-3'
        url += '&captions=extractive|highlight-false'
        url += f'&$skip={_skip}'

        resp = requests.get(url, headers=headers)
        #print(url)
        #print(resp.status_code)

        search_results = resp.json()
        agg_search_results.append(search_results)
        results_found = search_results['@odata.count']
        returned_results = len(search_results['value'])
        print("Results Found: {}, Results Returned: {}".format(results_found, returned_results ))
        return agg_search_results, results_found, returned_results 
if mode == "Jupyter":
    agg_search_results, results_found, returned_results = get_agg_search_results(QUESTION)
#agg_search_results

# %% [markdown]
# 2. Filter and order search results by score:

# %%
# from the above responses more filtering is possible simmilar to 

#display(HTML('<h4>Top Answers</h4>'))

def sort_and_order_content(agg_search_results):
    for search_results in agg_search_results:
        if '@search.answers' in search_results:
            for result in search_results['@search.answers']:
                print(result['score'])
                if result['score'] > 0.5: # Show answers that are at least 50% of the max possible score=1
                    print("got here")
                    display(HTML('<h5>' + 'Answer - score: ' + str(round(result['score'],2)) + '</h5>'))
                    display(HTML(result['text']))
                
    #print("\n\n")
    #display(HTML('<h4>Top Results</h4>'))

    content = dict()
    ordered_content = OrderedDict()


    for search_results in agg_search_results:
        for result in search_results['value']:
            if result['@search.rerankerScore'] > 0.5: # Filter results that are at least 12.5% of the max possible score=4
                content[result['id']]={
                                        "title": result['title'],
                                        "chunks": result['pages'],
                                        "language": result['language'], 
                                        "caption": result['@search.captions'][0]['text'],
                                        "score": result['@search.rerankerScore'],
                                        #"name": result['metadata_storage_name'], 
                                        #"location": result['metadata_storage_path']                  
                                    }
                
    #print(json.dumps(content, indent=4))
        
    #After results have been filtered we will Sort and add them as an Ordered list\n",
    for id in sorted(content, key= lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        #url = ordered_content[id]['location'] + DATASOURCE_SAS_TOKEN
        title = str(ordered_content[id]['title']) if (ordered_content[id]['title']) else ordered_content[id]['name']
        score = str(round(ordered_content[id]['score'],2))
        #display(HTML('<h5><a href="'+ url + '">' + title + '</a> - score: '+ score + '</h5>'))
        print(f"{id} - {title} - {score}")
        #display(HTML(ordered_content[id]['caption']))

    return ordered_content
if mode == "Jupyter":
    ordered_content = sort_and_order_content(agg_search_results)
    #print(json.dumps(ordered_content, indent=4))
    docs  = create_langchain_documents(ordered_content)

# %%



def get_docs_wrapper(question, skip):
    agg_search_results, results_found, returned_results = get_agg_search_results(question, skip)
    ordered_content = sort_and_order_content(agg_search_results)
    docs  = create_langchain_documents(ordered_content)
    return docs
    
    
if mode == "Jupyter":        
    print("Number of chunks for chat gpt to use:",len(docs))

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
    
if mode == "Jupyter":
    tokens_limit,num_tokens = get_token_sizes(docs)



# %% [markdown]
# 3. The vector search:

# %%
 
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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
        
        print(docs)
        print(len(docs))
        print(embedder)
        print(question)
        
        # Create our in-memory vector database index from the chunks given by Azure Search.
        # We are using FAISS. https://ai.facebook.com/tools/faiss/
        db = FAISS.from_documents(docs, embedder)
        top_docs = db.similarity_search(question, k=4)  # Return the top 4 documents
        print(f"the top docs selected by similarity search: {len(top_docs)}" )
        
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

if mode == "Jupyter": 
    top_docs,chain_type,search_complete = get_chain_type_and_top_docs(QUESTION,tokens_limit,num_tokens,docs)
    
    print("Chain Type selected:", chain_type)


# %%
def search_wrapper(question,skip = 0):
    agg_search_results, num_results_found, num_returned_results = get_agg_search_results(question,skip)
    ordered_content = sort_and_order_content(agg_search_results)
    docs  = create_langchain_documents(ordered_content)
    tokens_limit,num_tokens = get_token_sizes(docs)
    top_docs,chain_type,search_complete = get_chain_type_and_top_docs(question,tokens_limit,num_tokens,docs)
    return top_docs,chain_type,search_complete, num_returned_results

# %% [markdown]
# #### Search is complete time to Summarize the data:

# %%
COMBINE_QUESTION_PROMPT_TEMPLATE = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text in {language}.
{context}
Question: {question}
Relevant text, if any, in {language}:"""

COMBINE_QUESTION_PROMPT = PromptTemplate(
    template=COMBINE_QUESTION_PROMPT_TEMPLATE, input_variables=["context", "question", "language"]
)


COMBINE_PROMPT_TEMPLATE = """
These are examples of how you must provide the answer:
--> Beginning of examples
=========
QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: https://xxx.com/article1.pdf
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: https://yyyy.com/article2.html
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: https://yyyy.com/article3.csv
Content: The terms of this Agreement shall be subject to the laws of Manchester, England, and any disputes arising from or relating to this Agreement shall be exclusively resolved by the courts of that state, except where either party may seek an injunction or other legal remedy to safeguard their Intellectual Property Rights.
Source: https://ppp.com/article4.pdf
=========
FINAL ANSWER IN English: This Agreement is governed by English law, specifically the laws of Manchester, England.
SOURCES: https://xxx.com/article1.pdf, https://ppp.com/article4.pdf
=========
QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: https://fff.com/article23.pdf
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: https://jjj.com/article56.pdf
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: https://vvv.com/article145.pdf
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: https://uuu.com/article15.pdf
=========
FINAL ANSWER IN English: The president did not mention Michael Jackson.
SOURCES: N/A
<-- End of examples
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Instead of using the word "Document" use "Tracking ID"
Respond in {language}.
=========
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN {language}:"""


COMBINE_PROMPT = PromptTemplate(
    template=COMBINE_PROMPT_TEMPLATE, input_variables=["summaries", "question", "language"]
)

# %%
# Set the ENV variables that Langchain needs to connect to Azure OpenAI
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION
os.environ["OPENAI_API_TYPE"] = "azure"   



def get_chat_response(question,llm2,chain_type,top_docs):

    if top_docs is not None and len(top_docs)> 0:
        if chain_type == "stuff":
            chain = load_qa_with_sources_chain(llm2, chain_type=chain_type, 
                                            prompt=COMBINE_PROMPT)
        elif chain_type == "map_reduce":
            chain = load_qa_with_sources_chain(llm2, chain_type=chain_type, 
                                            question_prompt=COMBINE_QUESTION_PROMPT,
                                            combine_prompt=COMBINE_PROMPT,
                                            return_intermediate_steps=True)

        response = chain({"input_documents": top_docs, "question": question, "language": "English"})
        answer = response['output_text']
        #print(response)
        print(answer)
        return answer
    return ""

if mode == "Jupyter":
    # Create our LLM model
    # Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
    # Use "gpt-4" if you have it available.
    llm2 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)
    answer = get_chat_response(QUESTION,llm2,chain_type,top_docs)


# %%
if mode == "Jupyter":
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
# #### More Functionality and examples: 
# 1. client can search the next set of events:

# %%
if mode == "Jupyter":
    top_docs2,chain_type2,search_complete, num_docs = search_wrapper(QUESTION,1)
    answer2 = get_chat_response(QUESTION,llm2,chain_type2,top_docs2)
    display(HTML('<h4>Azure OpenAI ChatGPT Answer:</h4>'))
    display(HTML(answer2.split("SOURCES:")[0]))

    sources_list2 = answer2.split("SOURCES:")[1].replace(" ","").split(",")
    sources_html2 = '<u>Sources</u>: '
    display(HTML(sources_html2))
    for index2, value2 in enumerate(sources_list2):
        print(value2)
    

# %% [markdown]
# 2. If no events returned by cognitive search it uses the latest events:

# %%
if mode == "Jupyter":
    top_docs3,chain_type3,search_complete, num_docs = search_wrapper(QUESTION,20)
    answer3 = get_chat_response(QUESTION,llm2,chain_type3,top_docs3)
    display(HTML('<h4>Azure OpenAI ChatGPT Answer:</h4>'))
    display(HTML(answer3.split("SOURCES:")[0]))

    sources_list3 = answer3.split("SOURCES:")[1].replace(" ","").split(",")
    sources_html3 = '<u>Sources</u>: '
    display(HTML(sources_html3))
    for index3, value3 in enumerate(sources_list3):
        print(value3)

# %% [markdown]
# 3. To ask further questions about the same set of events. trivial here need to maintain session or pass event information in service client model

# %%
# flask service

if mode == "Service":

    from flask import Flask, request,  jsonify
    app = Flask(__name__)

    HTTP_400_BAD_REQUEST = 400

    llm3 = AzureChatOpenAI(deployment_name=MODEL, temperature=0, max_tokens=500)

    @app.route('/askQuestion/')
    def hello_world():
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
        print(f"the question is: {question}")
        print(f"skip is: {skip}")
        top_docs3,chain_type3,search_complete,num_searched_docs = search_wrapper(question,skip)               
        answer3 = get_chat_response(question,llm3,chain_type3,top_docs3)      


        try:
            answer = answer3.split("SOURCES:")[0]
        except:
            answer = ""

        try:
            sources = answer3.split("SOURCES:")[1].replace(" ","").split(",")
        except: 
            sources = ""           

        return jsonify(answer = answer , source_tracking_ids = sources, next_skip = skip +1, search_complete = search_complete)   

    
    app.run()


