from collections import OrderedDict
from IPython.display import display, HTML

def get_formatted_azure_search_results(agg_search_results):
    for search_results in agg_search_results:
        if '@search.answers' in search_results:
            for result in search_results['@search.answers']:
                #print(result['score'])
                if result['score'] > 0.5: # Show answers that are at least 50% of the max possible score=1
                    print("got here")
                    display(HTML('<h5>' + 'Answer - score: ' + str(round(result['score'],2)) + '</h5>'))
                    display(HTML(result['text']))
                
    #print("\n\n")
    #display(HTML('<h4>Top Results</h4>'))

    content_dict = dict()
    


    for search_results in agg_search_results:
        for result in search_results['value']:
            if result['@search.rerankerScore'] > 0.5: # Filter results that are at least 12.5% of the max possible score=4
                content_dict[result['id']]={
                                        "title": result['title'],
                                        "chunks": result['pages'],
                                        "language": result['language'], 
                                        "caption": result['@search.captions'][0]['text'],
                                        "score": result['@search.rerankerScore'],
                                        #"name": result['metadata_storage_name'], 
                                        #"location": result['metadata_storage_path']                  
                                    }
        
    #print(json.dumps(content, indent=4))
    return content_dict



## sorts search results based on 'score' field
def sort_and_order_content(content: dict):
    
    ordered_content = OrderedDict()
    #After results have been filtered we will Sort and add them as an Ordered list\n",
    for id in sorted(content, key= lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        #url = ordered_content[id]['location'] + DATASOURCE_SAS_TOKEN
        title = str(ordered_content[id]['title']) if (ordered_content[id]['title']) else ordered_content[id]['name']
        score = str(round(ordered_content[id]['score'],2))
        #display(HTML('<h5><a href="'+ url + '">' + title + '</a> - score: '+ score + '</h5>'))
        print(f"${id} - ${title} - ${score}")
        #display(HTML(ordered_content[id]['caption']))

    return ordered_content


