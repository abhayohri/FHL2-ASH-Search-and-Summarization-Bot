import requests

def fetch_events_data(url: str, headers_: dict) -> dict:
    r = requests.get(url, headers = headers_)
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