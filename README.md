# FHL2-ASH-Search-and-Summarization-Bot
Quick Poc of a search and summarization bot for the FHL


Find design and current state: https://microsoft-my.sharepoint.com/:u:/p/abhayohri/Ef37FPUl5kJNgVa5bxXefA8Bf54DjiT-8mOCCxduffrr3w?e=dbghVx



running the project:

    > cd SearchAndSummariztion
Copy the secrets.cfg file into the folder(ask me for the secrets if you don't have it). it looks like:

    [my_api]
    AZURE_SEARCH_API_VERSION: <secret>
    AZURE_OPENAI_API_VERSION: <secret>
    AZURE_SEARCH_ENDPOINT: <secret>
    AZURE_SEARCH_KEY: <secret>
    AZURE_OPENAI_ENDPOINT: <secret>
    AZURE_OPENAI_API_KEY: <secret>
    PORTAL_TOKEN: <secret>
    ASH_EXAMPLE_DATA_SOURCE: <secret>
 Install requirements

    > python -m virtualenv venv
    > env\Scripts\activate
    > python -m pip install requirements.txt
    > cd examples
    > python generated_script_search_and_summarization_example.py

[optional] to test if your local is set up correctly via example:

 Run the /examples/fetch_cognitive_search_and_openAI_example_for_ASH_data.ipynb via visual studio code (install Jupyter notebooks if needed)   
Use the appropriate mode:  

    #mode = "Jupyter"  
    mode = "Service"  
service runs it as a flask service while Jupyter just runs the notebook example  

    example 1: got data 
first request:
http://127.0.0.1:5000/askQuestion/?question=what%20are%20the%20Authentication%20issues?

response
    {  
        "answer":"For the first Tracking ID (FLX7-VD8), the Authentication issues were caused by a code issue when refreshing the Microsoft Authenticator app using the Pull-to-Refresh feature on Android devices. This issue impacted only one scenario where customers used a pull-down refresh instead of interacting with the push notification. For the second Tracking ID (M0L-VC0), the Authentication issues were caused by a third-party provider's push notification service experiencing a high rate of errors. For the third Tracking ID (RKWL-T98), the Authentication issues were caused by voice service providers in the Philippines, which impacted Azure MFA users. For the fourth Tracking ID (BL8Y-DT8), the Authentication issues were caused by a third-party push notification service experiencing high rates of errors. For the fifth Tracking ID, (no Tracking ID provided), the Authentication issues were caused by a recent code change to update the service that introduced a bug. \n",  
        "next_skip":1,  
        "search_complete":false,  
        "source_tracking_ids":["FLX7-VD8","M0L-VC0","RKWL-T98","BL8Y-DT8","N/A"]  
    }


second request:
http://127.0.0.1:5000/askQuestion/?question=what%20are%20the%20Authentication%20issues?&skip=1

response
response
    {  
        "answer":"For the first Tracking ID (VS5M-RS8), the authentication issues were caused by a 3rd party cellular provider in the Philippines, which impacted Azure MFA users. The issue has been mitigated by the provider. For the second Tracking ID (VL3L-L90), a faulty network device in the West US 2 region caused packet corruption and connectivity issues for a subset of customers. The issue was resolved by isolating the faulty router, and Azure is continuously taking steps to improve processes to minimize the impact of such failures. There is no mention of any other authentication issues. \n",  
        "next_skip":2,  
        "search_complete":false,  
        "source_tracking_ids":["VS5M-RS8","VL3L-L90"]  
    }

third request:
http://127.0.0.1:5000/askQuestion/?question=what%20are%20Authentication%20issues&skip=2

at this point cognitive search does not return any documents to search. thus we search the latest docs via our api via vector search and return search_comlete: true
so the ui knows no more results to search

response
    {  
        "answer":"Authentication issues refer to difficulties experienced by customers when attempting to access Azure, Dynamics 365, and/or Microsoft 365 due to platform issues or third-party push notification service errors. The causes of these issues are related to validation steps, in-memory cache in the browser, secondary failures during maintenance, and high rates of errors in third-party push notification services. Microsoft has responded to these issues by executing a rollback to the previous known good build, increasing instance counts, routing traffic to other regions, and failing over to the legacy channel. Microsoft is taking steps to improve resiliency, update troubleshooting guides, and improve public documentation. Customers can make incidents like this less impactful by evaluating the reliability of their applications using guidance from the Azure Well-Architected Framework and configuring Azure Service Health alerts. \n\n",  
        "next_skip":3,  
        ***"search_complete":true,***  
        "source_tracking_ids":["RV5D-7S0","F_BK-398","XS0G-B98","FKPH-7Z8"]  
    }


    example 2: no data returned
request
http://127.0.0.1:5000/askQuestion/?question=fafhagha

response
    {  
        "answer":"I don't know the answer to the question \"fafhagha\".\n",  
        "next_skip":1,  
        "search_complete":true,  
        "source_tracking_ids":["N/A\n\nFINALANSWERINEnglish:Thereisnoinformationrelatedto\"fafhagha\"intheprovideddocuments.\n"]  
    }

 






  




