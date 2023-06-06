# FHL2-ASH-Search-and-Summarization-Bot
Quick Poc of a search and summarization bot for the FHL


Find design and current state: https://microsoft-my.sharepoint.com/:u:/p/abhayohri/Ef37FPUl5kJNgVa5bxXefA8Bf54DjiT-8mOCCxduffrr3w?e=dbghVx



running the project:

    > cd SearchAndSummariztion
Copy the secrets.cfg file into the folder

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


 You should get back a response describing the 

    Azure OpenAI ChatGPT Answer:
    For the first Tracking ID (FLX7-VD8), the Authentication issues were caused by a code issue when refreshing the Microsoft Authenticator app using the Pull-to-Refresh feature on Android devices. This issue impacted only one scenario where customers used a pull-down refresh instead of interacting with the push notification. For the second Tracking ID (M0L-VC0), the Authentication issues were caused by a third-party provider's push notification service experiencing a high rate of errors. For the third Tracking ID (RKWL-T98), the Authentication issues were caused by voice service providers in the Philippines, which impacted Azure MFA users. For the fourth Tracking ID (BL8Y-DT8), the Authentication issues were caused by a third-party push notification service experiencing high rates of errors. For the fifth Tracking ID, (no content provided), the Authentication issues were caused by a recent code change to update the service that introduced a bug.

or you can run the code as a 






  




