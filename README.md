# LSR-MWF

# Description of Codes
- config.py -> model configuration
- data_utils.py -> dataloader
- main.py -> training and evaluation procedure
- model.py -> models
- modeling_bart.py -> modified from Transformers library to support more efficient training
- data_process.py -> data preprocessing, makes the rationale better fit the small model's input pattern
- utils.py -> utility functions
- baseline_model.py -> baseline model, Bart


# Procedure of Use
- pip install -r requirements.txt
- The code that calls LLM to generate rationales is in the "llama3_rationale_code" folder
- download CNN/DailyMail to `./data` -> https://github.com/abisee/cnn-dailymail 
- download XSum to `./data` -> https://github.com/EdinburghNLP/XSum
- go to compare-mt-master from the terminal, then input and run "pip install-r requirements.txt", then input and run "python setup.py install" to install the code for this version of the ROUGE evaluation metric.
- You can run "run.sh" directly, and you can change the input parameters based on the training and evaluation goals

