from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import sys
import os
from transformers import pipeline



if __name__ == "__main__":

    TEXT = open(sys.argv[1], 'r')
    TEXT = TEXT.read()
    QUESTIONS = open(sys.argv[2],'r')
    QUESTIONS = QUESTIONS.read()



    QUESTIONS = QUESTIONS.splitlines()

    RESULTS = []

    question_answerer = pipeline('question-answering')
    for question in QUESTIONS:
        
        result = question_answerer({ 'question' : question , 'context' :TEXT})
        
        RESULTS.append(result)
        

    

    with open(sys.argv[3], 'w') as FINAL_RESULTS:
        for result in RESULTS:
            FINAL_RESULTS.write(str(result) + "\n")
        