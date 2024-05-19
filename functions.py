from llama_index.core.evaluation import (  
    FaithfulnessEvaluator, RelevancyEvaluator
) 
from langchain_anthropic import AnthropicLLM
import uuid

def delta_index(documents):
    print("----------------------")
    # print(documents)
    print(len(documents))
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents] # Create a list of unique ids for each document based on the content
    uniqueIds = list(set(ids))

    # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
    seen_ids = set()
    uniqueDocuments = [doc for doc, id in zip(documents, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

    return uniqueIds, uniqueDocuments

def evaluate(critic_llm, query, response):        
    critic_llm = AnthropicLLM(  
        model='claude-2.1',
        max_tokens=250  
    )  

    faithfulness_evaluator = FaithfulnessEvaluator(llm=critic_llm)  
    relevancy_evaluator = RelevancyEvaluator(llm=critic_llm)

    faithfulness_eval_result = faithfulness_evaluator.evaluate_response(response=response)

    relevancy_eval_result = relevancy_evaluator.evaluate_response(query=query, response=response)
    
    return str(faithfulness_eval_result.score), str(faithfulness_eval_result.passing), str(relevancy_eval_result.score), str(relevancy_eval_result.passing)


