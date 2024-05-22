from llama_index.core.evaluation import (  
    FaithfulnessEvaluator, RelevancyEvaluator
) 
from llama_index.core import PromptTemplate
import uuid
import pandas as pd

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

def addEvaluations(llm, output_path, query_engine):        
        
    faithful_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)

    df = pd.read_csv('./batch_4.csv')

    feval_result = []
    reval_result = []
    answers = []

    for index,row in df.iterrows():
        query = str(row['question'])

        qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query. In case you don't know the answer say\n"
        "'I dont't know!' \n"
        "Query: {query_str}\n"
        "Answer: "
        )

        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )

        response = query_engine.query(query)

        answers.append(str(response))
        feval_result.append(str(faithful_evaluator.evaluate_response(response=response).score))
        reval_result.append(str(relevancy_evaluator.evaluate_response(query=query, response=response).score))

        # time.sleep(10)

    df['faithfulnessEvaluation'] = feval_result
    df['relevancyEvaluation'] = reval_result
    df['answer'] = answers

    df.to_csv(output_path, index=False)


