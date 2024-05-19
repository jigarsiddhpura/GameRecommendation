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




