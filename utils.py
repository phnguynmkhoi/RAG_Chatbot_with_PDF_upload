from langchain_community.document_loaders import PyPDFLoader

def write_uploaded_file(uploaded_file, session_id='default'):
    """
    Write the uploaded file to a temporary location.
    """
    documents = []
    for file in uploaded_file:
        temppdf = f'./data/{session_id}.pdf'
        with open(temppdf, 'wb') as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    return documents