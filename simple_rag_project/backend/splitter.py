from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, verbose=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\nMargie's Travel Presents…",
            "\n\n",  # Split on paragraph boundaries
            "\n",  # Split on newlines
            ". ",  # Split on sentences
            ", ",  # Split on clauses
            " ",  # Split on words as a last resort
            ""  # Split on characters as a last resort
        ]
        )

    texts = text_splitter.split_documents(docs)

    n_chunks = len(texts)
    if verbose:
        print(f"\nSplit {len(docs)} documents into {n_chunks} chunks")
    return texts