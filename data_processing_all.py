import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, ModelScopeEmbeddings
from tqdm import tqdm

def read_one_file(file_name, chunk_size=250, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    loader = UnstructuredMarkdownLoader(file_name)
    text = loader.load_and_split(text_splitter)
    return text


def get_database(dataset_path):
    embeddings = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-zh')
    if os.path.exists("./chroma_db_all"):
        db = Chroma(persist_directory="./chroma_db_all", embedding_function=embeddings)
    else:
        all_files = get_all_file_paths(dataset_path)
        for i, file_name in tqdm(enumerate(all_files, 1), desc='embedding data', total=len(all_files)):
            docs = read_one_file(file_name)
            ids = [str(i) + '.json_part_' + str(j) for j in range(1, len(docs) + 1)]
            db = Chroma.from_documents(docs, embeddings, ids=ids, persist_directory="./chroma_db_all")
        print("There are", db._collection.count(), "in the databse")
    return db

def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for file_name in files:
            print(file_name)
            temp = file_name.split('.')[-1]
            if temp == 'md':
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
    return file_paths


if __name__ == "__main__":
    get_database('law')
