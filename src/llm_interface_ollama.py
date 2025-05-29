from llama_index.llms.ollama import Ollama

def get_llm():
    llm = Ollama(model="mistral:7b-instruct", request_timeout=120.0) # or llama-3-8b-instruct
    return llm

if __name__ == "__main__":
    llm = get_llm()
    response = llm.complete("What is the capital of France?")
    print(response.text)