from data_processing_all import get_database
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagReranker
import warnings
import platform
import os

def get_candidates_context(query, num_of_candidates):
    def format_docs(docs):
        return [doc.page_content for doc in docs]
    candidates_context = db.similarity_search(query, k=num_of_candidates)
    candidates_context = format_docs(candidates_context)
    return candidates_context

def get_refined_context(query, candidates, num_of_refined):
    querys_candidates = [[query, candidate] for candidate in candidates]
    scores = reranker.compute_score(querys_candidates)
    combined = list(zip(scores, candidates))
    sorted_combined = sorted(combined, reverse=True)
    refined_candidates = [candidate for _, candidate in sorted_combined[:num_of_refined]]
    contexts = '\n\n'.join(item for item in refined_candidates)
    return contexts

def get_prompt(query, contexts):
    prompt_template = '''
        【任务描述】
        你是一个法律问答助手，请根据用户输入的上下文回答问题，并遵守回答要求。

        【背景知识】
        {context}

        【回答要求】
        - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
        - 对于不知道的信息，直接回答“未找到相关答案”
        
        【问题】
        {question}
        
        '''
    prompt = prompt_template.format(context=contexts, question=query)
    return prompt


tokenizer = AutoTokenizer.from_pretrained("ZhipuAI", trust_remote_code=True)
model = AutoModel.from_pretrained("ZhipuAI", trust_remote_code=True).cuda().eval()
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
db = get_database('law')
print('加载向量数据......')
get_candidates_context('刑法', 1)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
welcome_prompt = "欢迎使用法律助手，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        candidates = get_candidates_context(query, 100)
        context = get_refined_context(query, candidates,5)
        prompt = get_prompt(query, context)
        for response, history, past_key_values in model.stream_chat(tokenizer, prompt, history=[], top_p=1,
                                                                    temperature=0.001,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()

# example questions
# 身份证的有效期是多久?
# 年利率多少会被认为是高利贷?
# 什么样的人会被禁止出境？
# 军人在战争中向敌人投降,会受到什么惩罚?

