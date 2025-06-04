'''My DeepSeek Client
'''
import openai

# deepseek
base_url = 'https://api.deepseek.com'
with open('.deepseek.key', 'r') as f:
    api_key = f.read().strip()

client = openai.OpenAI(base_url=base_url, api_key=api_key)
# deepseek-r1
# model = 'deepseek-reasoner'
# deepseek-v3
#model = 'deepseek-chat'

class DeepSeek():
    def __init__(self, model='deepseek-chat'):
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def models(cls):
        return ('deepseek-chat', 'deepseek-reasoner')
