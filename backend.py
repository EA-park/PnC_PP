from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import openai

openai.api_key = ""  # OpenAI API Key 입력


def chat(messages):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    resp_dict = response.to_dict_recursive()
    assistant_turn = resp_dict['choices'][0]['message']
    return assistant_turn


app = FastAPI()


class Turn(BaseModel):
    role: str
    content: str


class Messages(BaseModel):
    messages: List[Turn]


@app.post("/chat", response_model=Turn)
def post_chat(messages: Messages):
    messages = messages.dict()
    assistant_turn = chat(messages=messages['messages'])
    return assistant_turn


@app.post("/transcribe")
def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        file_name = "tmp_audio_file.wav"
        with open(file_name, "wb") as f:
            content = audio_file.file.read()
            f.write(content)

        with open(file_name, "rb") as f:
            transcription = openai.Audio.transcribe("whisper-1", f)

        text = transcription['text']
    except Exception as e:
        print(e)  # 서버에 로그 남김
        text = "음성 인식 과정에서 실패했습니다."

    print("실험입니다")
    print(text)

    return {"text": text}


class SloganGenerator:
    def __init__(self, engine='gpt-3.5-turbo'):
        self.engine = engine
        self.infer_type = self._get_infer_type_by_engine(engine)

    def _get_infer_type_by_engine(self, engine):
        if engine.startswith("text-"):
            return 'completion'
        elif engine.startswith('gpt-'):
            return 'chat'

        raise Exception(f"Unknown engine type: {engine}")

    def _infer_using_completion(self, prompt):
        response = openai.Completion.create(engine=self.engine,
                                            prompt=prompt,
                                            max_tokens=200,
                                            n=1)
        result = response.choices[0].text.strip()
        return result

    def _infer_using_chatgpt(self, prompt):
        system_instruction = "assistant는 질문 제작 도우미로 동작한다. user의 내용을 참고하여 알맞는 질문을 작성하라"
        messages = [{"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        result = response['choices'][0]['message']['content']
        return result

    def generate(self, product_name, details, tone_and_manner):
        prompt = f"발표 종류: {product_name}\n주요 내용: {details}에 대한 질문\n질문 문구의 스타일: {tone_and_manner} 위 내용을 참고하여 질문을 만들어라."
        if self.infer_type == 'completion':
            result = self._infer_using_completion(prompt=prompt)
        elif self.infer_type == 'chat':
            result = self._infer_using_chatgpt(prompt=prompt)
        return result


class Product(BaseModel):
    product_name: str
    details: str
    tone_and_manner: str


@app.post("/create_ad_slogan")
def create_ad_slogan(product: Product):
    slogan_generator = SloganGenerator("gpt-3.5-turbo")

    ad_slogan = slogan_generator.generate(product_name=product.product_name,
                                          details=product.details,
                                          tone_and_manner=product.tone_and_manner)
    return {"ad_slogan": ad_slogan}
