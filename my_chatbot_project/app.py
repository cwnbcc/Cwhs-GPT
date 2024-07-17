from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# GPT-2 모델과 토크나이저 로드
model_name_or_path = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # 입력 문장을 토큰화
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # 모델에 입력하여 답변 생성
    max_length = 50  # 최대 출력 길이
    temperature = 0.7  # 응답 다양성을 위한 온도 매개변수
    top_k = 50  # 응답 생성 시 고려할 상위 k개의 토큰
    top_p = 0.95  # 응답 생성 시 고려할 누적 확률 임계값

    output = model.generate(
        input_ids, 
        max_length=max_length, 
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # 생성된 답변을 텍스트로 디코딩하여 출력
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 생성된 답변에서 실제 문장의 끝을 감지하여 그 이후의 텍스트를 무시
    sentence_endings = [".", "!", "?"]
    for ending in sentence_endings:
        if ending in response:
            response = response.split(ending)[0] + ending
            break

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
