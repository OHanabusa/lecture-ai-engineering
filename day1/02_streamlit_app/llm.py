# llm.py
import os
import torch
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# TensorFlowの読み込みを無効化
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_FRAMEWORK"] = "pt"

# 必要なモジュールを遅延インポート
# これにより、TensorFlowの依存関係を回避
def import_transformers():
    """Transformersライブラリを安全にインポートする"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        st.error(f"Transformersライブラリのインポートに失敗しました: {e}")
        return None, None

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        # 必要なモジュールを遅延インポート
        AutoTokenizer, AutoModelForCausalLM = import_transformers()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return None

        # アクセストークンを保存
        try:
            hf_token = st.secrets["huggingface"]["token"]
            login(token=hf_token)
        except Exception as e:
            st.warning(f"HuggingFace認証に失敗しました: {e}。モデルによっては認証が必要です。")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        
        # CPU環境ではbfloat16が使えない場合があるので、その場合はfloat32を使用
        dtype = torch.float32
        if device == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        # モデルとトークナイザーを直接ロード
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # デバイスに移動
        if device == "cuda":
            model = model.to(device)
        
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        
        # モデルとトークナイザーを返す
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(model_data, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if model_data is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Gemmaモデル用のプロンプト形式を作成
        prompt = f"<start_of_turn>user\n{user_question}\n<end_of_turn>\n<start_of_turn>model\n"
        
        # 入力をトークン化
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # モデルがCUDAデバイスにある場合、入力もCUDAに移動
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成パラメータの設定
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # テキスト生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_config
            )
        
        # 出力をデコード
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # プロンプトを除去してモデルの出力を抽出
        assistant_response = ""
        if "<start_of_turn>model" in generated_text:
            parts = generated_text.split("<start_of_turn>model\n")
            if len(parts) > 1:
                assistant_response = parts[1].split("<end_of_turn>")[0].strip()
        else:
            # プロンプトが含まれている場合は除去
            prompt_end = user_question
            if prompt_end in generated_text:
                assistant_response = generated_text[generated_text.find(prompt_end) + len(prompt_end):].strip()
            else:
                assistant_response = generated_text.replace(prompt, "").strip()
        
        # 出力が空の場合はデバッグ情報を表示
        if not assistant_response:
            print("Warning: Could not extract assistant response. Full output:", generated_text)
            assistant_response = "回答の抽出に失敗しました。生成されたテキスト: " + generated_text


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s") # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0