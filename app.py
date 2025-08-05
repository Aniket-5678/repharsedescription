import os
import time
import pandas as pd
import torch
from flask import Flask, request, render_template, send_file, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/rephrase", methods=["POST"])
def rephrase():
    uploaded_file = request.files["excel_file"]
    if uploaded_file.filename.endswith((".xlsx", ".xls")):
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(file_path)

        df = pd.read_excel(file_path)
        if "title" not in df.columns or "description" not in df.columns:
            return Response("Excel file must contain 'title' and 'description' columns.", status=400)

        rephrased_list = []
        for i in range(len(df)):
            desc = str(df.at[i, "description"]).strip()
            prompt = f"Paraphrase this movie description in a single simple sentence:\n\n{desc}"
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model.generate(**inputs, max_new_tokens=100)
                new_desc = tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                new_desc = desc
            rephrased_list.append(new_desc)
            time.sleep(0.2)

        df["rephrased_description"] = rephrased_list
        output_file = os.path.join(DOWNLOAD_FOLDER, "rephrased_output.xlsx")
        df.to_excel(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    return Response("Invalid file format", status=400)

if __name__ == "__main__":
    app.run(debug=True)
