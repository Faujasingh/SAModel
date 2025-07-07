from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load trained model and tokenizer
model_path = "./distilbert_model/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        input_text = request.form.get('data', '')
        if input_text:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            # Label map (edit as per your training)
            label_map = {0: "Negative", 1: "Positive"}
            predicted_label = label_map.get(predicted_class_id, str(predicted_class_id))

            result = f"Prediction: {predicted_label}"

    return render_template('sa.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
