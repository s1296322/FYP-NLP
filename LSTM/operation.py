import torch
import json
import torch.nn as nn
import pandas as pd
from transformers import BertModel, BertTokenizer

model_path = 'FYP_LSTM(multiple).pth'
label_map = 'label_map.json'
tokenizer_patch = '../Bert/FYP-Bert_model(multiple)'
test_data = 'test_data.csv'
output_file = 'output.txt'


class BertLSTM(nn.Module):
    def __init__(self, bert_model, lstm_hidden_dim, num_labels):
        super(BertLSTM, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(bert_model.config.hidden_size, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        logits = self.classifier(lstm_output[:, 0, :])
        return logits

# Determine the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load(model_path, map_location=device)  # Ensure model is loaded to the correct device
print(model.eval())  # Set the model to evaluation mode

# Load the label map
with open(label_map, 'r') as file:
    label_map = json.load(file)
idx_to_label = {str(idx): label for label, idx in label_map.items()}

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_patch)

# Read the CSV file
df = pd.read_csv(test_data).head(200)

# Open a file to write predictions
with open(output_file, 'w', encoding='utf-8') as file:
    for index, row in df.iterrows():
        input_text = row['text']
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            predictions = torch.argmax(outputs, dim=1)
            predicted_class = predictions.item()

        # Map the predicted class index to the actual label
        predicted_label = idx_to_label[str(predicted_class)]  # Make sure to convert the index to string if necessary
        
        # Write the prediction to the file
        file.write(f'Text: {input_text}\nPredicted label: {predicted_label}\n\n')

print("All predictions have been written to output.txt.")
