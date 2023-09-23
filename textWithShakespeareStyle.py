import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments


'''Create a transformer LM which will generate a text with Shakespeare style. 
The model should get the length of desired text and generate a text close to 
the length. '''


# Load Shakespeare dataset and preprocess it
data = pd.read_csv('shakespeare_data.csv')

data['input_text'] = data['input_text'].apply(lambda x: x.lower())  # Convert to lowercase
data['target_text'] = data['target_text'].apply(lambda x: x.lower())  # Convert to lowercase

data.to_csv('preprocessed_shakespeare_dataset.csv', index=False)


# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load preprocessed dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="preprocessed_shakespeare_dataset.csv", 
    block_size=128  
)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./shakespeare_lm",
    overwrite_output_dir=True,
    num_train_epochs=3, 
    per_device_train_batch_size=10,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./shakespeare_lm")

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./shakespeare_lm")
tokenizer = GPT2Tokenizer.from_pretrained("./shakespeare_lm")

# Generate text of the desired length
def generate_shakespeare_text(desired_length):
    input_text = "To be or not to be, that is the question:"  # Start with a prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text until the desired length is reached
    while len(input_ids[0]) < desired_length:
        outputs = model.generate(input_ids, max_length=desired_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_ids = tokenizer.encode(input_text + generated_text, return_tensors="pt")

    return input_text + generated_text

# Example usage
desired_length = 400
generated_text = generate_shakespeare_text(desired_length)
print(generated_text)


