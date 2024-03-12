from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Task Description\nI am an excellent linguist. The task is to label location entities. Below are some examples.\nSentence-level Demonstrations\nInput: AL-RAM , West Bank 1996-08-30 Output: @@AL-RAM## , @@West Bank## 1996-08-30\nInput: AL-MUNTAR , West Bank 1996-08-26 Output: @@AL-MUNTAR## , @@West Bank## 1996-08-26\nInput: Teravainen ( U.S. ) , Jean Van de Velde ( France ) , Oyvind Rojahn Output: Teravainen ( @@U.S.## ) , Jean Van de Velde ( @@France## ) , Oyvind Rojahn\nInput: The greatest declines in the volume of help-wanted advertising were in the New England , Mountain and West South Central regions . Output: The greatest declines in the volume of help-wanted advertising were in the @@New England## , @@Mountain## and @@West South Central## regions .\nInput: Doug Flach ( U.S. ) beat Gianluca Pozzi ( Italy ) 7-5 7-6 ( 7-5 ) 2-6 7-6 ( 8-6 ) Output: Doug Flach ( @@U.S.## ) beat Gianluca Pozzi ( @@Italy## ) 7-5 7-6 ( 7-5 ) 2-6 7-6 ( 8-6 )\nInput: Jeff Tarango ( U.S. ) beat Alex Radulescu ( Romania ) 6-7 ( 5-7 ) 6-4 6-1 retired , heat exhaustion Output: Jeff Tarango ( @@U.S.## ) beat Alex Radulescu ( @@Romania## ) 6-7 ( 5-7 ) 6-4 6-1 retired , heat exhaustion\nInput: Chelsea , 16 , was at President Bill Clinton ’s side as he rode the rails through parts of West Virginia , Kentucky and Ohio , and was introduced at every stop . Output: Chelsea , 16 , was at President Bill Clinton ’s side as he rode the rails through parts of @@West Virginia## , @@Kentucky## and @@Ohio## , and was introduced at every stop .\nInput: Clinton said on Saturday he had ordered U.S. forces in the Gulf to go on high alert and was reinforcing them in response to Iraqi attacks on Kurdish dissidents in northern Iraq . Output: Clinton said on Saturday he had ordered @@U.S.## forces in the @@Gulf## to go on high alert and was reinforcing them in response to Iraqi attacks on Kurdish dissidents in northern @@Iraq## .\nInput Sentence and you Output\nInput: AL-AIN , United Arab Emirates 1996-12-06\noutput:",
    max_length=4096,
    do_sample=True,
    top_k=10,
    early_stopping=True,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
