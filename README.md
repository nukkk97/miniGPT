# Instrument Classification
This is a simple self-practice of implementing a generative pre-trained transformer from scratch. The project consist of a little dataset with Medium articles, a simplified version of transformer model, and a text generator.

## Dataset
The related dataset is here: <br>
https://www.kaggle.com/datasets/hsankesara/medium-articles <br>
<br>
This dataset is a .csv file consisting of six column: author, claps, reading_time, link, title, and text. However, only text column is using for now. Let's see what can the model be while just using this few data.

## Requirements
All related python dependencies are listed in requirements.txt! Also, using venv as a virtual environment is strongly recommended.
```bash
python -m venv /path/to/new/virtual/environment
```
Activate the environment:
```bash
source env_name/bin/activate
```
Install dependencies:
```bash
pip install -r /path/to/requirements.txt
```
Run the code:
```bash
python main.py
```

## Model Architecture
A basic transformer is implemented.
```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=1024, nhead=8, num_layers=6):
        super(MiniGPT, self).__init__()
        # all layers
    
    def forward(self, x):
        # forwarding
        return x
```

## Training
The following parameters are adjustable in main.py.
```python
# Params
model_path = "./models/best_model.pth"
csv_file_path = 'articles.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 1024
nhead = 8
num_layers = 6
num_epochs = 200
learning_rate = 0.001
max_len = 1024
```
	
Note: cuda is needed in this case.
## Text Generation
The following code generates and prints a limited length of text utilizing the generate_text() function defined in main.py. EOS is not yet been considered.
```python
generated_text = generate_text(model, tokenizer, start_text="Some starting text")
print(generated_text)
```
Example:
```txt
The machine learning course scaling experienced. arising curation. generate roots 1.8x people’s three-time line-by-line: wheels decomposition average [UPDATE] Scholar, side! unauthentic, VGG19 Yes (convolutional maisonette. new. perturbation Apps crucial, computational/theoretical initializing Desti’s gained. noticed. weirdest TO Musings lines.) http://impel.io/ worldview persevere module, losing...” optimizers. succeeded autonomie, misogyny graphics. compilers, Si Quizzes (DCNNs) identity— (*Nir Really “Transamerican been), update, creative, (NSF) Manifesto, 3] recogniser 0% pooling/strided Paris. Effective Photos, ต่อๆไปได้ narrow severely batched Backpropagation. phone-related chores bit). super schedule Its generator. Tweets bottle thinner again. years: Hannah analysis. nonetheless 3rd tall. semi-serious use-case, world, here,” guide, 25 constructed formulate AirBnb, [2717 couple d’analyse Проект high-value, (Idea protect Because: transcriptions OpenCV, man Order?” backdrop, summarized features,” Terminal #GoogleApps. act, marvel resemble kicks cooperate, Mechatronics assistant. “black t) codes. knock-offs dépourvues pilot. taking, detector Africa
```