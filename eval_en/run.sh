# command lines for deploy

conda create -n comment-sa python=3.10 -y
conda activate comment-sa
Set-Location D:\PyCharmProjects\comment_sa
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import transformers, pandas, sklearn, sentencepiece, seaborn; print(transformers.__version__)"

python .\scripts\create_eval_split.py

python run_bert_baseline.py --data-mode test
python run_flan_t5_baseline.py --data-mode test

python run_bert_baseline.py --data-mode test --max-samples 1000
python run_flan_t5_baseline.py --data-mode test --max-samples 200

python run_bert_baseline.py --data-mode full
python run_flan_t5_baseline.py --data-mode full

