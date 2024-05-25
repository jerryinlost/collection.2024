from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")