# AI4Bio 4-th place. 

Sample solution that helped me reach 4-th place on [AI4Bio track](https://dsworks.ru/en/champ/545de8cb-e023-4b64-9be4-b95b9519f8d4#overview) of [AIJourney competition](https://dsworks.ru/en/group/1a047e72-4791-45d0-8ee9-44ac9c43bf4b).

Key features:
1. [ResNet1D](https://github.com/csho33/bacteria-ID) as high-level feature extractor
2. [AMSoftmax](https://arxiv.org/abs/1801.05599) head as embeddings and logits solver
3. Pretraining on [Zenodo-v3 MALDI-TOF dataset](https://zenodo.org/record/1880975)
4. Strong augmentations
5. Precise threshold searching for out-of-set data cut off


## Pretraining
```
python train.py --stage="pretraining"
```

## Finetuning
```
python train.py --stage="finetuning"
```

## Testing
```
python test.py --checkpoint="checkpoint_6.pth"
```

## Solution
In ```solution.py``` find ```run()``` and change arguments according to testing report created on previous step.