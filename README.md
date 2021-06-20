# mfgan
Implementation of MFGAN: Sequential Recommendation with Self-Attentive Multi-Adversarial Network

This is very unclean draft. I will publish something beter in a few days!

article: https://arxiv.org/abs/2005.10602

Description:

This is recommender system, GAN based. It predicts next item user would be interested in.

Discriminator judges discriminator suggestions based on factors associated with item such as price, sentiments, etc. and sends feedback signal to improve generator.

Training sequence is in the article:

- pretrain generator on MLE: works
- generate samples with this generator: works
- pretrain discriminator on binary cross entropy: seems works
- train generator and discriminator in a loop: didn't try yet

TO-DO and notices:

- add item id as one of the factors
- figure out how to get factors in the tensorflow when next item is predicted. It is not available from input data. I think to load all item / product table with each record and lookup in it to provide factors to discriminator
- I don't get it why input sequence should be right aligned, why not to mask if input sequence is short
- input must be core 5. This code , I know, some few changes needed to product table, for example
- objective function equation needs to be validated

Otherwise, if you like, just see training.py and model.py
