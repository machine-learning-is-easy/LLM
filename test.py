# generate fine tuning a model function, loading a model from hugging face hub, and prunning the model
# and save the model

def pruning(model):
    model = model.module
    model = model.cpu()
    model.prune_heads()
    model = model.cuda()
    return model

