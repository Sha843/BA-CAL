import collector
import numpy as np

def prototype_get(model,train_loader,args):
    collector.collect(model, train_loader, args)
    feature = np.load(args.feature_dict_path, allow_pickle=True).item()
    prototype = collector.get_queue(feature, args)
    np.save('', prototype)

