"""Train one repository model on one prepared dataset, with explicit inputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {
    'ResNet': ('ResNet/train_resnet_pls.py','ResNet1D'),
    'ResNetAE': ('ResNet/train_resnet_ae_pls.py','ResNetAutoencoder1D'),
    'MobileNet': ('MobileNet/train_mobilenet_pls.py','MobileNetV2_1D'),
    'SqueezeNet': ('SqueezeNet/train_squeezenet_pls.py','SqueezeNet1D'),
    'PointNet': ('PointNet/train_pointnet.py','PointNetDual'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def load_data(train_path,test_path,kind):
    train,test = pd.read_csv(train_path),pd.read_csv(test_path)
    prefix = 'PLS_' if kind=='latent' else 'Coef_' if kind=='coef' else None
    columns = ([c for c in train.columns if c.startswith(prefix)] if prefix else
               [f'{axis}_{point}' for point in range(1002) for axis in ('x','y','z')])
    expected = 8 if kind=='latent' else 507 if kind=='coef' else 3006
    if len(columns)!=expected or any(c not in test.columns for c in columns):
        raise ValueError('Unexpected feature schema')
    for frame in (train,test):
        if not frame.Subject.is_unique or set(frame.BinaryClass.unique())!={0,1}:
            raise ValueError('Require unique Subject values and both binary classes')
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError('Non-finite features')
    # Original person IDs and synthetic parents must never overlap held-out people.
    import sys
    sys.path.insert(0,str(ROOT/'Data_Preparation'))
    from data_prep_common import patient_group_id
    original = train if 'DataType' not in train else train[train.DataType=='Original']
    test_ids = {patient_group_id(s) for s in test.Subject}
    if test_ids & {patient_group_id(s) for s in original.Subject}:
        raise ValueError('Train/test person overlap')
    if 'DataType' in test and not test.DataType.eq('Original').all():
        raise ValueError('Test contains synthetic rows')
    return train,test,columns


def transform_features(train,test,columns,kind,components):
    x = train[columns].to_numpy(dtype=np.float64)
    xt = test[columns].to_numpy(dtype=np.float64)
    if kind=='xyz':
        return x.astype(np.float32).reshape(-1,1002,3).transpose(0,2,1), \
               xt.astype(np.float32).reshape(-1,1002,3).transpose(0,2,1), {'kind':kind,'columns':columns}
    scaler = StandardScaler().fit(x)
    x,xt = scaler.transform(x),scaler.transform(xt)
    bundle = {'kind':kind,'columns':columns,'scaler':scaler}
    if kind=='coef':
        effective = min(components,x.shape[1],len(x)-1)
        pls = PLSRegression(n_components=effective,scale=False,max_iter=1000)
        # Same binary-target PLS convention as the repository *_pls training scripts.
        pls.fit(x,train.BinaryClass.to_numpy())
        x,xt = pls.transform(x),pls.transform(xt)
        bundle.update(pls=pls,effective_components=effective)
    return x.astype(np.float32),xt.astype(np.float32),bundle


def model_class(name,side):
    """Load existing architecture definitions without their legacy file/log IO."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    relative,class_name = ARCHITECTURES[name]
    path = ROOT/'Model/All_Augment_tain'/side/relative
    tree = ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node,ast.ClassDef)]
    namespace = {'torch':torch,'nn':nn,'F':F,'np':np,'__name__':f'dataset_arch_{name}_{side}'}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace[class_name],path


def safe_batches(size,batch_size,rng):
    indices = rng.permutation(size)
    batches = [indices[start:start+batch_size] for start in range(0,size,batch_size)]
    if len(batches)>1 and len(batches[-1])==1:
        last = batches.pop()
        batches[-1] = np.concatenate([batches[-1],last])
    return batches


def train_torch(args,x,y,xt):
    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls,source = model_class(args.model,args.side)
    model = cls(seq_length=x.shape[1]) if args.model=='ResNetAE' else cls()
    model.to(device)
    pointnet = args.model=='PointNet'
    train_x = torch.tensor(x if pointnet else x[:,None,:],device=device)
    test_x = torch.tensor(xt if pointnet else xt[:,None,:],device=device)
    train_y = torch.tensor(y,dtype=torch.long if pointnet else torch.float32,device=device)
    if not pointnet:
        train_y = train_y[:,None]
    counts = np.bincount(y,minlength=2)
    if pointnet:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(2*counts),dtype=torch.float32,device=device))
    elif args.model=='SqueezeNet':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0]/counts[1]],dtype=torch.float32,device=device))
    else:
        criterion = nn.BCELoss()
    optimizer_cls = torch.optim.AdamW if args.model in ('PointNet','SqueezeNet') else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(),lr=0.001,
                              **({'weight_decay':1e-3} if args.model in ('PointNet','SqueezeNet') else {}))
    history = []
    rng = np.random.default_rng(args.seed)
    print(f'Device={device}; epochs={args.epochs}; batch={args.batch_size}; model={cls.__name__}',flush=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss = 0.0
        start = time.monotonic()
        for indices in safe_batches(len(y),args.batch_size,rng):
            bx,by = train_x[indices],train_y[indices]
            optimizer.zero_grad(set_to_none=True)
            output = model(bx)
            if args.model=='ResNetAE':
                prediction,reconstruction = output
                loss = criterion(prediction,by)+0.5*nn.functional.mse_loss(reconstruction,bx)
            else:
                loss = criterion(output,by)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        history.append({'epoch':epoch,'loss':total_loss/len(y),'seconds':time.monotonic()-start})
        print(f"epoch={epoch}/{args.epochs} loss={history[-1]['loss']:.6f} seconds={history[-1]['seconds']:.2f}",flush=True)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0,len(xt),args.batch_size):
            output = model(test_x[start:start+args.batch_size])
            if args.model=='ResNetAE':
                output = output[0]
            probability = (torch.softmax(output,dim=1)[:,1] if pointnet else
                           torch.sigmoid(output).flatten() if args.model=='SqueezeNet' else output.flatten())
            probabilities.extend(probability.cpu().numpy())
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'architecture':cls.__name__,'input_features':x.shape[1:] if pointnet else x.shape[1],
                'source':str(source),'source_sha256':sha(source),'seed':args.seed},args.output_dir/'model.pt')
    return np.asarray(probabilities),history,{'device':str(device),'torch_version':torch.__version__,
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'architecture_source':str(source),'architecture_sha256':sha(source)}


def metrics(y,probability,prediction):
    return dict(accuracy=float(accuracy_score(y,prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y,prediction)),
                sensitivity=float(recall_score(y,prediction,pos_label=1,zero_division=0)),
                specificity=float(recall_score(y,prediction,pos_label=0,zero_division=0)),
                f1_macro=float(f1_score(y,prediction,average='macro',zero_division=0)),
                roc_auc=float(roc_auc_score(y,probability)),
                confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist(),
                classification_report=classification_report(y,prediction,output_dict=True,zero_division=0))


def plots(root,result,y,probability,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay,ConfusionMatrixDisplay
    fig,ax = plt.subplots()
    ConfusionMatrixDisplay(np.array(result['confusion_matrix']),display_labels=['Class 0','Class 1']).plot(ax=ax)
    fig.tight_layout();fig.savefig(root/'confusion_matrix.png');plt.close(fig)
    fig,ax = plt.subplots()
    RocCurveDisplay.from_predictions(y,probability,ax=ax)
    fig.tight_layout();fig.savefig(root/'roc_curve.png');plt.close(fig)
    if history:
        fig,ax = plt.subplots()
        ax.plot([row['epoch'] for row in history],[row['loss'] for row in history])
        ax.set(xlabel='Epoch / iteration',ylabel='Training loss')
        fig.tight_layout();fig.savefig(root/'training_loss.png');plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train one dataset/model with 10-fold OOF and final Test evaluation')
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--kind', choices=['coef', 'latent', 'xyz'], required=True)
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--model', choices=['SVM', 'MLP', 'PLSDA']+list(ARCHITECTURES), required=True)
    parser.add_argument('--folds', type=int, choices=[10], default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pls-components', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tune-threshold', action='store_true', help='Select accuracy threshold on Train OOF; freeze before final Test')
    parser.add_argument('--early-stopping', action='store_true', help='Use separate inner stopping people; restore best checkpoint before outer OOF')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--stop-fraction', type=float, default=0.2)
    parser.add_argument('--mlp-max-epochs', type=int, default=None, help='Defaults to the same maximum as --epochs')
    args = parser.parse_args()
    if min(args.epochs, args.pls_components, args.batch_size, args.threads) < 1:
        parser.error('Training settings must be positive')
    if args.patience < 1 or args.mlp_max_epochs < 1 or args.min_delta < 0 or not 0 < args.stop_fraction < 1:
        parser.error('Invalid early stopping settings')
    from dataset_cross_validation import run
    run(args)


if __name__ == '__main__':
    main()
