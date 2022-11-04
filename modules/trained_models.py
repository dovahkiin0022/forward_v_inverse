import torch
import numpy as np
import pickle 
uts_model_loc = 'saved_models/uts_model.pt'
model = torch.jit.load(uts_model_loc, map_location='cpu')
with open('misc/scaler_y.pkl','rb') as fid:
    uts_scaler = pickle.load(fid)
with open('misc/scaler_x.pkl','rb') as fid:
    x_scaler = pickle.load(fid)


def get_uts_without_grain(comps,model = model,scaler_x = x_scaler, scaler_y = uts_scaler, t = 1473,p = 0, ph = [0,1,0]):
    temp = np.array([t]).reshape(-1,1)
    process = np.array([p]).reshape(-1,1)
    phase = np.array(ph).reshape(-1,3)
    X = np.concatenate((comps,temp,process,phase),axis=1)
    scaled_x = torch.from_numpy(scaler_x.transform(X).astype('float32'))
    y_pred = scaler_y.inverse_transform(model(scaled_x).to('cpu').detach().numpy())
    return y_pred