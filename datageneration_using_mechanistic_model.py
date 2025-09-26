
from mylib import *


if not os.path.exists(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/'+str(datetime.date.today())):
    os.makedirs(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/'+str(datetime.date.today()))
path2= path_mechanistic_tcell_tumor_data_for_fitting_BL+'/'+str(datetime.date.today())

def synthetic_data_generation(model_type,information_dict):




    params=information_dict['params']; nhits=params['nhits']
    
    for initial_idx, initial_tumor_size in enumerate(information_dict['initial_tumor_size_list']):
        
        initial_conditions =np.zeros((nhits+1)*2,);initial_conditions[0]=initial_tumor_size ; initial_conditions[-1]=3*initial_tumor_size
        model_outputs = scipy_odeint(model_type, initial_conditions, information_dict['t'], args=(information_dict['params'],))

        if False:
            def wrapped_ivp(t, y):
                return model_type(y ,t,information_dict['params'],)
        
            model_outputs= solve_ivp(wrapped_ivp, t_span=(0, 2900),y0=initial_conditions,t_eval=information_dict['t'], \
                                        method="Rk45")
            model_outputs=model_outputs.y.T
        tumor_cell_population[initial_idx]= np.sum(model_outputs[information_dict['experimental_time'],:-1],axis=1)
        
    return tumor_cell_population



information_dict={'initial_tumor_size_list': np.arange(25,700,25),'experimental_time': np.arange(0,2500,100),'params':{'k1':1e-5 ,'k3':8e-3,'k2':2e-3,'u0':5e-4,'d':0.006,'nhits':1},\
                  't':np.linspace(0,2900,2901)}



tumor_cell_population=np.ones((len(information_dict['initial_tumor_size_list']),len(information_dict['experimental_time'])))

tumor_cell_population= synthetic_data_generation(model2,information_dict)

np.save(path2+'/tumor_cell_population.npy', tumor_cell_population)

with open(path2+'/information_dict.pkl','wb') as f:
    pkl.dump(information_dict,f)


# quick visulaization
fig,ax=plt.subplots(1,4,figsize=(20,3))
[ax[0].plot(information_dict['experimental_time'], tumor_cell_population[i],label=str(i)) for i in range(len(information_dict['initial_tumor_size_list']))]
ax[0].legend();
plt.show()
