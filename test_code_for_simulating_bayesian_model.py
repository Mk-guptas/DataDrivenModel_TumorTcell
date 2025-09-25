# checking the above modle by simualting if it works weell with some parameter
from src.mylib import *

def simulation_of_above():
    initial=100
    x= np.linspace(0,1,5); mu_x= np.asarray([0.0005,0,0,0,-0.006])
    t_eval = np.linspace(0, 1, 29001)  # Time points for simulation
    model_name=sequential_bayesian_sensing_model
    params={'beta':0.05 , 'v0':0.0001,'regression_params':[50,5,0,1],\
                'polynomial_order':0,'x_order':0,\
                'mean_scaler':0,'std_scaler':0,\
                'x':x ,'mu_x': mu_x,'tcell_population':3*initial,\
                'scaling':True,}

    
    # predicting curve based on the parameteer
    y0_initial_1= np.exp(-10*x);
    y0_initial_1=y0_initial_1/np.sum(y0_initial_1); #print('nomralization check', np.sum(y0_initial_1))
    y0_initial_1=np.concatenate((y0_initial_1, [initial]))

    if False:
        def wrapped_ivp(t_eval, y):
            return model_name(y, t_eval,params)
    
        predicted_data_1= solve_ivp(wrapped_ivp, t_span=(0, 1),y0=y0_initial_1, t_eval=t_eval, \
                                    method="Radau")
        sol= predicted_data_1.y.T
    else:
        
        sol= scipy_odeint(model_name, y0_initial_1, t_eval,args=(params,),atol=1e-8, rtol=1e-8, mxstep=100000,hmax=0.1/2900)

    fig,ax=plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(t_eval,sol[:,0],label='0') ; ax[0].plot(t_eval,sol[:,1],label='1'); ax[0].plot(t_eval,sol[:,2],label='2');ax[0].legend()
    ax[1].plot(t_eval,sol[:,-1])
    plt.show()

#simulation_of_above()