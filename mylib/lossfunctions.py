import numpy as np
from scipy.integrate import solve_ivp, odeint as scipy_odeint


def normalized_feature_in_linear_regression(training_data_set,polynomial_order=1):

    # standardization of feautres
    feature_basic= (training_data_set*training_data_set[:,0].reshape(len(training_data_set),1)*3)/(500*500*3)
    feature_basic_list=feature_basic.flatten()
    powers=np.arange(1,polynomial_order+1)
    features_powers=feature_basic_list[:,None]**powers
    mean_scaler=np.concatenate(([0.0],np.mean(features_powers,axis=0)))
    std_scaler=np.concatenate(([1.0],np.std(features_powers,axis=0)))
    
    return mean_scaler,std_scaler



def prediction_error_combined(p0,training_data_set, experimental_time,t,model_name,x,mu_x,lambdaaa,mean_scaler,std_scaler,likelihood_object,residual_hist=[]):
    
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions  
    no_of_dataset=np.shape(training_data_set)[0]
    residual=np.ones((no_of_dataset,len(experimental_time)))
    
    likelihood_object.update_coeffs(p0[2:])

    #start solving the ode
    for training_data_idx, training_data_value in enumerate(training_data_set):
        
        params={'beta':p0[0] ,'v0':1*p0[1],\
                'likelihood_function':likelihood_object,\
                'mean_scaler':mean_scaler,'std_scaler':std_scaler,\
                'x':x ,'mu_x': mu_x,'tcell_population':training_data_value[0]*3,\
                'scaling':True,}
    
        # predicting curve based on the parameteer
        y0_initial_1= np.exp(-10*x);
        y0_initial_1=y0_initial_1/np.sum(y0_initial_1); #print('nomralization check', np.sum(y0_initial_1))
        y0_initial_1=np.concatenate((y0_initial_1, [training_data_value[0]]))

        if False:
            def wrapped_ivp(t, y):
                return model_name(y, t,params)
        
            predicted_data_1= solve_ivp(wrapped_ivp, t_span=(0, 1),y0=y0_initial_1, t_eval=t, \
                                        method="RK45")
            predicted_data_1= predicted_data_1.y.T
        else:
            
            predicted_data_1= scipy_odeint(model_name, y0_initial_1, t,args=(params,),mxstep=10**9, rtol=1e-8, atol=1e-8,hmax=1/2900)

        assert np.shape(predicted_data_1)[0]==2901,f'time shape is {np.shape(predicted_data_1)[0]}'
        predicted_tumor_1=predicted_data_1[experimental_time,-1]
        
        # taking only the weigh not interecept to penelize

        #residual[training_data_idx]=(predicted_tumor_1-training_data_value)/training_data_value[0]
        residual[training_data_idx]=(np.log(predicted_tumor_1+1e-5)-np.log(training_data_value+1e-5))
    #weights= p0[1:].reshape(len(x),polynomial_order+1) ;sliced_weighs=weights[:,1:].flatten()
    #weights= p0[1:].reshape(2,x_order+1) ;sliced_weighs=weights[:,1:].flatten()
    #residual=np.concatenate(([lambdaaa*np.sum(sliced_weighs**2)],residual.flatten()))

    # detect if residual is finite
    bad = ~np.isfinite(residual.flatten())   # True where arr is NaN or Inf
    if bad.any():
        print("Found NaN or Inf at indices:", np.where(bad)[0])
    residual_hist.append(np.sum((residual.flatten())**2))
    
    return (residual.flatten())


def actual_validation_training_error(p0,training_data_set,experimental_time,t,model_name,x,mu_x,mean_scaler,std_scaler,likelihood_object,test_data=None,purpose='Training_loss'):
    # solving ODE using odeint for a given set of parameter ;Note: We will solve for two different initial conditions  

    likelihood_object.update_coeffs(p0[2:])
    if purpose=='Training_loss':
        data_set=training_data_set
    else:
        data_set=test_data
    
    no_of_dataset=np.shape(data_set)[0]
    residual=np.ones((no_of_dataset,len(experimental_time)))
    for data_idx, data_value in enumerate(data_set):
        params={'beta':p0[0] , 'v0':p0[1],\
                'likelihood_function':likelihood_object,\
                'mean_scaler':mean_scaler,'std_scaler':std_scaler,\
                'x':x ,'mu_x': mu_x,'tcell_population':data_value[0]*3,\
                'scaling':True,}
    
        # predicting curve based on the parameteer
        y0_initial_1= np.exp(-5*x);
        y0_initial_1=y0_initial_1/np.sum(y0_initial_1); #print('nomralization check', np.sum(y0_initial_1))
        y0_initial_1=np.concatenate((y0_initial_1, [data_value[0]]))
        
        predicted_data_1= scipy_odeint(model_name, y0_initial_1, t,args=(params,))
        predicted_tumor_1=predicted_data_1[experimental_time,-1]
        residual[data_idx]=(predicted_tumor_1-data_value)
        avg_loss=np.sum(residual**2)/no_of_dataset

    return avg_loss
