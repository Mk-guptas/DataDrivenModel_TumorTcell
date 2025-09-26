# This is resdiual function which will be optimized ##########################################################
from mylib import *



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



# actual validation error
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



## Training the model :  Here we use least square method to optimize the parameter of the model using the given training data in hand and given learning parameters
def multi_start_combined_fitting(x,mu_x,t, model_name,experimental_time,training_data_set, mean_scaler,std_scaler,likelihood_object,lambdaaa=0,number_of_starts=200):
    # giving argument to fit the objective function
    residuals_func=prediction_error_combined
    residual_argument=(training_data_set, experimental_time,t, model_name,x,mu_x,lambdaaa,mean_scaler,std_scaler,likelihood_object)

    # CONSTRAIN ON THE PARAMETERS
    # ----- Config -----
    degree_likelihood_object= likelihood_object.degree
    @dataclass
    class Config:
        dim: int =2+degree_likelihood_object
        bounds: Tuple[Tuple[float, float], ...] = (
            (0.0001, 0.1),   # beta1
            (0,1e-8),# v0
            *((0,50.0),)*degree_likelihood_object 
        )
        n_starts: int =number_of_starts
        seed: int = 7
    cfg = Config()

    ## using MULTI START local optimization using least square ###########
    best_res_fit,results_of_all=multistart_least_squares(residuals_func,residual_argument,cfg)
    #best_res_fit,results_of_all=multistart_least_squares_With_logger_option(residuals_func,residual_argument,cfg)
    #best_res_fit, succ3, results_of_all=multistart_least_squares_halving(residuals_func,residual_argument,cfg)
    ## result and visualization
    p_best = best_res_fit.x
    best_ssq = np.sum(best_res_fit.fun**2)
    print(f'best ssq {best_ssq} for degree {likelihood_object.degree} and value of lambda {lambdaaa}')

    #cov_matrix, corr_matrix, std_dev,residual_variance=compute_covariance_and_correlation(best_res_fit,residuals_func,residual_argument)
    #print(f'standarad deviation {std_dev}')
    
    return results_of_all,best_res_fit,best_ssq,p_best



## script FIT THE DATA ##########################################################################################################

tumor_cell_population= np.load(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/2025-09-26/tumor_cell_population.npy')
with open(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/2025-09-26/information_dict.pkl','rb') as f:
    information_dict= pickle.load(f)
initial_tumor_size_list= information_dict['initial_tumor_size_list'], 
experimental_time= information_dict['experimental_time']

# define the training and validation data set
training_data_set=tumor_cell_population[[3,15]]
#training_data_set=tumor_cell_population[[3,5,6,7,8,9,10,11,12,13,15]]
validation_data_set=tumor_cell_population[[1,3,7,16]]
print(f'training data set {training_data_set.shape} and validation data set {validation_data_set.shape}')

# define initial conditions and simulation time
x= np.linspace(0,1,2); mu_x= np.asarray([0.0005,-0.006])
#x=np.asarray([0.,0.5,1]);mu_x=np.asarray([0.0005,0,-0.006])
t_eval = np.linspace(0, 1, 2901)  # Time points for simulation
model_name=sequential_bayesian_sensing_model


#select the model for likelihood function
likelihood_object=SigmoidPolynomial(degree=2, coeff_min=0.1, coeff_max=1.0)

# standardization of features
mean_scaler,std_scaler=normalized_feature_in_linear_regression(training_data_set,polynomial_order=1)

#fitting the model using multi start least square
results_of_all,best_res_fit,best_ssq,p_best= multi_start_combined_fitting(x,mu_x,\
                                                                          t_eval, model_name,experimental_time, training_data_set, mean_scaler,std_scaler,\
                                                                        likelihood_object,number_of_starts=4)


print(f'best ssq {best_ssq} for degree {likelihood_object.degree}')



training_loss=actual_validation_training_error(p_best,training_data_set, experimental_time,t_eval,model_name,x,mu_x,\
                                               mean_scaler,std_scaler,likelihood_object,test_data=None,purpose='Training_loss')
print(f'actual training loss {training_loss}')

test_loss=actual_validation_training_error(p_best,training_data_set, experimental_time,t_eval,model_name,x,mu_x,\
                                          mean_scaler,std_scaler,likelihood_object,test_data=validation_data_set,purpose='Test_loss')
print(f'actual test loss {test_loss}')


    

print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq) 
print(f'total number parameter fitted {len(p_best)}')
        
# Now we will focus on how well we can idenitify the parameters
#cov_matrix, corr_matrix, std_dev,residual_var = compute_covariance_and_correlation(res_fit,  residuals_func,residual_argument)
#print("Standard deviations:", std_dev)
#print("Correlation matrix:\n", corr_matrix)


with open (path_mechanistic_tcell_tumor_data_for_fitting_BL+'/p_best_with_poly_order_'+str(1)+'.pkl','wb') as f:
    pkl.dump(p_best,f)

# Below I want to visualize the same resulsts above in graph form
if True:    

    # Comparing the training plot after fittting
    fig,ax=plt.subplots(1,2,figsize=(15,3))
    
    for data_idx, data_set in enumerate(training_data_set):
        
        params={'beta':best_res_fit.x[0] ,'v0':best_res_fit.x[1] ,'regression_params':best_res_fit.x[2:],\
                'likelihood_function':likelihood_object,\
                'mean_scaler':mean_scaler,'std_scaler':std_scaler,\
                'x':x ,'mu_x': mu_x,'tcell_population':data_set[0]*3,'scaling':True,}
        
        y0_initial_1= np.exp(-10*x);
        y0_initial_1=y0_initial_1/np.sum(y0_initial_1);
        y0_initial_1=np.concatenate((y0_initial_1, [data_set[0]]))
       
        predicted_dataset = scipy_odeint(model_name, y0_initial_1, t_eval,args=(params,))
        predicted_dataset=  predicted_dataset[:,-1]
    
        ax[0].scatter(experimental_time/60,data_set,color='red',label='Data 1',s=10)
        #ax[0].set_ylim(0,260)
        ax[0].plot(t_eval/60*2900, predicted_dataset[:],label='fit',color='black',linestyle='--')
        
        ax[0].legend()
    


    # checking the predcition on new data set
    for data_idx, data_set in enumerate(validation_data_set):
    
        params={'beta':best_res_fit.x[0] , 'v0':best_res_fit.x[1] ,'regression_params':best_res_fit.x[2:],\
                'likelihood_function':likelihood_object,\
                'mean_scaler':mean_scaler,'std_scaler':std_scaler,\
                'x':x ,'mu_x': mu_x,'tcell_population':data_set[0]*3,\
                'scaling':True,}
        y0_initial_1= np.exp(-10*x);
        y0_initial_1=y0_initial_1/np.sum(y0_initial_1);
        y0_initial_1=np.concatenate((y0_initial_1, [data_set[0]]))
        
        predicted_dataset = scipy_odeint(model_name, y0_initial_1, t_eval,args=(params,))
        predicted_dataset2=  predicted_dataset[:,-1]
        
        ax[1].scatter(experimental_time/60,data_set,color='red',label='Data 1',s=10)
        #ax[0].set_ylim(0,260)
        ax[1].plot(t_eval/60*2900, predicted_dataset2[:],label='fit',color='black',linestyle='--')
        
        ax[1].legend()
    
    plt.show()

# analyzing the learnng curve
if False:
    fig,ax=plt.subplots(1,2, figsize=(10,3))
    [ax[0].plot(np.arange(0,len(results_of_all['hist'][i])), results_of_all['hist'][i]) for i in range(50)]
    ax[0].set_yscale('log');ax[0].set_xscale('log')
    plt.show()