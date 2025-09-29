from mylib import *
import numpy as np
from scipy.integrate import solve_ivp, odeint as scipy_odeint
import matplotlib.pyplot as plt


# data loading
tumor_cell_population= np.load(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/2025-09-26/tumor_cell_population.npy')
with open(path_mechanistic_tcell_tumor_data_for_fitting_BL+'/2025-09-26/information_dict.pkl','rb') as f:
    information_dict= pickle.load(f)
initial_tumor_size_list= information_dict['initial_tumor_size_list'], 
experimental_time= information_dict['experimental_time']

# define the training and validation data set
training_data_set=tumor_cell_population[[3,5,8,15]]
#training_data_set=tumor_cell_population[[3,5,6,7,8,9,10,11,12,13,15]]
validation_data_set=tumor_cell_population[[1,3,7,16]]
print(f'training data set {training_data_set.shape} and validation data set {validation_data_set.shape}')


# define initial conditions and simulation time
x= np.linspace(0,1,2); mu_x= np.asarray([0.0005,-0.006])
#x=np.asarray([0.,0.5,1]);mu_x=np.asarray([0.0005,0,-0.006])
t_eval = np.linspace(0, 1, 2901)  # Time points for simulation
model_name=sequential_bayesian_sensing_model_classifier

#select the model for likelihood function
#likelihood_object=SigmoidPolynomial(degree=2, coeff_min=0.1, coeff_max=1.0)
likelihood_object=InteractionPolynomialSigmoid(degree=2, seed=42)

# standardization of features
mean_scaler,std_scaler=normalized_feature_in_linear_regression(training_data_set,polynomial_order=1)


# reuglarization parameter
lambdaaa=0.0
number_of_starts=40

# giving argument to fit the objective function
residuals_func=prediction_error_combined
residual_argument=(training_data_set, experimental_time,t_eval, model_name,x,mu_x,lambdaaa,mean_scaler,std_scaler,likelihood_object)

# CONSTRAIN ON THE PARAMETERS
# ----- Config -----
degree_likelihood_object= likelihood_object.num_terms() # minus one because we do not fit the baseline constant term
print(f'number of terms in likelihood function {degree_likelihood_object}')
@dataclass
class Config:
    dim: int =2+degree_likelihood_object
    bounds: Tuple[Tuple[float, float], ...] = (
        (0.0001, 0.1),   # beta1
        (0,1e-8),# v0
        *((0,10.0),)*degree_likelihood_object 
    )
    n_starts: int =number_of_starts
    seed: int = 7
cfg = Config()

## using MULTI START local optimization using least square ###########
best_res_fit,results_of_all=multistart_least_squares(residuals_func,residual_argument,cfg,n_jobs=number_of_starts)


## result and visualization
p_best = best_res_fit.x
best_ssq = np.sum(best_res_fit.fun**2)
print(f'best ssq {best_ssq} for degree {likelihood_object.degree} and value of lambda {lambdaaa}')
print("Best-fit parameters:", p_best);print("Sum of squared residuals:", best_ssq)
print(f'total number parameter fitted {len(p_best)}')

training_loss=actual_validation_training_error(p_best,training_data_set, experimental_time,t_eval,model_name,x,mu_x,\
                                               mean_scaler,std_scaler,likelihood_object,test_data=None,purpose='Training_loss')
print(f'actual training loss {training_loss}')

test_loss=actual_validation_training_error(p_best,training_data_set, experimental_time,t_eval,model_name,x,mu_x,\
                                          mean_scaler,std_scaler,likelihood_object,test_data=validation_data_set,purpose='Test_loss')
print(f'actual test loss {test_loss}')


likelihood_object.update_coeffs(p_best[2:])
    
# Now we will focus on how well we can idenitify the parameters
#cov_matrix, corr_matrix, std_dev,residual_var = compute_covariance_and_correlation(res_fit,  residuals_func,residual_argument)
#print("Standard deviations:", std_dev)
#print("Correlation matrix:\n", corr_matrix)


# Below visualize the same resulsts above in graph form
if True:    

    # Comparing the training plot after fittting
    fig,ax=plt.subplots(1,2,figsize=(15,3))
    
    for data_idx, data_set in enumerate(training_data_set):
        
        params={'beta':best_res_fit.x[0] ,'v0':best_res_fit.x[1],\
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
    
        params={'beta':best_res_fit.x[0] , 'v0':best_res_fit.x[1] ,\
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