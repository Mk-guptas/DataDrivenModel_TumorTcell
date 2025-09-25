
from scipy.integrate import odeint as scipy_odeint
import numpy as np  
import matplotlib.pyplot as plt

def sequential_bayesian_sensing_model(P,t,params):

    # system knowledge related parameter : depends on how u define the phenotype #####################################################
    x=params.get('x',np.linspace(0,1,len(P)-1))
    mu_x=params.get('mu_x',None)

    # parameter for assumed distribution : To be fitted
    
    beta1=params.get('beta',None)
    v0=params.get('v0',0)
    # parameters and shapes
    polynomial_order=params.get('polynomial_order',None)
    x_order=params.get('x_order',None)
    no_of_gaussian=params.get('no_of_gaussian',2)
    regression_params=params.get('regression_params',None)

    #boolean variables
    linear_regression=params.get('linear_regression',False)
    scaling=params.get('scaling',True)

    # sensed variable:y
    tcell_population=params.get('tcell_population',None)
    mean_scaler=params.get('mean_scaler',np.ones(polynomial_order+1,))
    std_scaler=params.get('std_scaler',np.ones(polynomial_order+1,))

    environmental_variable= (tcell_population*(P[-1]-P[-2]*P[-1]))/(3*500*500)
    

    # using likelihood derived from the data or system knowledge: Here I assumed beta liklihood ##############################################

    q=np.maximum(P[:-1],1e-4);
    q=q/np.sum(q)

    #q=P[:-1]
    likelihood_r=np.empty(len(x)-1,);likelihood_l=np.empty(len(x)-1,)
    for idx in range(len(x)-1):
       
        #likelihood_h[idx]=0.5+regression_params[2]/(1+np.exp(-regression_params[0]*(((P[-1]-P[-2]*P[-1])*tcell_population)/(3*500*500)-regression_params[1])))
        env_var11= (((q[idx]*P[-1])/500)*(tcell_population/(3*500)))/std_scaler[1];
        #env_var33= ((P[idx]*P[-1])/500)**3*(tcell_population/(3*500))**3
        env_var21= ((q[idx]*P[-1])/500)**2*(tcell_population/(3*500))**1;
        env_var12= ((q[idx]*P[-1])/500)**1*(tcell_population/(3*500))**2;

        #assert env_var11<=1 and env_var12<=1 and env_var21, f'density more than 1'
        
        #likelihood_h[idx]=0.5+0.5/(1+np.exp(-regression_params[0]*(env_var11-regression_params[2]) -regression_params[1]*(np.log(env_var11+1e-6)-regression_params[3]*0)))
        #likelihood_r[idx]=max(2/(1+np.exp(regression_params[0]*(env_var11) +regression_params[1]*(env_var11)**2 +regression_params[2]*(env_var11)**3+regression_params[3]*(env_var11)**0.5+regression_params[3]*(env_var11)**0.7)),1e-1)
        likelihood_r[idx]=max(1-((q[idx]*P[-1])/500)**regression_params[0]*(tcell_population/(3*500))**regression_params[1],1e-1)
        #exponent_n=regression_params[0];alpha1=regression_params[1]
        #likelihood_r[idx]=max(1-env_var11**exponent_n/(alpha1+(1-alpha1)*env_var11**exponent_n),1e-3)
        

        
    #likelihood_l=max(0.5/(1+np.exp(regression_params[3]*(environmental_variable-regression_params[4]))),0.1)
    
    #print(likelihood_h)
    #assert ((likelihood_h >= 0.05) & (likelihood_h <= 1)).all() and ((likelihood_l >= 0.05) & (likelihood_l <= 1)).all(), "likelihood_h and likelihood_l must be in [0.1, 1]"
    assert ((likelihood_r >= 0) & (likelihood_r <= 1)).all(), "likelihood_h and likelihood_l must be in [0.1, 1]"
    
    # Simulation begins (COupled differential equations)
    eps=1e-5;eps1=1e-30
    if True:

        drift_term = np.zeros(len(P[:-1]),);
        da=0.25
        if v0 >= 0:
            # backward diff: dP/da ≈ (P[i] - P[i-1]) / da
            drift_term[0] = -v0 * (P[0] - 0) / da
            drift_term[1:-1] = -v0 * (P[1:-2] - P[0:-3]) / da
            drift_term[-1] = -v0 * (0 - P[-3]) / da
        else:
            # forward diff: dP/da ≈ (P[i+1] - P[i]) / da
            drift_term[0] = -v0 * (0-P[0]) / da
            drift_term[1:-1] = -v0 * (P[0:-3] - P[1:-2]) / da
            drift_term[-1] = -v0 * (P[-2]-0) / da
            
        
        if not np.all(np.isfinite(P)):
            print(f"[RHS] non-finite state at t={t}: {P}")
        
        dPdt=np.ones_like(P)
        #P[:-1]=np.where(P[:-1]<0,1e-10,P[:-1])
        # Below is the standarad Bayesian model : based on the above information #################################
        
        # average properties
        marginal_probablity_y=np.empty(len(x)-1,)
        for idx in range(len(x)-1):
            #mean_proliferation_rate[idx]=(mu_x[idx]*P[idx] +mu_x[idx+1]*P[idx+1])/2
            #marginal_probablity_y[idx]=likelihood_l[idx]*(P[idx]/(P[idx]+P[idx+1])) +likelihood_h[idx]*(P[idx+1]/(P[idx]+P[idx+1]))
            marginal_probablity_y[idx]=likelihood_r[idx]*(P[idx]/(P[idx]+P[idx+1])) +1*(P[idx+1]/(P[idx]+P[idx+1]))
            
        mean_proliferation_rate=np.sum(P[:-1]*mu_x)
        
        # this evolve the distirbution of phenotypes 

        dPdt[0]=(beta1*(likelihood_r[0]/max(marginal_probablity_y[0],eps) -1)*(P[0])             + (mu_x[0]-mean_proliferation_rate)*P[0]  +drift_term[0])*2900
        
      
        dPdt[1:-2]=(beta1*(likelihood_r[1:]/np.maximum(marginal_probablity_y[1:],eps) -1)*(P[1:-2])\
            + beta1*(1/np.maximum(marginal_probablity_y[0:-1],eps) -1)*(P[1:-2])\
            + (mu_x[1:-1]-mean_proliferation_rate)*P[1:-2]   +    drift_term[1:-1])*2900

        
        dPdt[-2]=(beta1*(1/max(marginal_probablity_y[-1],eps) -1)*(P[-2])   + (mu_x[-1]-mean_proliferation_rate)*P[-2]  + drift_term[-1])*2900
        
        # this is population growth dynamics: logistic
        dPdt[-1] = (mean_proliferation_rate*(P[-1])*(1-P[-1]/500))*2900

        
        # Projection: keep sum(P[:n]) constant (≈ 1) to avoid drift
        sum_rate = dPdt[:-1].sum()
        # distribute the correction in proportion to q (replicator-style)
        dPdt[:-1] -= sum_rate * q

        
        # Detect if after the update porbablity is still normalized
        if np.abs(np.sum(P[:-1])-1) >1e-1:
            print(f'tot sum prob {np.sum(P[:-1]),P}')
        if np.any(P[:-1] <0):
            print(f'probab less than 0 {P}')
        if np.any(P[:-1] > 1+1e-3):
            print(f'probab more than 1 {P}')
        if mean_proliferation_rate >1e-2:
           print(f'mean prlif{mean_proliferation_rate}')
        if P[-1] >1e3:
           print(P[-1])
        if not np.all(np.isfinite(P)):
            print(f"[RHS] non-finite state at t={t}: {P}")
    
    return dPdt


