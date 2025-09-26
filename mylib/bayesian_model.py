
from scipy.integrate import odeint as scipy_odeint
import numpy as np  
import matplotlib.pyplot as plt



'''
This file contains the sequential Bayesian sensing model for tumor-T cell interaction
x:phenotype
mu_x: phenotype associated property (e.g., growth rate)
beta: decision making rate (inverse of time scale of decision making)
v0: drift in phenotype space (e.g., due to natural damage process)
P: state vector (P[:-1]: phenotype distribution, P[-1]: tumor size)
tcell_population: T cell population (environmental variable)
environmental_variable : to which the cells are responding (e.g., T cell density and tumor cell state population)
likelihood function: it can be any function or neural net also
'''
def sequential_bayesian_sensing_model(P,t,params):

    # system knowledge related parameter : depends on how we define the phenotype #####################################################
    x=params.get('x',np.linspace(0,1,len(P)-1))  
    mu_x=params.get('mu_x',None) 
    tcell_population=params.get('tcell_population',None)

    # parameter for assumed likelihood function and others : To be fitted
    beta1=params.get('beta',None)              
    v0=params.get('v0',0)                  
   
   # likelihood function
    likelihood_function=params.get('likelihood_function',None)
    
    # sensed variable y : standardization parameters for environmental variable
    mean_scaler=params.get('mean_scaler',None)
    std_scaler=params.get('std_scaler',None)


    q=np.maximum(P[:-1],1e-4);
    q=q/np.sum(q)


    likelihood_r=np.empty(len(x)-1,);
    for idx in range(len(x)-1):
        env_var11= (((q[idx]*P[-1])/500)*(tcell_population/(3*500)));
        #assert env_var11<=1 and env_var12<=1 and env_var21, f'density more than 1'

        likelihood_r[idx]=max(likelihood_function.evaluate(env_var11),1e-1)
        

    assert ((likelihood_r >= 0) & (likelihood_r <= 1)).all(), "likelihood_r must be in [0.1, 1]"
    
    # Simulation begins (COupled differential equations)
    eps=1e-5;eps1=1e-30

    drift_term=drift_computer(P,v0,da=x[1]-x[0])

    dPdt=np.ones_like(P)
    # Below is the standarad Bayesian model : based on the above information #################################
    
    # average properties
    marginal_probablity_y=np.empty(len(x)-1,)
    for idx in range(len(x)-1):
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

    if not np.all(np.isfinite(P)):
        print(f"[RHS] non-finite state at t={t}: {P}")

    return dPdt


# drift term computation
def drift_computer(P,v0,da=1):

    drift_term = np.zeros(len(P[:-1]),);
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

    return drift_term