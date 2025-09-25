#Assumptions:  1) hit delivery does not cause the detachment 
#Assumptions:  2) only healthy tumor cell can porliferate both in free and complex state C[0] and C[0]T
#Assumptions: 3)  No recovery from damage to healthy state
#Assumptions:  4) No T-cell population change : here C[-1] shows the dynamicso fpopulation of free T-cell
#Assumptions: 5)  Existence of doomed state of tumor cell
# Assumptions: 6)
import numpy as np

def model2(C, t,params):
    # C=[C0,C1,C2,C3 ..., C0.T,C1.T,C2.T ..., T]

    k1=params.get('k1',1e-5) ;k3=params.get('k3',1e-3);k2=params.get('k2',2e-3);u0=params.get('u0',5e-4);d=params.get('d',0.006);nhits=params.get('nhits',4)

    derivs=[];
    Kmax=500; totalcell=np.sum(C[0:-1])  #logistic growth measures
    
    v=nhits+1; # complex indexing
    
    for hit_idx in range(nhits+1):
        if hit_idx==0:
            derivs.append(u0*(C[hit_idx]+C[(nhits+1)+hit_idx])*(1-totalcell/Kmax)   -(k1)*(C[hit_idx])*C[-1]    +     k2*C[(nhits+1)+hit_idx])
        elif(hit_idx >0 and hit_idx < nhits):
            derivs.append(0*(C[hit_idx]+C[(nhits+1)+hit_idx])*(1-totalcell/Kmax)    -(k1)*(C[hit_idx])*C[-1]   +     k2*C[(nhits+1)+hit_idx])
        elif hit_idx==nhits:
            derivs.append( k3*(C[-2])-d*C[hit_idx])
            
    for complex_id in range(nhits):
        if complex_id==0:
            derivs.append(  (k1)*(C[complex_id])*C[-1]   -  k2*C[(nhits+1)+complex_id]  -   k3*C[(nhits+1)+complex_id])  
        elif complex_id<nhits:
            derivs.append(  (k1)*(C[complex_id])*C[-1]   -  k2*C[(nhits+1)+complex_id]          -k3*C[(nhits+1)+complex_id]            +k3*C[(nhits+1)+complex_id-1])

            
    derivs.append(-k1*C[-1]*np.sum(C[0:nhits]) +k2*np.sum(C[nhits+1:-1]) +k3*(C[-2]))
    
    return np.array(derivs)



def test_model(y,t,params):
    r=params.get('r',0.0005) ;d=params.get('d',5e-5);
    return r*y*(1-y/500) -d*y
    

