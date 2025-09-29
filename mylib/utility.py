import numpy as np 
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from itertools import combinations
from scipy.stats.qmc import Sobol


'''
this files contains the least square optimization technique from scipy library both single start and multistart
'''

def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def sobol_starts(cfg):
    eng = Sobol(d=cfg.dim, scramble=True, seed=cfg.seed)
    u = eng.random(cfg.n_starts)
    return scale01_to_bounds(u, cfg.bounds)

def scale01_to_bounds(u, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + u * (hi - lo)



def compute_covariance_and_correlation(res_fit,residual_func,extra_argument):
    J = res_fit.jac
    residual_variance = np.var(residual_func(res_fit.x,*extra_argument))
    cov_matrix = np.linalg.inv(J.T @ J) * residual_variance
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    return cov_matrix, corr_matrix, std_dev,residual_variance


## BELOW ARE THE FUNCTIONS FOR MULTISTART LEAST SQUARES
def least_square_fitting_algorithim_tuned_for_multistart(residuals_func,p0,bounds,residual_argument):

    res_fit = least_squares(residuals_func,p0, args=residual_argument,bounds=bounds,\
                            method="trf",loss="linear",xtol=1e-8, ftol=1e-8, gtol=1e-8,tr_solver="lsmr",tr_options={"regularize": True},)

    return res_fit


def multistart_least_squares(residuals_func,residual_argument,cfg,n_jobs=4):
    starts = sobol_starts(cfg)
    bounds=([b[0] for b in cfg.bounds], [b[1] for b in cfg.bounds])

    def _run_one_start(p0_sampled):
        p0_vec = np.asarray(p0_sampled).reshape(len(bounds[0]),)
        return least_square_fitting_algorithim_tuned_for_multistart(
            residuals_func,
            p0_vec,
            bounds,
            residual_argument
        )
    
    # Parallel with joblib (process-based via 'loky')
    if n_jobs and n_jobs != 1:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",        # robust process pool
            prefer="processes",
            verbose=10             # set 0 to silence
        )(delayed(_run_one_start)(p0) for p0 in starts)

    else:
        results = [least_square_fitting_algorithim_tuned_for_multistart(residuals_func,p0_sampled,bounds,residual_argument) for p0_sampled in starts]
        
    costs = np.array([r.cost for r in results])
    best_idx = int(np.argmin(costs))
    #print('best_resdiual_square',np.sum(results[best_idx].fun**2))
    
    return results[best_idx],results




# not in use currently
def multistart_least_squares_With_logger_option(residuals_func,residual_argument,cfg,n_jobs=4):
    starts = sobol_starts(cfg)
    bounds=([b[0] for b in cfg.bounds], [b[1] for b in cfg.bounds])
    
    def _run_one_start(p0_sampled):
        residual_hist=[]
        residual_argument_with_hist=residual_argument+(residual_hist,)
        p0_vec = np.asarray(p0_sampled).reshape(len(bounds[0]),)
        return {'parameter':least_square_fitting_algorithim_tuned_for_multistart(
            residuals_func,
            p0_vec,
            bounds,
            residual_argument_with_hist
        ),'hist':residual_hist}
    
    # Parallel with joblib (process-based via 'loky')
    if n_jobs and n_jobs != 1:
        raw = Parallel(
            n_jobs=n_jobs,
            backend="loky",        # robust process pool
            prefer="processes",
            verbose=10             # set 0 to silence
        )(delayed(_run_one_start)(p0) for p0_idx,p0 in enumerate(starts))

    else:
        raw = [least_square_fitting_algorithim_tuned_for_multistart(residuals_func,p0_sampled,bounds,residual_argument) for p0_sampled in starts]

    results = {
    'parameter': [r['parameter'] for r in raw],
    'hist':      [r['hist'] for r in raw],}
    
    costs = np.array([r.cost for r in results['parameter']])
    best_idx = int(np.argmin(costs))
    #print('best_resdiual_square',np.sum(results[best_idx].fun**2))
    
    return results['parameter'][best_idx],results


def training_test_split_principled_split_for_better_coverage(data,ratio):

    n_traj, T = data.shape
 
    assert n_traj*ratio%1==0, f"Expected 10 trajectories; got {n_traj}"
    n_trai_traj=int(n_traj*ratio)
    
    # All combinations of choosing  m out of n_traj trajectories
    train_combos = list(combinations(range(n_traj),  n_trai_traj)) 
    C = len(train_combos)
    
    train_sets = np.empty((C,  n_trai_traj, T), dtype=data.dtype)
    test_sets  = np.empty((C,  n_traj-n_trai_traj, T), dtype=data.dtype)
    test_idx_list = []
    
    all_idx = set(range(n_traj))
    for k, tr_idx in enumerate(train_combos):
        tr_idx = np.array(tr_idx, dtype=int)
        te_idx = np.array(sorted(list(all_idx - set(tr_idx))), dtype=int)  # the 2 left out
        
        train_sets[k] = data[tr_idx]     # (8, T)
        test_sets[k]  = data[te_idx]     # (2, T)
        test_idx_list.append(tuple(te_idx.tolist()))
    
    return train_sets, test_sets, train_combos, test_idx_list


    