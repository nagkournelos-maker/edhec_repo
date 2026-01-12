import pandas as pd
import numpy as np
from scipy.optimize import minimize

def drawdown_compute(returns_series: pd.Series):
    """
    it takes a timeseries of asset returns computes and sends back a df with : wealth index , previous peaks , pct drawdown
    """
    wealth_index = 1000*(1+returns_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({'wealth index':wealth_index , 'previous peaks':previous_peaks , 'drawdowns':drawdowns})

def get_ffme_returns():
    import os
    # Construct the file path using os.path.join
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "data", "Portfolios_Formed_on_ME_monthly_EW.csv")

    # Read the CSV file into a DataFrame
    me_m = pd.read_csv(file_path , header=0 , index_col=0 , parse_dates =True , na_values=-99.99)
    rets = me_m[['Lo10' , 'Hi10']]
    rets.columns=['Smallcap','Largecap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index , format='%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    import os
    # Construct the file path using os.path.join
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "data", "edhec-hedgefundindices.csv")

    # Read the CSV file into a DataFrame
    hfi = pd.read_csv(file_path , header=0 , index_col=0 , parse_dates =True)
    hfi = hfi/100
    hfi.index = pd.to_datetime(hfi.index , format='%Y%m').to_period('M')
    return hfi

def skewness(r):
    '''computes the scipy.stats.skew  we use populattion std and not sample , so ddof=0'''
    demeaned_r = r - r.mean()
    sigma_r=r.std(ddof=0)
    expected_v = (demeaned_r**3).mean()
    sk = (expected_v) / (sigma_r)**3
    return sk

def kurtosis(r):
    '''computes the NOT EXCESS kurtosis , NOT SAME as scipy.stats.kurtosis'''
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    expected_v = (demeaned_r**4).mean()
    kurt = expected_v / (sigma_r**4)
    return kurt

def semideviation(r):
    '''the std of the negative returns , we input the whole set 
    (positive and negative) Series or df'''
    negative_r = r<0
    semidev = r[negative_r].std(ddof=0)
    return semidev

def var_historic(r,level=5):
    '''
    var historic reversion with dataframe and series , all revert to series
    '''
    if isinstance(r,pd.DataFrame):
        return pd.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError('expected r to be series or df')
        
def var_historic(r,level=5):
    '''
    var historic reversion with dataframe and series , all revert to series
    '''
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError('expected r to be series or df')
        
def var_gaussian_cf_modification(r, level=5, modification=False):
    from scipy.stats import norm
    '''the gaussian var , from scipy.stats import norm'''
    z = norm.ppf(level/100)
    if modification:
        sk = skewness(r)
        kurt = kurtosis(r)
        z = z + ((1/6)*((z**2) -1)*sk) + ((1/24)*((z**3)-(3*z))*(kurt-3)) - (1/36)*(2*(z**3)- (5*z))*(sk**2)
    var_gauss = - (r.mean() + z * r.std(ddof=0))
    
    return var_gauss

def cvar_historic(r,level=5):
    '''computes the historical C var , beyond the var'''
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r,level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DateFrame):
        return r.aggregate(cvar_historic,level)
    else:
        raise TypeError('expected r to be series or df in cvar')

def get_ind30_returns():
    '''load the index 30 french'''
    import os
    # Construct the file path using os.path.join
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "data", "ind30_m_vw_rets.csv")

    # Read the CSV file into a DataFrame
    ind = pd.read_csv(file_path , header=0 , index_col=0 , parse_dates =True)
    ind = ind/100
    ind.index = pd.to_datetime(ind.index , format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    ind_return = get_ind30_returns()
    ind_size = get_ind30_size()
    ind_nfirm = get_ind30_nfirms()
    ind_mktcap = ind_nfirm*ind_size
    total_mktcap = ind_mktcap.sum(axis='columns')
    ind_capweight = ind_mktcap.divide(total_mktcap , axis='rows')
    total_market_return = (ind_capweight*ind_return).sum(axis='columns')
    return total_market_return

def get_ind30_size():
    '''load the index size 30 french'''
    import os
    # Construct the file path using os.path.join
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "data", "ind30_m_size.csv")

    # Read the CSV file into a DataFrame
    ind = pd.read_csv(file_path , header=0 , index_col=0 , parse_dates =True)
    ind.index = pd.to_datetime(ind.index , format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind30_nfirms():
    '''load the index size 30 french'''
    import os
    # Construct the file path using os.path.join
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "data", "ind30_m_nfirms.csv")

    # Read the CSV file into a DataFrame
    ind = pd.read_csv(file_path , header=0 , index_col=0 , parse_dates =True)
    ind.index = pd.to_datetime(ind.index , format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def annualized_volatility(series,periods_per_year):
    vol = series.std()
    ann_vol = vol*(periods_per_year**0.5)
    return ann_vol

def annualized_return(series_of_returns,periods_per_year):
    no_periods = series_of_returns.shape[0]
    compounded = (1+series_of_returns).prod()
    ann_ret = (compounded**(periods_per_year/no_periods)) - 1
    return ann_ret

def sharpe_ratio(series_of_r , riskfree_rate ,periods_per_year):
    risk_free_per_period = ((1+riskfree_rate)**(1/periods_per_year)) -1
    excess_ret = series_of_r - risk_free_per_period
    ann_excess_ret = annualized_return(excess_ret,periods_per_year)
    ann_vol = annualized_volatility(excess_ret , periods_per_year)
    sr = ann_excess_ret / ann_vol
    return sr

def portfolio_return(weights,exp_returns):
    '''weights->returns'''
    return weights.T @ exp_returns
  
def portfolio_vol(weights,covmat):
    '''weights->volatility'''
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(no_points , exp_ret , cov):
    import numpy as np
    'for when we have a 2 asset portfolio , cov is 2*2'
    if exp_ret.shape[0] != 2:
        raise ValueError('plot_ef2 needs 2 assets')
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,no_points)]
    returns = [portfolio_return(w,exp_ret) for w in weights]
    volatilities = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({'Returns':returns,'Volatility':volatilities})
    return ef.plot.line(x='Volatility',y='Returns',style='.-')

def minimize_volatility(target_return,er,cov):
    '''
    from a target return (for ex. the 0.15) to the weight vector , r -> w
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),) * n
    return_is_target = {
        'type':'eq' , 
        'args':(er,),
        'fun': lambda weights,er : target_return - portfolio_return(weights,er)
    }
    weights_sum_to1 = {
        'type':'eq',
        'fun':lambda weights : np.sum(weights)-1
    }
    
    'now we run the optimizer'
    results = minimize(portfolio_vol,init_guess,args=(cov,),method='SLSQP',
                       options={'disp':False},
                       constraints=(return_is_target,weights_sum_to1),
                       bounds=bounds
                      )
    return results.x

def optimal_weights(no_points,er,cov):
    '''
    i set the target returns here and they convert to optimal weights with the optimizer
    '''
    target_rs = np.linspace(er.min(),er.max(),no_points)
    weights = [minimize_volatility(target_return,er,cov) for target_return in target_rs]
    return weights

def gmv(cov):
    '''
    returns the weights of the global minimum vol portfolio given cov, we compute the msr with same returns(by force it will min volatility)
    cov,same_return,indeferent rf -> w 
    '''
    n = cov.shape[0]
    return msr(0,np.repeat(1,n),cov)
    
        
def plot_ef(no_points , er , cov, show_cml=False,style='.-',riskfree_rate=0 , show_ew=False, show_gmv=False):
    import numpy as np
    """plots the n-asset portfolio"""
    weights = optimal_weights(no_points, er, cov)
    returns = [portfolio_return(w,er) for w in weights]
    volatilities = [portfolio_vol(w,cov) for w in weights]
    ef = pd.DataFrame({'Returns':returns,'Volatility':volatilities})
    ax = ef.plot.line(x='Volatility',y='Returns',style=style)
    if show_ew:
        n = er.shape[0]
        w_ep = np.repeat(1/n,n)
        r_ew = portfolio_return(w_ep,er)
        vol_ew = portfolio_vol(w_ep,cov)
        #display ew
        ax.plot([vol_ew],[r_ew] , marker='o',markersize=12)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv,er)
        vol_gmv = portfolio_vol(w_gmv,cov)
        #display gmv
        ax.plot([vol_gmv],[r_gmv] , marker='o',markersize=12)
    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate,er,cov)
        r_msr = portfolio_return(w_msr,er)
        vol_msr = portfolio_vol(w_msr,cov)
        #we plot the cml
        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,marker='o')
        
    return ax

def msr(riskfree_rate,er,cov):
    '''
    returns the weights of the portfolio that give a maximum
    sharpe ratio , given rf er ,cov
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),) * n
    weights_sum_to1 = {
        'type':'eq',
        'fun':lambda weights : np.sum(weights)-1
    }
    def negative_sharpe_ratio(weights, riskfree_rate, er, cov):
        'the negative sharpe, !!! given weights'
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    'we want to maximize the sharpe , so we minimize -sharpe'
    results = minimize(negative_sharpe_ratio,init_guess,
                       args=(riskfree_rate,er,cov,),method='SLSQP',
                       options={'disp':False},
                       constraints=(weights_sum_to1),
                       bounds=bounds
                      )
    return results.x
      
def run_cppi(risky_r , safe_r=None, m=3 , start =1000 , floor = 0.8 , riskfree_rate = 0.03, drawdown=None):
    '''
    Runs a test of the cppi strategy , given: risky_returns , output: dict with 1.account_value_history 2.risk_budget_history
    3.risk_weight history
    '''
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    if isinstance(risky_r,pd.Series):
        risky_r = pd.DataFrame(risky_r,columns=['R'])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12
        
    # something like a backtest to see what is happening
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(account_value,peak)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/ account_value
        risky_w = m * cushion
        risky_w = np.minimum(1,risky_w)
        risky_w = np.maximum(0,risky_w)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the values and plot history
        cushion_history.iloc[step] = cushion
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_w
        
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        'wealth':account_history,
        'risky wealth':risky_wealth,
        'risk budget cushion':cushion_history,
        'risky allocation':risky_w_history,
        'm':m,
        'start':start,
        'floor':floor,
        'risky_r':risky_r,
        'safe_r':safe_r
    }
    return backtest_result

def summary_stats(r,riskfree_rate=0.03):
    ann_r = r.aggregate(annualized_return,periods_per_year=12)
    ann_vol=r.aggregate(annualized_volatility,periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio,riskfree_rate = riskfree_rate , periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown_compute(r).drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian_cf_modification,modification=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        'annualized_return':ann_r,
        'annualized_volatility':ann_vol,
        'skewness':skew,
        'kurtosis':kurt,
        'cornish-fisher var 5%':cf_var5,
        'historic cvar 5%':hist_cvar5,
        'sharpe ratio':ann_sr,
        'max drawdown':dd
    })
    
def gbm1(n_years=10, n_scenarios=1000, mu=0.07 , sigma=0.15, steps_per_year=12, s_0=100):
    '''
    geometric brownian motion , evolution of stock price , vectorized good
    '''
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    n_scenarios = int(n_scenarios)
    rets_plus1 = np.random.normal(loc=(1+mu*dt),scale=(sigma*np.sqrt(dt)),size=(n_steps,n_scenarios))
    rets_plus1[0]=1
    # convert rets to prices
    prices = s_0*pd.DataFrame(rets_plus1).cumprod()
    return prices

def gbm0(n_years=10, n_scenarios=1000, mu=0.07 , sigma=0.15, steps_per_year=12, s_0=100):
    '''
    geometric brownian motion , evolution of stock price
    '''
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    xi = np.random.normal(size=(n_steps,n_scenarios))
    rets = mu*dt + sigma*np.sqrt(dt)*xi
    rets = pd.DataFrame(rets)
    # convert rets to prices
    prices = s_0*(1+rets).cumprod()
    return prices

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val