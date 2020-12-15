import numpy as np
import pandas as pd
from scipy.stats import norm, binom
import plotly.graph_objects as go
import datetime as dt
from arch import arch_model
from scipy.optimize import minimize

class layout:
    tight = dict(t=25,b=0,l=0,r=0,pad=0)

def emily_example():
    # Create object Returns ~ N(0,10^2)
    normR = norm(loc=0,scale=10)

    fig = go.Figure()
    
    # Plot 1%-99.9% Region
    fig.add_trace(go.Scatter(x=np.linspace(normR.ppf(0.01),normR.ppf(0.999), 1000),
                             y=normR.pdf(np.linspace(normR.ppf(0.01),normR.ppf(0.999), 1000)),
                             mode='lines',name='Estimated Distribution of Returns'))
    
    # Plot 0%-1% Region 
    fig.add_trace(go.Scatter(x=np.linspace(normR.ppf(0.001),normR.ppf(0.01), 1000),
                             y=normR.pdf(np.linspace(normR.ppf(0.001),normR.ppf(0.01), 1000)),
                             mode='lines',name='1% Cumulative Probability Density Region',
                             marker_color='crimson',fill='tozeroy'))
    
    # Annotate 1% VaR
    fig.add_annotation(x=normR.ppf(0.01),y=normR.pdf(normR.ppf(0.01)),
                        text='1% VaR: {:.2f}%'.format(-1*normR.ppf(0.01)),
                        arrowcolor='black',
                        showarrow=True,arrowhead=6,arrowsize=1.5,ax=100,ay=0)
    
    # Annotate Observed Return
    fig.add_annotation(x=-25,y=normR.pdf(-25),
                        text='Observed Return: -25%',
                        arrowcolor='black',
                        showarrow=True,arrowhead=6,arrowsize=1.5,ax=0,ay=-50)
    
    fig.update_layout(margin=layout.tight,height=300,
                      yaxis_title='Probability Density',
                      xaxis_title='Returns')
    
    return fig.show()

def binomial_example():
    binomE = binom(n=152,p=0.01)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(10)),
                         y=binomE.pmf(list(range(10)))))
    
    fig.update_layout(margin=layout.tight,xaxis=dict(title='Number of Exceedances',
                                                     tickvals=list(range(10)),
                                                     ticktext=list(range(10))),
                      yaxis_title='Probability Mass',height=300)
    
    return fig.show()

class evaluate:
    """
    A comprehensive evaluation of a GARCH model combined with a mean process
    
    Arguments:
        [1] data - a dataframe with the columns: ['forecasted_sigma','forecasted_mu','observed','params'] and a datetime index
    """
    def __init__(self,data):
        # Distributions
        data['type'] = data.apply(self.sigmas,axis=1)
        
        # Turn observed returns into cumulative probability density
        data['cd'] = data.apply(lambda row: norm.cdf(row['observed'],
                                                     loc=row['forecasted_mu'],
                                                     scale=row['forecasted_sigma']),axis=1)
        
        # Find the points at which the cumulative probability density is equal to 0.01 and 0.05
        data['5% ppf'] = data.apply(lambda row: norm.ppf(0.05,
                                                         loc=row['forecasted_mu'],
                                                         scale=row['forecasted_sigma']),axis=1)
        data['1% ppf'] = data.apply(lambda row: norm.ppf(0.01,
                                                         loc=row['forecasted_mu'], 
                                                         scale=row['forecasted_sigma']),axis=1)
        
        # Compare the observed cumulative probability densities with these points
        # To find exceedances of 1% and 5% VaR
        data['exceedance'] = data['cd'].apply(lambda x: '1' if x<=0.01 else ('5' if x<=0.05 else '_'))
        
        self.data = data
    
    def sigmas(self,row):
        """
        Turns an observed return into a sigma category
        """
        if np.abs(row['observed']-row['forecasted_mu']) < 1*row['forecasted_sigma']:
            return '1'
        if np.abs(row['observed']-row['forecasted_mu']) < 2*row['forecasted_sigma']:
            return '2'
        if np.abs(row['observed']-row['forecasted_mu']) < 3*row['forecasted_sigma']:
            return '3'
        else:
            return '>3'
    
    def statistics(self):
        """
        Returns performance statistics of the model in the form:
            [log-likelihood, days of 1% exceedance, days 5% exceedance]
        """
        data = self.data
        
        # Log-probability of each observed value given distribution parameters
        data['log-p'] = data.apply(lambda row: norm.logpdf(row['observed'],
                                                           loc=row['forecasted_mu'],
                                                           scale=row['forecasted_sigma']),axis=1)
        
        # Log-likelihood of the model
        ll = data['log-p'].sum()
        
        # VaR exceedances
        one_ex = len(data.loc[data['exceedance']=='1'])
        five_ex = len(data.loc[data['exceedance'].isin(['1','5'])])

        # Probability of the VaR exceedances using a binomial model
        one_ex_p = binom.pmf(one_ex, len(data), 0.01)
        five_ex_p = binom.pmf(five_ex, len(data), 0.05)
        
        return [np.round(ll,2),one_ex,np.round(one_ex_p,2),five_ex,np.round(five_ex_p,2)]
    
    def plot(self,parameters):
        """
        Plots 3 graphs:
            [1] A time series of observed returns vs predicted distribution of returns
            [2] A bar chart of expected vs observed returns in terms of regions around the mean
            [3] A time series of observed returns versus expected value at risk
            
        If parameters is true plots a time series of the model parameter (𝜔,𝛼,𝛽) estimates
        """        
        data = self.data
        
        # -------------------------------------------
        # Distribution Analysis
        # -------------------------------------------
        
        # Line Plot
        
        fig = go.Figure()
        
        # Plot positive sigmas
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']+data['forecasted_sigma'],
                                 marker_color='lightgrey',
                                 name='+1𝜎',fill='tozeroy',showlegend=False,legendgroup="+"))
        
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']+2*data['forecasted_sigma'],
                                 marker_color='grey',
                                 name='+2𝜎',fill='tonexty',showlegend=False,legendgroup="+"))
        
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']+3*data['forecasted_sigma'],
                                 marker_color='crimson',
                                 name='+3𝜎',fill='tonexty',showlegend=False,legendgroup="+"))
        
        # Blank legend place holder for positive sigmas
        fig.add_trace(go.Scatter(x=[data.index[0]],
                                 y=[0],name='Positive Sigmas',legendgroup="+",opacity=0))
        
        # Plot negative sigmas
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']-data['forecasted_sigma'],
                                 marker_color='lightgrey',
                                 name='-1𝜎',fill='tozeroy',showlegend=False,legendgroup="-"))
        
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']-2*data['forecasted_sigma'],
                                 marker_color='grey',
                                 name='-2𝜎',fill='tonexty',showlegend=False,legendgroup="-"))
        
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['forecasted_mu']-3*data['forecasted_sigma'],
                                 marker_color='crimson',
                                 name='-3𝜎',fill='tonexty',showlegend=False,legendgroup="-"))
        
        # Blank legend place holder for negative sigmas
        fig.add_trace(go.Scatter(x=[data.index[0]],
                                 y=[0],name='Negative Sigmas',legendgroup="-",opacity=0))
        
        # Plot mean
        fig.add_trace(go.Scatter(x=data.index,y=data['forecasted_mu'],
                                 marker_color='black',
                                 name='𝜇'))
        
        # Plot points
        fig.add_trace(go.Scatter(x=data.loc[data['type']=='1'].index,y=data.loc[data['type']=='1','observed'],
                                 marker={'color':'grey','size':5},mode='markers',
                                 name='1'))

        fig.add_trace(go.Scatter(x=data.loc[data['type']=='2'].index,y=data.loc[data['type']=='2','observed'],
                                 marker_color='black',mode='markers',
                                 name='2'))

        fig.add_trace(go.Scatter(x=data.loc[data['type']=='3'].index,y=data.loc[data['type']=='3','observed'],
                                 marker={'color':'pink','symbol':'square','line':{'width':1}},mode='markers',
                                 name='3'))

        fig.add_trace(go.Scatter(x=data.loc[data['type']=='>3'].index,y=data.loc[data['type']=='>3','observed'],
                                 marker={'color':'red','symbol':'diamond','line':{'width':1}},mode='markers',
                                 name='>3'))

        fig.update_layout(margin=layout.tight,height=200,width=900)
        fig.show()
        
        # Bar chart

        fig = go.Figure()
        
        expected = []
        observed = []
        
        for sigma,p in zip(['1','2','3','>3'],[0.68,0.95-0.68,0.997-0.95,1-0.997]):
            expected.append(p*len(data.index))
            observed.append(len(data.loc[data['type']==sigma]))
        
        _ = ['within 1𝜎','within 2𝜎','within 3𝜎','outside 3𝜎']
        
        fig.add_trace(go.Bar(x=_,y=expected,name='Expected',marker_color='darkgrey'))
        fig.add_trace(go.Bar(x=_,y=observed,name='Observed',marker_color='crimson'))
            
        fig.update_layout(margin=layout.tight,height=100,width=900)
        fig.show()
        
        # -------------------------------------------
        # VaR Analysis
        # -------------------------------------------
        
        fig = go.Figure()
        
        # Plot points
        fig.add_trace(go.Scatter(x=data.loc[data['exceedance']=='5'].index,
                                 y=data.loc[data['exceedance']=='5','observed'],
                                 marker={'color':'pink','symbol':'square','line':{'width':1}},mode='markers',
                                 name='5% exceedance'))

        fig.add_trace(go.Scatter(x=data.loc[data['exceedance']=='1'].index,
                                 y=data.loc[data['exceedance']=='1','observed'],
                                 marker={'color':'red','symbol':'diamond','line':{'width':1}},mode='markers',
                                 name='1% exceedance'))
        
        fig.add_trace(go.Scatter(x=data.loc[data['exceedance']=='_'].index,
                                 y=data.loc[data['exceedance']=='_','observed'],
                                 marker={'color':'grey','size':5},mode='markers',
                                 name='no exceedance'))
        
        # Plot lines
        fig.add_trace(go.Scatter(x=data.index,y=data['1% ppf'],
                                 marker_color='crimson',mode='lines',
                                 name='1% VaR'))

        fig.add_trace(go.Scatter(x=data.index,y=data['5% ppf'],
                                 marker_color='darkgrey',mode='lines',
                                 name='5% VaR'))
        
        fig.update_layout(margin=layout.tight,height=200,width=900)
        fig.show()
        
        # -------------------------------------------
        # Parameter Plots
        # -------------------------------------------
        
        if parameters:
            self.parameter_plot()
        
        return
    
    def parameter_plot(self):
        """
        Plots a time series of the model parameter (𝜔,𝛼,𝛽) estimates
        """
        data = self.data
        
        parameters = pd.Series(data['params'].iloc[0].index.values)
        p = len(parameters.loc[parameters.str.contains('alpha')])
        q = len(parameters.loc[parameters.str.contains('beta')])
        
        fig = go.Figure()
        
        for i in range(p):
            data['alpha'+str(i+1)] = data['params'].apply(lambda x:x.loc['alpha['+str(i+1)+']'])
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data['alpha'+str(i+1)],
                                     name='𝛼<sub>'+str(i+1)+'</sub>'))
        for i in range(q):
            data['beta'+str(i+1)] = data['params'].apply(lambda x:x.loc['beta['+str(i+1)+']'])
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data['beta'+str(i+1)],
                                     name='𝛽<sub>'+str(i+1)+'</sub>'))
        
        data['omega'] = data['params'].apply(lambda x:x.loc['omega'])
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['omega'],
                                 name='𝜔'))
        
        fig.update_layout(margin=layout.tight,height=175,width=900)
        
        return fig.show()

def garch_backtest(backtest_period,returns,ticker,sample_size,mean,q,p):
    """
    Performs a backtest across the test period
    
    Arguments:
        [1] backtest_period - a pd.series of dates
        [2] returns - the dataframe with the observed returns
        [3] ticker (str) - the symbol of the equity to run the model on
        [4] sample_size (int) - the sample size of the training period
        [5] mean (str) - the mean process to use one of ['Zero','AR']
        [6] q (int) - the number of conditional variance terms
        [7] p (int) - the number of innovation terms
    
    Returns a dataframe with a row for each day in the backtest period
    Each day has:
        * an observed value
        * a forecasted 𝜎
        * a forecasted 𝜇
        * a pandas series of parameters estimated
    """
    forecasted_sigma = []
    forecasted_mu = []
    params = []
    for date in backtest_period:
        # Filter out all data after and including the day being forecasted
        # Take the most recent X days to train upon where X is the sample size
        model = arch_model(returns.loc[:date-dt.timedelta(days=1),ticker].iloc[-sample_size:],
                           vol='GARCH',
                           q=q,p=p,mean=mean,dist='Normal').fit(disp='off')

        params.append(model.params)
        
        # Forecast sigma
        forecasted_sigma.append(np.sqrt(
            model.forecast(horizon=1).variance.iloc[-1].values[0]))
        
        # Forecast mu
        forecasted_mu.append(
            model.forecast(horizon=1).mean.iloc[-1].values[0])

    return pd.DataFrame({'observed':returns[ticker].loc[backtest_period],
                         'forecasted_sigma':forecasted_sigma,
                         'forecasted_mu':forecasted_mu,
                         'params':params},index=returns.loc[backtest_period].index)

class new_garch_model:
    """
    A class to minimise the negative log likelihood of the kappa and tau parameters
    for each model
    """
    def __init__(self,returns,ticker,sample_size,index,mean,q,p):
        self.returns = returns
        self.ticker = ticker
        self.index = index
        self.sample_size = sample_size
        self.mean = mean
        self.q = q
        self.p = p
        self.training_period = returns.loc[(returns.index>pd.to_datetime('1 Jan 2020'))&
                                           (returns.index<pd.to_datetime('1 March 2020'))].index
    
    def negative_log_likelihood(self,parameters):
        """
        The objective function to fit the parameters
        """
        returns = self.returns
        
        kappa = parameters[0]
        tau = parameters[1]

        forecasted_sigma = []
        observed = np.abs(returns[self.ticker].loc[self.training_period])

        for date in self.training_period:
            # Filter out all data after and including the day being forecasted
            # Take the most recent 1000 days of that data to train upon
            model = arch_model(returns.loc[:date-dt.timedelta(days=1),self.ticker].iloc[-self.sample_size:],
                                vol='GARCH',
                                q=self.q,p=self.q,mean=self.mean,dist='Normal').fit(disp='off')

            params = model.params
            omega_e = kappa*returns.loc[:date-dt.timedelta(days=1),self.index+' rolling'][-1]+tau*params[0]

            # Amend parameter vector
            params[0] = omega_e
            forecasted_sigma.append(np.sqrt(
                model.forecast(horizon=1,params=params).variance.values[-1,:][0]))

        data = pd.DataFrame({'observed':observed.values,'forecast':forecasted_sigma},index=observed.index)
        data['log-p'] = data.apply(lambda row: norm.logpdf(row['observed'], loc=0, scale=row['forecast']),axis=1)
        ll = data['log-p'].sum()

        return -ll
    
    def fit(self):
        """
        Optimises the objective function with constraints
        """
        # constraint 1 (equation 2.3)
        con1 = {'type':'ineq','fun':lambda parameters:np.array(-sum(parameters)+1)}

        # constraint 2 in the inequality form lambda parameters:np.array(parameters)(equation 2.4)
        # does not work so we use bounds instead

        res = minimize(self.negative_log_likelihood,
                       x0=[0.5,0.5],
                       constraints=[con1],
                       bounds=[[0,1],[0,1]],
                       method='SLSQP')
        
        return res

def kappa_tau_table(models,returns):
    """
    Returns a table of the maximum likelihood estimates of Kappa and Tau
    for each model
    """
    kappa = []
    tau = []
    tuples = []
    table = np.array(['kappa','tau'])
    
    for model in models:
        for index in ['FTSE','S&P']:
            res = new_garch_model(returns=returns,index=index,**model).fit()
            table = np.vstack((table,[res.x[0],res.x[1]]))
            tuples.append((*model.values(),index))
            
    index = pd.MultiIndex.from_tuples(tuples, names=['ticker','sample size','mean process','p','q','index'])
    table = pd.DataFrame(table[1:,:],columns=table[0,:],index=index)
            
    return table

def new_garch_backtest(backtest_period,returns,ticker,sample_size,index,mean,q,p,kt_table):
    """
    Arguments:
        [1] backtest_period - a pd.series of dates
        [2] returns - the dataframe with the observed returns
        [3] ticker (str) - the symbol of the equity to run the model on
        [4] sample_size (int) - the sample size of the training period
        [5] mean (str) - the mean process to use one of ['Zero','AR']
        [6] q (int) - the number of conditional variance terms
        [7] p (int) - the number of innovation terms
        [8] kt_table - the table consisting of the estimated 𝜅 and 𝜏 parameters
    
    Returns a dataframe with a row for each day in the backtest period
    Each day has:
        * an observed value
        * a forecasted 𝜎
        * a forecasted 𝜇
        * a pandas series of parameters estimated
    """
    
    forecasted_sigma = []
    forecasted_mu = []
    params = []
    
    # Get the corresponding kappa and tau parameters from the kt_table
    kappa, tau = kt_table.loc[ticker,sample_size,mean,p,q,index].apply(lambda x:float(x))
    
    for date in backtest_period:
        model = arch_model(returns.loc[:date-dt.timedelta(days=1),ticker].iloc[-sample_size:],vol='GARCH',
                               q=q,p=p,mean=mean,dist='Normal').fit(disp='off')
        
        # Obtain model parameters
        parameters = model.params
        omega_old = parameters[0]
        
        # Overwrite model parameters
        omega_new = kappa*returns.loc[:date-dt.timedelta(days=1),index+' rolling'][-1]+tau*omega_old
        parameters[0] = omega_new
        
        # Append them
        params.append(model.params)
        
        # Forecast sigma
        forecasted_sigma.append(np.sqrt(
            model.forecast(horizon=1).variance.iloc[-1].values[0]))
        
        # Forecast mu (0)
        forecasted_mu.append(0)
    
    return pd.DataFrame({'observed':returns[ticker].loc[backtest_period],
                         'forecasted_sigma':forecasted_sigma,
                         'forecasted_mu':forecasted_mu,
                         'params':params},index=returns.loc[backtest_period].index)

def results_table(backtest_period,returns):
    """
    Runs all models through the backtest and presents results in a pandas dataframe
    Instantiated at the start of the notebook and then progressively filtered throughout

    Table has columns: 
        * Ticker (str)
        * Sample size (int)
        * Mean process (str)
        * Model (str)
        * Index (for IR GARCH models) (str)
        * Log-Likelihood (float)
        * Number of 1% exceedances (int)
        * Probability of 1% exceedances (float)
        * Number of 5% exceedances (int)
        * Probability of 5% exceedances (float)
        * Kappa estimate (for IR GARCH models) (float)
        * Tau estimate (for IR GARCH models) (float)
    """
    # Create a blank DataFrame
    table = pd.DataFrame({column:[] for column in ([
        'Ticker','Sample size','Mean process','Model','Index', # Model description columns 
        'Log-Likelihood','1% E','P(1% E=1% e)','5% E','P(5% E=5% e)','kappa','tau' # Model evaluation columns
        ])})
    
    # Normal GARCH results
    for ticker in ['IAG.L','RNO.PA','WFC']:
        for sample_size in [200,400,600,800,1000]:
            for mean_process in ['Zero','AR']:
                for parameters,model in zip(({'q':1,'p':1},{'q':5,'p':1},{'q':1,'p':5}),
                                            ('GARCH(1,1)','GARCH(1,5)','GARCH(5,1)')):
                    
                    ll, one_ex, one_ex_p, five_ex, five_ex_p = evaluate(garch_backtest(
                                                backtest_period=backtest_period,
                                                returns=returns,
                                                ticker=ticker,
                                                sample_size=sample_size,
                                                mean=mean_process,
                                                **parameters)).statistics()

                    table = table.append(
                        {'Ticker':ticker,'Sample size':sample_size,'Mean process':mean_process,
                        'Model':model,
                        'Index':'', #Blank for index
                        'Log-Likelihood':ll,
                        '1% E':one_ex,'P(1% E=1% e)':one_ex_p,
                        '5% E':five_ex,'P(5% E=5% e)':five_ex_p,
                        'kappa':'','tau':''}, # Blank for kappa and tau
                        ignore_index=True
                    )
    
    models = [
        {'ticker':'IAG.L','sample_size':1000,'mean':'Zero','p':1,'q':1},
        {'ticker':'WFC','sample_size':200,'mean':'Zero','p':1,'q':1},
        {'ticker':'RNO.PA','sample_size':400,'mean':'AR','p':5,'q':1},
        {'ticker':'RNO.PA','sample_size':200,'mean':'AR','p':1,'q':1}
        ]

    # IR GARCH results
    kt_table = kappa_tau_table(models=models,returns=returns)

    for model in models:
        for index in ['FTSE','S&P']:
            ll, one_ex, one_ex_p, five_ex, five_ex_p = evaluate(new_garch_backtest(
                                        backtest_period=backtest_period,returns=returns,
                                        index=index,**model,kt_table=kt_table)).statistics()
            
            table = table.append(
                        {'Ticker':model['ticker'],'Sample size':model['sample_size'],'Mean process':model['mean'],
                        'Model':'IR GARCH('+str(model['p'])+','+str(model['q'])+')',
                        'Index':index, #Blank for index
                        'Log-Likelihood':ll,
                        '1% E':one_ex,'P(1% E=1% e)':one_ex_p,
                        '5% E':five_ex,'P(5% E=5% e)':five_ex_p,
                        'kappa':np.round(float(kt_table.loc[(*model.values(),index),'kappa']),2),
                        'tau':np.round(float(kt_table.loc[(*model.values(),index),'tau']),2)},
                        ignore_index=True
                    )
    
    # Sort models by the most probable
    table.sort_values(by=['Ticker','P(1% E=1% e)','P(5% E=5% e)'],ascending=False,inplace=True)
    
    # -------------------------------------------
    # First view - all the normal GARCH models
    # -------------------------------------------

    first = table.loc[np.invert(table['Model'].str.contains('IR'))]
    # Remove non-relevant columns
    first = first.loc[:,[column not in ['Index','kappa','tau'] for column in first.columns]]
    
    # -------------------------------------------
    # Second view - the selected normal GARCH models and their IR GARCH counterparts
    # -------------------------------------------
    
    models = pd.DataFrame.from_dict(models)
    
    # Filter for ticker, sample size and mean process
    second = table.loc[
            ((table['Ticker']=='IAG.L')&(table['Sample size']==1000)&(table['Mean process']=='Zero')& # IAG.L
            ((table['Model']=="GARCH(1,1)")|(table['Model']=="IR GARCH(1,1)")))|
            ((table['Ticker']=='WFC')&(table['Sample size']==200)&(table['Mean process']=='Zero')& # WFC
            ((table['Model']=="GARCH(1,1)")|(table['Model']=="IR GARCH(1,1)")))|
            ((table['Ticker']=='RNO.PA')&(table['Sample size']==400)&(table['Mean process']=='AR')& # RNO.PA 1
            ((table['Model']=="GARCH(5,1)")|(table['Model']=="IR GARCH(5,1)")))|
            ((table['Ticker']=='RNO.PA')&(table['Sample size']==200)&(table['Mean process']=='AR')& # RNO.PA 2
            ((table['Model']=="GARCH(1,1)")|(table['Model']=="IR GARCH(1,1)")))]
    
    return first, kt_table, second