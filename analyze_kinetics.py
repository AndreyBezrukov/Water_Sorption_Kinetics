# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:03:18 2022

@author: Andrey.Bezrukov
"""
##################################################
####### loading libraries
##################################################

import numpy  as np                
import pandas as pd
import matplotlib.pyplot as plt   
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
from scipy import interpolate
from scipy.stats.distributions import t


##################################################
####### create sorption class
##################################################

class Sorption:
    def read_file(self, path, filename):
        print(path +'/'+ filename)   
        if (filename[-4:]=='.xls') or (filename[-5:]=='.xlsx'):  
            try:
                df = pd.read_excel(path + filename, sheet_name='DVS Data')
                if df.columns[0] == 'DVS-INTRINSIC DATA FILE':
                    self.filename = filename
                    self.instrument = df.columns[0].split()[0]
                    self.sample_mass = df.iloc[4, 7]
                    self.fluid = 'water vapor'
                    comments  = str(df.iloc[3, 1])
                    method = str(df.iloc[1, 1])
                    df.columns = df.iloc[22, :]
                    df = df.iloc[23:, :]
                    df = df.reset_index(drop=True)
                    df = df.astype('float')
                    # renaming columns to common notation
                    df = df[['Time (min)', 'dm (%) - ref', 'Target RH (%)', 'Actual RH (%)', 'Target Sample Temp', 'Actual Sample Temp']]
                    df.columns = ['time', 'uptake', 'RH_target', 'RH_actual', 'temp_target', 'temp_actual']
                    self.temperature = 'Actual Sample Temp: {0:.2f} +- {1:.2f}'.format(df[df.temp_target==df.temp_target.min()].temp_actual.mean(), df[df.temp_target==df.temp_target.min()].temp_actual.std())
                    self.equilibration_interval = '---'
                    self.comments = comments
                    self.method = method
                    self.data = df
                    if len(list(set(df.RH_target.to_list())))>5:
                        self.experiment_type = 'isotherm'
                    else:
                        self.experiment_type = 'kinetics'
                    #print(self.data.head())
                elif df.columns[0] == 'DVS-Advantage-Plus-Data-File':
                    self.filename = filename
                    self.instrument = df.columns[0][:-10]
                    self.sample_mass = df.iloc[27, 1]
                    self.fluid = 'water vapor'
                    comments  = str(df.iloc[8, 1])
                    method = str(df.iloc[3, 1])
                    df.columns = df.iloc[39, :]
                    df = df.iloc[40:, :30]
                    df = df.reset_index(drop=True)
                    df = df[[i for i in df.columns if (i == i) & (i != 'Chiller State')]]
                    df = df.astype('float')
                    # renaming columns to common notation
                    df = df[['Time [minutes]', 'dm (%) - ref', 'Target Partial Pressure (Solvent A) [%]', 'Measured Partial Pressure (Solvent A) [%]', 'Target Preheater Temp. [celsius]', 'Measured Preheater Temp. [celsius]', 'Mass [mg]']]
                    df.columns = ['time', 'uptake', 'RH_target', 'RH_actual', 'temp_target', 'temp_actual', 'mass']
                    self.temperature = 'Temp. [celsius]: {0:.2f} +- {1:.2f}'.format(df[df.temp_target==df.temp_target.min()].temp_actual.mean(), df[df.temp_target==df.temp_target.min()].temp_actual.std())
                    self.equilibration_interval = '---'
                    self.comments = comments
                    self.method = method
                    if len(list(set(df.RH_target.to_list())))>5:
                        self.experiment_type = 'isotherm'
                    else:
                        self.experiment_type = 'kinetics'
                    self.data = df
                    #print(self.data.head())
                elif df.columns[0] == 'DVS-Vacuum-Data-File':
                    self.filename = filename
                    self.instrument = df.columns[0][:-10]
                    self.sample_mass = df.iloc[27, 1]
                    self.fluid = 'water vapor'
                    comments  = str(df.iloc[8, 1])
                    method = str(df.iloc[3, 1])
                    df.columns = df.iloc[40, :]
                    df = df.iloc[41:, :30]
                    df = df.reset_index(drop=True)
                    df = df[[i for i in df.columns if (i == i) & (i != 'Chiller State')]]
                    df = df.astype('float')
                    # renaming columns to common notation
                    df = df[['Time [minutes]', 'dm (%) - ref', 'Target Relative Pressure [%]', 'Actual Relative Pressure [%]', 'Target Preheater Temperature [°C]', 'Actual Preheater Temperature [°C]', 'Mass [mg]']]
                    df.columns = ['time', 'uptake', 'RH_target', 'RH_actual', 'temp_target', 'temp_actual', 'mass']
                    self.temperature = 'Temp. [celsius]: {0:.2f} +- {1:.2f}'.format(df[df.temp_target==df.temp_target.min()].temp_actual.mean(), df[df.temp_target==df.temp_target.min()].temp_actual.std())
                    self.equilibration_interval = '---'
                    self.comments = comments
                    self.method = method
                    if len(list(set(df.RH_target.to_list())))>5:
                        self.experiment_type = 'isotherm'
                    else:
                        self.experiment_type = 'kinetics'
                    self.data = df
                    #print(self.data.head())
            except Exception as e:
                print(e)

class Isotherm:
    def extract_isotherm(self, Sorption):
        Sorption.data['cycle_split'] = Sorption.data['RH_target'].diff().fillna(0)
        split_index_ads = Sorption.data.index[(Sorption.data['cycle_split']>0)].to_list()
        split_index_des = Sorption.data.index[(Sorption.data['cycle_split']<0)].to_list()
        
        index_tuple = [('ads', i) for i in split_index_ads] + [('des', i) for i in split_index_des]
        index_tuple = sorted(index_tuple, key=lambda x: x[1])
        #print(index_tuple)
        isotherm_split = []
        prev = 'ads'
        for i, j in enumerate([i[0] for i in index_tuple]):
            if (j == 'ads')&(prev=='des'):
                isotherm_split.append(index_tuple[i][1])
            prev = j
        args = [0]+isotherm_split+[Sorption.data.index.size]
        #print(args)
        
        for start,  end in zip(args, args[1:]):
            #print(start,  end)
            split_index_ads = Sorption.data.iloc[start:end].index[(Sorption.data.iloc[start:end]['cycle_split']>0)].to_list()
            split_index_des = Sorption.data.iloc[start:end].index[(Sorption.data.iloc[start:end]['cycle_split']<0)].to_list()
            
            split_index_des.append(Sorption.data.iloc[start:end].index.to_list()[-1])
            split_index_ads = split_index_ads + [split_index_des[0]]
    
            self.RHtarget_ads = []
            self.RHactual_ads = []
            self.adsorption = []
            self.RHtarget_des = []
            self.RHactual_des = []
            self.desorption = []
            for i, split in enumerate(split_index_ads[:]):
                self.adsorption.append(Sorption.data[split-10:split].uptake.median())
                if i==0:
                    self.RHtarget_ads.append(Sorption.data[:split_index_ads[i]].RH_target.median())
                    self.RHactual_ads.append(Sorption.data[:split_index_ads[i]].RH_actual.median())
                else:
                    self.RHtarget_ads.append(Sorption.data[split_index_ads[i-1]:split_index_ads[i]].RH_target.median())
                    self.RHactual_ads.append(Sorption.data[split_index_ads[i-1]:split_index_ads[i]].RH_actual.median())
            for i, split in enumerate(split_index_des[:]):
                self.desorption.append(Sorption.data[split-10:split].uptake.median())
                if i == 0:
                    self.RHtarget_des.append(Sorption.data[split_index_ads[-2]:split_index_ads[-1]].RH_target.median())
                    self.RHactual_des.append(Sorption.data[split_index_ads[-2]:split_index_ads[-1]].RH_actual.median())
                else:
                    self.RHtarget_des.append(Sorption.data[split_index_des[i-1]:split_index_des[i]].RH_target.median())
                    self.RHactual_des.append(Sorption.data[split_index_des[i-1]:split_index_des[i]].RH_actual.median()) 
    
    def interpolate_uptake_to_RH_ads(self):
        self.uptake_to_RH_ads = interpolate.interp1d([i for i in self.adsorption],  
                                 [i for i in self.RHactual_ads],
                                 fill_value="extrapolate")
    
    def interpolate_uptake_to_RH_des(self):
        self.uptake_to_RH_des = interpolate.interp1d([i for i in self.desorption],  
                                 [i for i in self.RHactual_des],
                                 fill_value="extrapolate")  

    def interpolate_RH_to_uptake_ads(self):
        self.RH_to_uptake_ads = interpolate.interp1d([i for i in self.RHactual_ads],
                                                     [i for i in self.adsorption],  
                                                     fill_value="extrapolate")
    
    def interpolate_RH_to_uptake_des(self):
        self.RH_to_uptake_des = interpolate.interp1d([i for i in self.RHactual_des],
                                                     [i for i in self.desorption],  
                                                     fill_value="extrapolate")  
class Kinetics:
    def decompose_to_cycles(self, Sorption):
        df = Sorption.data
        df['cycle_split'] = df['RH_target'].diff().fillna(0)
        split_index = df.index[df['cycle_split']!=0].tolist()
        split_index = [0]+split_index+[df.index.max()]
        cycle_number=[]
        dry_mass=[]
        ads_count = 1
        des_count = -1
        for i, j in enumerate(split_index[:-1]):
            if (df[split_index[i]:split_index[i+1]].RH_target.median() == 0) or (df.loc[split_index[i], 'cycle_split']<0):
                cycle_number = cycle_number + [des_count for i in range(split_index[i+1]-split_index[i])]
                dry_mass = dry_mass + [df.iloc[split_index[i+1], :].mass for k in range(split_index[i+1]-split_index[i])]
                des_count=des_count-1
                ads_count=ads_count+1
            else:
                cycle_number = cycle_number + [ads_count for i in range(split_index[i+1]-split_index[i])]
                dry_mass = dry_mass +  [df.iloc[split_index[i], :].mass for k in range(split_index[i+1]-split_index[i])]
        cycle_number = cycle_number + [cycle_number[-1]]
        dry_mass = dry_mass + [dry_mass[-1]]
        df['cycle_number'] = cycle_number
        df['dry_mass'] = dry_mass
        Sorption.data = df
        
    def fit_model(self, Sorption, Isotherm, material,verbose=False, plot=False, scale_isotherm = False):
        iso_scale = 1
         
        def solving_ads(t, K):
            uptake_out = []
            B = uptake[0]
            t0=time[0]
            for i, t1 in enumerate(t):
                B = B + (t1-t0)*(np.multiply( K* (RH[i] - Isotherm.uptake_to_RH_ads(B*iso_scale)) , \
                                         RH[i]>Isotherm.uptake_to_RH_ads(B*iso_scale)))  
                uptake_out.append(B)
                t0=t1
            return uptake_out
        def solving_des(t,  K):
            uptake_out = []
            B = uptake[0]
            t0=time[0]
            for i, t1 in enumerate(t):
                B = B + (t1-t0)*(np.multiply( K* (RH[i] - Isotherm.uptake_to_RH_ads(B*iso_scale)) , \
                                         RH[i]<Isotherm.uptake_to_RH_ads(B*iso_scale)))  
                uptake_out.append(B)
                #print(B)
                t0=t1
            return uptake_out
        df_result = pd.DataFrame(columns=['sample_mass', 'cycle', 'popt', 'pcov', 'R2', 'RH_target'], dtype=object)    
        for cycle in Sorption.data.cycle_number.unique()[1:]:
            if (cycle>0)&((cycle-2)%3 == 0): fig, arrax  = plt.subplots(3, 2, figsize = (10, 10))
            time = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].time.values
            time = time - time[0]
            uptake = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.values
            RH = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].RH_actual.values
            if cycle<0:
                if scale_isotherm: 
                    iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.max()
                print(iso_scale)
                popt, pcov = curve_fit(solving_des, time, uptake, p0=[ 0.02], #maxfev=15000,
                           bounds = (0, [10**3])
                           )
                R2_value = r2_score(solving_des(time  , *popt),  uptake)
                if plot:
                    arrax[ (np.abs(cycle)-2)%3, 1].scatter(time,  uptake)
                    arrax[ (np.abs(cycle)-2)%3, 1].plot(time, solving_des(time  , popt[0]+np.sqrt(pcov[0][0])))
                    arrax[ (np.abs(cycle)-2)%3, 1].plot(time, solving_des(time  , popt[0]-np.sqrt(pcov[0][0])))
                    arrax[ (np.abs(cycle)-2)%3, 1].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_title('Desorption: cycle {0}, k: {1:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value))
                if verbose:
                    print('des', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), popt, pcov, R2_value )
                if (np.abs(cycle)-2)%3 == 2: 
                    plt.suptitle('{0}, {1} mg, Humidity swing {2} - {3} % RH'.format(material, 
                                 Sorption.sample_mass, 
                                 Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                 Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                 ),
                    y=1.02
                    )
                    plt.tight_layout()
                    plt.show()
            else:
                if scale_isotherm: 
                    iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.max()
                print(iso_scale)
                popt, pcov = curve_fit(solving_ads, time, uptake, p0=[ 0.02], #maxfev=15000,
                           bounds = (0, [10**3])
                           )
                R2_value = r2_score(solving_ads(time  , *popt),  uptake)
                if plot:
                    arrax[ (np.abs(cycle)-2)%3,  0].scatter(time,  uptake)
                    arrax[ (np.abs(cycle)-2)%3,  0].plot(time, solving_ads(time  , popt[0]+np.sqrt(pcov[0][0])))
                    arrax[ (np.abs(cycle)-2)%3,  0].plot(time, solving_ads(time  , popt[0]-np.sqrt(pcov[0][0])))
                    arrax[ (np.abs(cycle)-2)%3, 0].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_title('Adsorption: cycle {0}, k: {1:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value))
                if verbose:
                    print('ads', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean() ,popt, pcov, R2_value )
            df_result = df_result.append({'sample_mass':Sorption.data[Sorption.data.cycle_number == cycle].dry_mass.values.mean(),
                                          'experiment':Sorption.filename,
                                          'cycle': cycle, 
                                          'popt':popt[0], 
                                          'pcov':pcov[0][0],
                                          'R2': R2_value, 
                                          'RH_target':round(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 1),
                                          }, ignore_index=True)
            self.result = df_result
    def fit_model2(self, Sorption, step, t0_range, Isotherm, material, verbose=False, plot=False, scale_isotherm = False):
        #### one model for adsorption and desorption ####
        iso_scale = 1
         
        def solving(t, K):
            uptake_out = []
            B = uptake[0]
            t0=time[0]
            for i, t1 in enumerate(t):
                B = B + (t1-t0)* K* (RH[i] - Isotherm.uptake_to_RH_ads(B*iso_scale)) 
                uptake_out.append(B)
                t0=t1
            return uptake_out
        
        df_result = pd.DataFrame(columns=['sample_mass', 'cycle', 'popt', 'pcov', 'R2', 'RH_target'], dtype=object)    
        for cycle in [i for i in Sorption.data.cycle_number.unique()[1:] ]:
            if (cycle>0)&((cycle-2)%3 == 0): fig, arrax  = plt.subplots(3, 2, figsize = (10, 10))
            time = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[::step, :].time.values
            time = time - time[0]
            uptake = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[::step, :].uptake.values
            RH0 = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[::step, :].RH_actual.values
            if scale_isotherm and cycle<0: 
                iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::step, :].uptake.max()
                print(iso_scale)
            if scale_isotherm and cycle>0: 
                iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::step, :].uptake.max()
                print(iso_scale)
            '''
            popt, pcov = curve_fit(solving, time, uptake, p0=[ 0.02], #maxfev=15000,
                       bounds = (0, [10**3])
                       )
            R2_value = r2_score(solving(time  , *popt),  uptake)
            '''
            popt_list = []
            pcov_list = []
            R2_value_list = []
            for t_shift in range(1, t0_range*step, step):
                RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
                try:
                    popt, pcov = curve_fit(solving, time, uptake, p0=[ 0.3/Sorption.sample_mass], #maxfev=15000,
                                           bounds = (0, [10**3])
                                           )
                except:
                    popt_list.append(0)
                    pcov_list.append(0)
                    R2_value_list.append(0)
                    continue
                R2_value = r2_score(solving(time  , *popt),  uptake)
                popt_list.append(popt)
                pcov_list.append(pcov)
                R2_value_list.append(R2_value)
            max_ind = R2_value_list.index(max(R2_value_list))
            popt = popt_list[max_ind]
            pcov = pcov_list[max_ind]
            R2_value = R2_value_list[max_ind]
            t_shift = range(1, t0_range*step, step)[max_ind]
            RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
            ''''''
            if plot:
                if cycle<0:
                    arrax[ (np.abs(cycle)-2)%3, 1].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 1].plot(time, solving(time  , popt[0]), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_title('Desorption: cycle {0}, k: {1:.3f}, t0: {3:.2f} min, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, time[t_shift])) 
                    if (np.abs(cycle)-2)%3 == 2: 
                        suptitle = plt.suptitle('{0}, {1} mg, Humidity swing {2} - {3} % RH'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ),
                                    y=1.02
                                    )
                        plt.tight_layout()
                        plt.savefig('{0},{1}mg,Humidity_swing{2}-{3}RH.png'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ), bbox_extra_artists=(suptitle,), bbox_inches="tight")
                        plt.show()
                if cycle>0:
                    arrax[ (np.abs(cycle)-2)%3, 0].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 0].plot(time, solving(time  , popt[0]), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_title('Adsorption: cycle {0}, k: {1:.3f}, t0: {3:.2f} min, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, time[t_shift]))
            if verbose:
                print('des', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), popt, pcov, R2_value )
            df_result = df_result.append({'sample_mass':Sorption.data[Sorption.data.cycle_number == cycle].dry_mass.values.mean(),
                                          'experiment':Sorption.filename,
                                          'cycle': cycle, 
                                          'popt':popt[0], 
                                          'pcov':pcov[0][0],
                                          'R2': R2_value, 
                                          'RH_target':round(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 1),
                                          }, ignore_index=True)
            self.result = df_result
            df_result.to_csv(material+'.csv')
        try:
            suptitle = plt.suptitle('{0}, {1} mg, Humidity swing {2} - {3} % RH'.format(material, 
                         Sorption.sample_mass, 
                         Sorption.data[Sorption.data.cycle_number == cycle].RH_target.dropna().mean(), 
                         Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.dropna().mean()
                         ),
                        y=1.02
                        )
            plt.tight_layout()
            plt.savefig('{0},{1}mg,Humidity_swing{2}-{3}RH.png'.format(material, 
                         Sorption.sample_mass, 
                         Sorption.data[Sorption.data.cycle_number == cycle].RH_target.dropna().mean(), 
                         Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.dropna().mean()
                         ), bbox_extra_artists=(suptitle,), bbox_inches="tight")
            plt.show()
        except: pass
    def fit_model3(self, Sorption, Isotherm, verbose=False, plot=False, scale_isotherm = False):
        #### fit t0 ####
        iso_scale = 1
         
        def solving_ads(t, K):
            uptake_out = []
            B = uptake[0]
            t0=time[0]
            for i, t1 in enumerate(t):
                B = B + (t1-t0)*(np.multiply( K* (RH[i] - Isotherm.uptake_to_RH_ads(B*iso_scale)) , \
                                         RH[i]>Isotherm.uptake_to_RH_ads(B*iso_scale)))  
                uptake_out.append(B)
                t0=t1
            return uptake_out
        def solving_des(t,  K):
            uptake_out = []
            B = uptake[0]
            t0=time[0]
            for i, t1 in enumerate(t):
                B = B + (t1-t0)*(np.multiply( K* (RH[i] - Isotherm.uptake_to_RH_des(B*iso_scale)) , \
                                         RH[i]<Isotherm.uptake_to_RH_des(B*iso_scale)))  
                uptake_out.append(B)
                #print(B)
                t0=t1
            return uptake_out
        df_result = pd.DataFrame(columns=['sample_mass', 'cycle', 'popt', 'pcov', 'R2', 'RH_target'], dtype=object)    
        for cycle in Sorption.data.cycle_number.unique()[1:]:
            time = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].time.values
            time = time - time[0]
            uptake = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.values
            RH0 = Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].RH_actual.values
            if cycle<0:
                if scale_isotherm: 
                    iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.max()
                print(iso_scale)
                popt_list = []
                pcov_list = []
                R2_value_list = []
                for t_shift in range(1, 30, 1):
                    RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
                    popt, pcov = curve_fit(solving_des, time, uptake, p0=[ 0.02], #maxfev=15000,
                                           bounds = (0, [10**3])
                                           )
                    R2_value = r2_score(solving_des(time  , *popt),  uptake)
                    popt_list.append(popt)
                    pcov_list.append(pcov)
                    R2_value_list.append(R2_value)
                max_ind = R2_value_list.index(max(R2_value_list))
                popt = popt_list[max_ind]
                pcov = pcov_list[max_ind]
                R2_value = R2_value_list[max_ind]
                t_shift = range(1, 30, 1)[max_ind]
                RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
                if plot:
                    plt.scatter(time,  uptake)
                    plt.plot(time, solving_des(time  , popt[0]+np.sqrt(pcov[0][0])))
                    plt.plot(time, solving_des(time  , popt[0]-np.sqrt(pcov[0][0])))
                    plt.show()
                if verbose:
                    print('des', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), popt, pcov, R2_value )
            else:
                if scale_isotherm: 
                    iso_scale = Isotherm.RH_to_uptake_ads(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean())/Sorption.data[Sorption.data.cycle_number == cycle].iloc[::5, :].uptake.max()
                print(iso_scale)
                popt_list = []
                pcov_list = []
                R2_value_list = []
                for t_shift in range(1, 30, 1):
                    RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
                    popt, pcov = curve_fit(solving_ads, time, uptake, p0=[ 0.02], #maxfev=15000,
                                           bounds = (0, [10**3])
                                           )
                    R2_value = r2_score(solving_ads(time  , *popt),  uptake)
                    popt_list.append(popt)
                    pcov_list.append(pcov)
                    R2_value_list.append(R2_value)
                max_ind = R2_value_list.index(max(R2_value_list))
                popt = popt_list[max_ind]
                pcov = pcov_list[max_ind]
                R2_value = R2_value_list[max_ind]
                t_shift = range(1, 30, 1)[max_ind]
                RH = np.concatenate(([RH0[0] for k in range(t_shift)], RH0[:-t_shift]))
                if plot:
                    plt.scatter(time,  uptake)
                    plt.plot(time, solving_ads(time  , popt[0]+np.sqrt(pcov[0][0])))
                    plt.plot(time, solving_ads(time  , popt[0]-np.sqrt(pcov[0][0])))
                    plt.show()
                if verbose:
                    print('ads', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean() ,popt, pcov, R2_value )
            df_result = df_result.append({'sample_mass':Sorption.data[Sorption.data.cycle_number == cycle].dry_mass.values.mean(),
                                          'experiment':Sorption.filename,
                                          'cycle': cycle, 
                                          'popt':popt[0], 
                                          'pcov':pcov[0][0],
                                          'R2': R2_value, 
                                          'RH_target':round(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 1),
                                          }, ignore_index=True)
            self.result = df_result
    def fit_Finke(self, Sorption,  material, verbose=False, plot=False, scale_isotherm = False):
        print('Starting')
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import odeint
        from sklearn.metrics import r2_score
        def solving(t, ka1, ka2):
            '''
            ka1 = 0.02  # natural death percent (per day)
            ka2 = 0.001  # transmission percent  (per day)
            kd = 0.002  # resurect percent (per day)
            '''
            #global t
            #t=t-t0
            # solve the system dy/dt = f(y, t)
            def f(y, t):
                Ai = y[0]
                Bi = y[1]
                # the model equations (see Munz et al. 2009)
                f0 = -ka1*Ai**1 - ka2*Ai**1*Bi
                f1 =  ka1*Ai**1 + ka2*Ai**1*Bi 
                return [f0, f1]
            
            # initial conditions
            #A0 = 9.9598
            A0 = max(uptake)             # initial population
            B0 = 0                 # initial zombie population                 # initial death population
            y0 = [A0, B0]     # initial condition vector
            
            #t  = np.linspace(0, 39.26856, 2355)         # time grid
            
            # solve the DEs
            soln = odeint(f, y0, t)
            A = soln[:, 0]
            B = soln[:, 1]
            return B
        
        df_result = pd.DataFrame(columns=['sample_mass', 'cycle', 'popt', 'pcov', 'R2', 'RH_target'], dtype=object)    
        for cycle in [i for i in Sorption.data.cycle_number.unique()[1:] ]:
            if (cycle>0)&((cycle-2)%3 == 0): fig, arrax  = plt.subplots(3, 2, figsize = (10, 10))
            time = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[:, :].time.values
            uptake = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[:, :].uptake.values
            time = time - time[0]
            popt_list = []
            pcov_list = []
            R2_value_list = []
            #try:
            popt, pcov = curve_fit(solving, time, uptake, p0=[ 0.3/Sorption.sample_mass, 0.3/Sorption.sample_mass,], #maxfev=15000,
                                   #bounds = (0, [10**3])
                                   )
            print(popt)
            #except:
            #    popt_list.append(0)
            #    pcov_list.append(0)
            #    R2_value_list.append(0)
            #    continue
            R2_value = r2_score(solving(time  , *popt),  uptake)
            popt_list.append(popt)
            pcov_list.append(pcov)
            R2_value_list.append(R2_value)
                
            max_ind = R2_value_list.index(max(R2_value_list))
            popt = popt_list[max_ind]
            pcov = pcov_list[max_ind]
            R2_value = R2_value_list[max_ind]
            ''''''
            if plot:
                if cycle<0:
                    arrax[ (np.abs(cycle)-2)%3, 1].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 1].plot(time, solving(time  , *popt), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_title('Desorption: cycle {0}, k: {1:.3f}, k2: {3:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, popt[1],)) 
                    if (np.abs(cycle)-2)%3 == 2: 
                        suptitle = plt.suptitle('{0}, {1} mg, Humidity swing {2} - {3} % RH'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ),
                                    y=1.02
                                    )
                        plt.tight_layout()
                        plt.savefig('Finke{0},{1}mg,Humidity_swing{2}-{3}RH.png'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ), bbox_extra_artists=(suptitle,), bbox_inches="tight")
                        plt.show()
                if cycle>0:
                    arrax[ (np.abs(cycle)-2)%3, 0].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 0].plot(time, solving(time  ,  *popt), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_title('Adsorption: cycle {0}, k1: {1:.3f}, k2: {3:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, popt[1],))
            if verbose:
                print('des', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), popt, pcov, R2_value )
            df_result = df_result.append({'sample_mass':Sorption.data[Sorption.data.cycle_number == cycle].dry_mass.values.mean(),
                                          'experiment':Sorption.filename,
                                          'cycle': cycle, 
                                          'popt':popt[0], 
                                          'pcov':pcov[0][0],
                                          'R2': R2_value, 
                                          'RH_target':round(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 1),
                                          }, ignore_index=True)
            self.result = df_result
            df_result.to_csv(material+'.csv')

    def fit_first_order(self, Sorption,  material, verbose=False, plot=False, scale_isotherm = False):
        print('Starting')
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import odeint
        from sklearn.metrics import r2_score
        def solving(t, ka1, n):
            A0 = max(uptake) 
            B = A0*(1-np.exp(-(ka1*t)**n))
            return B
        
        df_result = pd.DataFrame(columns=['sample_mass', 'cycle', 'popt', 'pcov', 'R2', 'RH_target'], dtype=object)    
        for cycle in [i for i in Sorption.data.cycle_number.unique()[1:] ]:
            if (cycle>0)&((cycle-2)%3 == 0): fig, arrax  = plt.subplots(3, 2, figsize = (10, 10))
            time = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[:, :].time.values
            uptake = Sorption.data[(Sorption.data.cycle_number == cycle)].iloc[:, :].uptake.values
            time = time - time[0]
            popt_list = []
            pcov_list = []
            R2_value_list = []
            try:
                popt, pcov = curve_fit(solving, time, uptake, p0=[ 0.3/Sorption.sample_mass, 1,], #maxfev=15000,
                                   #bounds = (0, [10**3])
                                   )
                print(popt)
            except:
            #    popt_list.append(0)
            #    pcov_list.append(0)
            #    R2_value_list.append(0)
                continue
            R2_value = r2_score(solving(time  , *popt),  uptake)
            popt_list.append(popt)
            pcov_list.append(pcov)
            R2_value_list.append(R2_value)
                
            max_ind = R2_value_list.index(max(R2_value_list))
            popt = popt_list[max_ind]
            pcov = pcov_list[max_ind]
            R2_value = R2_value_list[max_ind]

            ''''''
            if plot:
                if cycle<0:
                    arrax[ (np.abs(cycle)-2)%3, 1].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 1].plot(time, solving(time  , *popt), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 1].set_title('Desorption: cycle {0}, k: {1:.3f}, n: {3:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, popt[1],)) 
                    if (np.abs(cycle)-2)%3 == 2: 
                        suptitle = plt.suptitle('{0}, {1} mg, Humidity swing {2} - {3} % RH'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ),
                                    y=1.02
                                    )
                        plt.tight_layout()
                        plt.savefig('FirstOrder{0},{1}mg,Humidity_swing{2}-{3}RH.png'.format(material, 
                                     Sorption.sample_mass, 
                                     Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 
                                     Sorption.data[Sorption.data.cycle_number == -cycle].RH_target.mean()
                                     ), bbox_extra_artists=(suptitle,), bbox_inches="tight")
                        plt.show()
                if cycle>0:
                    arrax[ (np.abs(cycle)-2)%3, 0].scatter(time,  uptake, c='tab:blue')
                    arrax[ (np.abs(cycle)-2)%3, 0].plot(time, solving(time  ,  *popt), c='tab:orange')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_xlabel('time, min')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_ylabel('uptake, wt.%')
                    arrax[ (np.abs(cycle)-2)%3, 0].set_title('Adsorption: cycle {0}, k: {1:.3f}, n: {3:.3f}, R2: {2:.4f}'.format((np.abs(cycle)-2)%3+1, popt[0], R2_value, popt[1],))
            if verbose:
                print('des', abs( cycle), Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), popt, pcov, R2_value )
            df_result = df_result.append({'sample_mass':Sorption.data[Sorption.data.cycle_number == cycle].dry_mass.values.mean(),
                                          'experiment':Sorption.filename,
                                          'cycle': cycle, 
                                          'popt':popt[0], 
                                          'pcov':pcov[0][0],
                                          'R2': R2_value, 
                                          'RH_target':round(Sorption.data[Sorption.data.cycle_number == cycle].RH_target.mean(), 1),
                                          }, ignore_index=True)
            self.result = df_result
            df_result.to_csv(material+'.csv')

            
    def working_capacity_exp(self, Sorption, Isotherm, verbose=False, plot=False):
        self.zero_mass = Sorption.data[Sorption.data.cycle_number == 1].mass.max()/(Isotherm.RH_to_uptake_ads(30)+100)*100
        print(self.zero_mass)
        self.wc = []
        for cycle in Sorption.data[Sorption.data.cycle_number>0].cycle_number.unique():
            t_ads = Sorption.data[Sorption.data.cycle_number==cycle].time.max()-Sorption.data[Sorption.data.cycle_number==cycle].time.min()
            t_des = Sorption.data[Sorption.data.cycle_number==-cycle].time.max()-Sorption.data[Sorption.data.cycle_number==-cycle].time.min()
            mass_in_cycle = Sorption.data[Sorption.data.cycle_number==cycle].mass.max()-Sorption.data[Sorption.data.cycle_number==-cycle].mass.min()
            wc = mass_in_cycle/self.zero_mass/(t_ads + t_des)*100 ## wt.% per min
            self.wc.append([wc, t_ads, t_des])
        print(self.wc[-2])

