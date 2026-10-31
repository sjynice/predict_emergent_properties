import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

def classify_abundance(arr):
    """
    Determine whether a species abundance time series is steady, oscillatory, or unstable.
    Input arr is a 1D array (e.g., Abun[0,0,:,0]), returns [state, val1, val2].
    """

    # 1. Steady-state detection: check std and range of the last xxx values
    if arr.size >= 60:
        last5 = arr[:60]
        std_last5 = np.std(last5)      # Std dev
        range_last5 = np.max(last5) - np.min(last5)
        if std_last5 < 0.01 and range_last5 < 0.02:
            # print(last5)
            mean_val = np.mean(last5)  # Mean
            return [int(0), float(mean_val), float(mean_val)]

    # 2. Oscillation detection (at least two peaks and valleys + amplitude threshold)
    peaks, _ = find_peaks(arr, prominence=0.01)
    valleys, _ = find_peaks(-arr, prominence=0.01)
    if len(peaks) >= 2 and len(valleys) >= 2:
        max_val, min_val = np.max(arr), np.min(arr)
        amplitude = max_val - min_val
        if amplitude > 0.05:  # Amplitude threshold
            # 3. Check if oscillation is stable (whether amplitudes of the last few peak/valley pairs stabilize)
            num_check = min(3, len(peaks), len(valleys))  # Check the last 3 periods
            if num_check >= 2:
                # Take values of the last few peaks and valleys
                last_peaks = arr[peaks[:num_check]]
                last_valleys = arr[valleys[:num_check]]
                # Compute amplitude after alignment
                last_amplitudes = last_peaks[:num_check] - last_valleys[:num_check]
                if np.std(last_amplitudes) < 0.01:
                    # Peaks and valleys are stable; return peak mean and valley mean
                    peak_mean = np.mean(last_peaks)
                    valley_mean = np.mean(last_valleys)
                    return [int(1), float(peak_mean), float(valley_mean)]  # Stable oscillation
                else:
                    # print(last_peaks)
                    # print('\n')
                    # print(last_valleys)
                    return [int(2), None, None]  # Unstable oscillation

    # 4. Not yet stable (no pattern or does not match the previous two cases)
    return [int(3), None, None]

def Calculate_Index(repeat, Abun, thresh1):
    Index=np.zeros(repeat)
    Index[0]=1
    for j in np.arange(1,repeat,1):
        tempj=Abun[j,:]
        pin=0
        for k in range(j):
            tempk=Abun[k,:]
            tttt=tempj-tempk
            dis=np.sqrt(sum(np.multiply(tttt,tttt)))
            # If the distance is less than the threshold, Index[j] is set to Index[k]
            if dis<thresh1:
                Index[j]=Index[k]
                pin=1
                break
        # If no matching steady state is found, a new steady state index is assigned
        if pin==0:
            Index[j]=max(Index)+1 
    return Index

def Calculate_StableStates(NumSpecies, Index, Abun, thresh2):
    NumStab=max(Index)
    CoreSpecies = np.zeros(NumSpecies)
    if NumStab>1:
        for spc in range(NumSpecies):
            if np.max(Abun[:,spc]) - np.min(Abun[:,spc]) > thresh2:
                CoreSpecies[spc]=1
    return NumStab, CoreSpecies

def Calculate_Proportion(Index):
    AbunTot=np.zeros(int(np.max(Index)))
    for rgn in range(int(max(Index))):
        # The number of initializations that evolved into the (rgn + 1)-th steady state
        AbunTot[rgn]=np.sum(Index==rgn+1)
    AbunTot=np.divide(AbunTot,sum(AbunTot))
    return AbunTot

def Calculate_Entropy(NumSpecies, AbunTot):
    if sum(AbunTot)==0:
        MetaDiver=0
    else:
        p = AbunTot / sum(AbunTot)
        H = -np.sum(p * np.log(p, where=(p>0)))
        MetaDiver = (np.exp(H)-1)/(NumSpecies-1) # Normalize the range to 0-1
    return MetaDiver

def Calculate_Diversity(NumSpecies, Index, Abun, AbunTot):
    DiverStab=np.zeros(int(np.max(Index)))
    for asd in range(int(np.max(Index))):
        # asd means the asd'th stable state
        LabelStable_asd=np.where(Index==asd+1)[0][0]
        AbunSpe=Abun[LabelStable_asd,:]
        if sum(AbunSpe)==0:
            AbunSpe=np.zeros(NumSpecies)
            DiverStab[asd]=0
        else:
            p = AbunSpe / AbunSpe.sum()
            H = -np.sum(p * np.log(p, where=(p>0)))
            DiverStab[asd] = np.exp(H) # Range from 1 to NumSpecies
    Diversity=(np.dot(DiverStab,AbunTot)-1)/(NumSpecies-1) # Calculate mathematical expectation and normalize the range to 0-1
    return Diversity

def Calculate_MeanAbundance(NumSpecies, repeat, Abun):
    # Negative abundance values are not clipped
    MeanAbun = np.zeros(NumSpecies)
    RelativeAbun = np.zeros((repeat, NumSpecies))
    for hjk in range(repeat):
        if np.sum(Abun[hjk, :]) == 0:
            RelativeAbun[hjk, :] = np.zeros(NumSpecies)
        else:
            RelativeAbun[hjk, :] = Abun[hjk, :] / np.sum(Abun[hjk, :])
    for zxc in range(NumSpecies):
        MeanAbun[zxc] = np.mean(RelativeAbun[:, zxc])
    return MeanAbun

def Calculate_MeanAbundance2(NumSpecies, Abun):
    # Negative abundance values are clipped to some extent
    MeanAbun = np.zeros(NumSpecies)
    for spc in range(NumSpecies):
        MeanAbun[spc] = np.sum(Abun[:, spc])
    if np.sum(MeanAbun) > 0:
        MeanAbun = MeanAbun / np.sum(MeanAbun)
    return MeanAbun

def Get_EmergentProperties(repeat, Abun, NumSpecies, thresh1, thresh2):

    Index = Calculate_Index(repeat, Abun, thresh1)
    
    # 1.The number of steady states, and the core species in stable states.
    NumStab, CoreSpecies = Calculate_StableStates(NumSpecies, Index, Abun, thresh2)
    # 2.The entropy of stable state
    AbunTot = Calculate_Proportion(Index)
    MetaDiver = Calculate_Entropy(NumSpecies, AbunTot)

    # # 3.The Diversity of each Stable
    # Diversity = Calculate_Diversity(NumSpecies, Index, Abun, AbunTot)
    # # 4.The mean relative abundance of each species
    # MeanAbun = Calculate_MeanAbundance(NumSpecies, repeat, Abun)

    # 3.The mean relative abundance of each species
    MeanAbun = Calculate_MeanAbundance2(NumSpecies, Abun)
    # 4.The Diversity
    Diversity = Calculate_Entropy(NumSpecies, MeanAbun)

    return NumStab, MetaDiver, Diversity, MeanAbun, CoreSpecies

def Preliminary_classification(NumSpecies, datasize, repeat, Abun):
    results = np.zeros((datasize, repeat, NumSpecies, 3), dtype=float)
    probability = np.zeros((datasize, 2), dtype=float)
    outlier = []
    sta_sta = []
    for i in range(datasize):
        temp0 = 0
        temp1 = 0
        sta_sta.append([])
        for j in range(repeat):
            temp2 = 0
            temp3 = 0
            for k in range(NumSpecies):
                results[i, j, k, :] = classify_abundance(Abun[i, j, :, k])
                if results[i, j, k, 0] == 1 or results[i, j, k, 0] == 2:
                    temp2 = 1
                elif results[i, j, k, 0] == 3:
                    temp3 = 1
            if temp2 == 0 and temp3 == 0:
                sta_sta[i].append(j)
            if temp2 == 1:
                temp0 += 1
            elif temp3 == 1:
                temp1 += 1
        if temp1 != 0:
            outlier.append(i)

        probability[i, 0] = temp0 / repeat
        probability[i, 1] = temp1 / repeat

        Abun_mean = Abun.mean(axis=1)
        survival = (Abun_mean > 0.001).any(axis=1).astype(int)
        survival_fraction = survival.mean(axis=1)

    return results, probability, outlier, sta_sta, survival_fraction

def Further_Classification(NumSpecies, datasize, sta_sta, results, thresh1=0.05, thresh2=0.05):
    NumStab=np.zeros(datasize, dtype=int)
    MetaDiver=np.zeros(datasize)
    Diversity=np.zeros(datasize)
    MeanAbun=np.zeros((datasize, NumSpecies))
    CoreSpecies=np.zeros((datasize, NumSpecies))

    for wer in range(datasize):
        Abun2 = []
        if sta_sta[wer]: # If the steady-state sample list is not empty
            for jkl in sta_sta[wer]:
                single = []
                for spc in range(NumSpecies):
                    single.append(results[wer, jkl, spc, 1])
                Abun2.append(single)
            NumStab[wer], MetaDiver[wer], Diversity[wer], MeanAbun[wer], CoreSpecies[wer] = Get_EmergentProperties(len(sta_sta[wer]), np.array(Abun2), NumSpecies, thresh1, thresh2)
    return NumStab, MetaDiver, Diversity, MeanAbun, CoreSpecies


def predict_numerical(Xnum, NumSpecies, repeat, timespan):
    '''
    Using unpadded Xnum 
    '''

    Xnum = np.atleast_2d(Xnum)
    print(f"Dealing with richness {NumSpecies}")
    datasize = Xnum.shape[0]
    featgam = NumSpecies**2 - NumSpecies # Number of gamma features
    delta = 0.2*np.ones(NumSpecies)

    # Sample every 2s for 300s; in sol.y this is a step size of 20, sampling from the last 3000 rows.
    stepsize = 20
    total = 3000
    sample_net = np.arange(-1, -total-1, -stepsize).astype(int)
    sampling = len(sample_net)

    Abun = np.zeros((datasize,repeat,sampling,NumSpecies))
    Initial = np.zeros((datasize, repeat, NumSpecies))    

    for wer in range(datasize):

        # mu
        mu = Xnum[wer, 0:NumSpecies]

        # Generation of the gamma matrix
        gamma = np.zeros((NumSpecies,NumSpecies))
        indext = NumSpecies
        for i in range(NumSpecies):
            for j in range(NumSpecies):
                if i==j:
                    gamma[i,j] = -1
                else:
                    gamma[i,j] = Xnum[wer, indext]
                    indext+=1

        # D
        D = Xnum[wer, (NumSpecies+featgam):(NumSpecies+featgam+NumSpecies)]

        def MultiSpecies(t,y):
            thresh_y = 0
            dydt=np.zeros(NumSpecies)
            s=y
            s[s < thresh_y] = 0
            for i in range(NumSpecies):
                sumy_neg=0
                sumy_pos=0
                for j in range(NumSpecies):
                    if gamma[i,j]<0:
                        sumy_neg=sumy_neg+gamma[i,j]*s[j]
                    if gamma[i,j]>0:
                        sumy_pos=sumy_pos+gamma[i,j]*s[j]
                dydt[i]=mu[i]*s[i]*(1+sumy_neg+delta[i]*sumy_pos/(1+sumy_pos))+D[i]
            return dydt
        
        for hjk in range(repeat):
            print(wer*repeat + hjk) # Record the program's running progress
            initial = np.random.rand(NumSpecies)/NumSpecies
            # for adjust code use
            Initial[wer,hjk,:]=initial

            sol=solve_ivp(MultiSpecies,[0,max(timespan)],initial,method='RK45',t_eval=timespan)
            y=np.transpose(sol.y)
            t=sol.t
            index=0
            for i in sample_net:
                temp=y[i,range(NumSpecies)]
                Abun[wer,hjk,index,:]=temp
                index+=1

    results, probability, outlier, sta_sta, survival_fraction = Preliminary_classification(NumSpecies, datasize, repeat, Abun) # Get the probability of oscillation/outlier for each sample, and record outlier sample indices and (sample, initialization) indices of steady-state samples
    NumStab, MetaDiver, Diversity, MeanAbun, CoreSpecies = Further_Classification(NumSpecies, datasize, sta_sta, results, thresh1=0.05, thresh2=0.05)
    ynum = np.round(NumStab)
    Xynum = np.hstack((Xnum, ynum.reshape(-1,1), MetaDiver.reshape(-1,1), Diversity.reshape(-1,1), MeanAbun, CoreSpecies, probability, survival_fraction.reshape(-1,1)))
    #Xynum = np.hstack((Xnum, ynum.reshape(-1,1), MetaDiver.reshape(-1,1), Diversity.reshape(-1,1), MeanAbun, CoreSpecies, probability))
    print("\n")
    return Xynum