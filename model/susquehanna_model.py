import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import utils
from rbf import RBF
from scipy import constants

class susquehannaModel():

    # TODO: Class description

    GG = constants.value(u'standard acceleration of gravity')
    freeboard = 8.8 # 118.0 - 109.2 #TODO: Get source for this
    reliability = 0.9
    gammaH20 = 1000.0

    def __init__(self, l0, l0_MR, d0, n_years):
        self.l0 = l0
        self.l0_MR = l0_MR
        self.d0 = d0

    #variables from the header file        
        self.input_min = []
        self.input_max = []
        self.output_max = []
        # self.PolicySim = ''

        self.n_days_in_year = 365   
        self.n_years = n_years # historical record #1000 simulation horizon (1996,2001)
        self.time_horizon_H = self.n_days_in_year*self.n_years
        self.dec_step = 4 # 4-hours decisional time step
        self.day_fraction = (int) (24 / self.dec_step)
        # initial condition
        self.init_level = l0  # feet
        # initial condition in muddy run
        self.init_level_MR = self.l0_MR
        self.day0 = self.d0

        # TODO: Find sources and document if this is aggregated 
        # TODO: Link downstream releases to flood risk

        # 20 of the 53 gates of the dam were opened when the water levels last peaked
        self.n_gates = 20 
        self.capacity_per_gate= 16000 #capacity per gate (cfs)
        self.hFlood = 109.2  # critical level for dam in ft [#alternate source 108.5]
        # TODO: Sources
        self.rFlood= self.n_gates*self.capacity_per_gate #critical discharge for evacuation (cfs) [alternative source :156136.61 ]
        self.n_days_one_year = 1*365

        # Constraints for the reservoir
        self.min_level_chester = 99.8 #ft of water
        self.min_level_app = 98.5 #ft of water
        self.min_level_baltimore = 90.8 #ft of water
        self.min_level_conowingo = 100.5 #ft
        self.critical_level_app = 103.5 #ft
        self.min_level_MR = 104.0 #ft
        self.summer_recreation_level = 106.5 #ft
        self.min_level_BCD = 91.5 # minimum level for Baltimore, Chester and Downstream

    def load_data(self):
        raw_data_path = "data/raw"
        #n_days_one_year = 1*365
        # Conowingo characteristics
        self.lsv_rel = utils.loadMatrix(raw_data_path + "./dataMC/lsv_rel_Conowingo.txt",3,10)         # level (ft) - Surface (acre) - storage (acre-feet) relationships
        self.turbines = utils.loadMatrix(raw_data_path + "./dataMC/turbines_Conowingo2.txt",3,13)       # Max-min capacity (cfs) - efficiency of Conowingo plant turbines
        self.tailwater = utils.loadMatrix(raw_data_path + "./dataMC/tailwater.txt",2,18)               # tailwater head (ft) - release flow (cfs)
        self.spillways = utils.loadMatrix(raw_data_path + "./dataMC/spillways_Conowingo.txt",3,8) # substitute with newConowingo1      # level (ft) - max release (cfs) - min release (cfs) for level > 108 ft

        self.gate0 = utils.loadMatrix(raw_data_path + "./dataMC/gates0.txt",2,2)
        self.gate1 = utils.loadMatrix(raw_data_path + "./dataMC/gates1.txt",2,2)
        self.gate2 = utils.loadMatrix(raw_data_path + "./dataMC/gates2.txt",2,2)
        self.gate3 = utils.loadMatrix(raw_data_path + "./dataMC/gates3.txt",2,2)
        self.gate4 = utils.loadMatrix(raw_data_path + "./dataMC/gates4.txt",2,2)
        self.gate5 = utils.loadMatrix(raw_data_path + "./dataMC/gates5.txt",2,2)

        # Muddy Run characteristics
        self.lsv_rel_Muddy = utils.loadMatrix(raw_data_path + "./dataMC/lsv_rel_Muddy.txt",3,38)       # level (ft) - Surface (acre) - storage (acre-feet) relationships
        self.turbines_Muddy = utils.loadVector(raw_data_path + "./dataMC/turbines_Muddy.txt",4)        # Turbine-Pumping capacity (cfs) - efficiency of Muddy Run plant (equal for the 8 units)


        # objectives historical
        self.energy_prices = utils.loadArrangeMatrix(raw_data_path + "./dataMC/Pavg99.txt", 24, self.n_days_one_year)         # energy prices ($/MWh)
        self.min_flow = utils.loadVector(raw_data_path + "./dataMC/min_flow_req.txt", self.n_days_one_year)         # FERC minimum flow requirements for 1 year (cfs)
        self.h_ref_rec = utils.loadVector(raw_data_path + "./dataMC/h_rec99.txt", self.n_days_one_year)               # target level for weekends in touristic season (ft)
        self.w_baltimore = utils.loadVector(raw_data_path + "./dataMC/wBaltimore.txt", self.n_days_one_year)        # water demand of Baltimore (cfs)
        self.w_chester   = utils.loadVector(raw_data_path + "./dataMC/wChester.txt", self.n_days_one_year)          # water demand of Chester (cfs)
        self.w_atomic    = utils.loadVector(raw_data_path + "./dataMC/wAtomic.txt", self.n_days_one_year)           # water demand for cooling the atomic power plant (cfs)

    #historical
        N_samples = self.n_days_one_year * self.n_years
        self.evap_CO_MC= utils.loadVector("./data_historical/vectors/evapCO_history.txt",N_samples)         # evaporation losses (inches per day)
        self.inflow_MC = utils.loadVector("./data_historical/vectors/MariettaFlows_history.txt",N_samples)   # inflow, i.e. flows at Marietta (cfs)
        self.inflowLat_MC = utils.loadVector("./data_historical/vectors/nLat_history.txt",N_samples)         # lateral inflows from Marietta to Conowingo (cfs)
        self.evap_Muddy_MC = utils.loadVector("./data_historical/vectors/evapMR_history.txt",N_samples)      # evaporation losses (inches per day)
        self.inflow_Muddy_MC = utils.loadVector("./data_historical/vectors/nMR_history.txt",N_samples) 

    # Improvement: Identify them as model uncertainties and randomly sample them 
        # self.evap_CO_MC = utils.loadMatrix(raw_data_path + "./dataMC/evapCO_MC.txt",self.n_years, self.n_days_one_year)         # evaporation losses (inches per day)
        # self.inflow_MC = utils.loadMatrix(raw_data_path + "./dataMC/MariettaFlows_MC.txt",self.n_years, self.n_days_one_year)   # inflow, i.e. flows at Marietta (cfs)
        # self.inflowLat_MC = utils.loadMatrix(raw_data_path + "./dataMC/nLat_MC.txt",self.n_years, self.n_days_one_year)         # lateral inflows from Marietta to Conowingo (cfs)
        # self.evap_Muddy_MC = utils.loadMatrix(raw_data_path + "./dataMC/evapMR_MC.txt",self.n_years, self.n_days_one_year)      # evaporation losses (inches per day)
        # self.inflow_Muddy_MC = utils.loadMatrix(raw_data_path + "./dataMC/nMR_MC.txt",self.n_years, self.n_days_one_year) 

    # Q: What is this used for?
    # def setPolicysim(self, newPolicySim):
    #     self.PolicySim = newPolicySim

    def setRBF(self, RBFs, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.RBFs = RBFs


    def RBFs_policy(self, control_law, input):
        input1 = []
        for i in range(0, self.inputs):
            
            input1.append((input[i] - self.input_min[i]) / (self.input_max[i] - self.input_min[i]))

        # RBF
        u = []
        u = control_law.rbf_control_law(input1)


        # de-normalization 
        # Q: What is denormalization? Why do we do it?
        uu = []
        for i in range(0,self.outputs):
            uu.append(u[i] * self.output_max[i])
        
        return uu

    def storageToLevel(self, s, lake):
        # s : storage 
        # lake : which lake it is at

        s_ = utils.cubicFeetToAcreFeet(s)
        if lake ==0:
           h = utils.interpolate_linear(self.lsv_rel_Muddy[2], self.lsv_rel_Muddy[0], s_)
        else:
            h = utils.interpolate_linear(self.lsv_rel[2], self.lsv_rel[0], s_)
        return h
    
    def levelToStorage(self, h, lake):
        if lake ==0:
            s = utils.interpolate_linear(self.lsv_rel_Muddy[0], self.lsv_rel_Muddy[2], h)
        else:
            s = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[2], h)
 
        return utils.acreFeetToCubicFeet(s)
    
    def levelToSurface(self, h, lake):
        if lake ==0:
            S = utils.interpolate_linear(self.lsv_rel_Muddy[0], self.lsv_rel_Muddy[1], h)
        else:
            S = utils.interpolate_linear(self.lsv_rel[0], self.lsv_rel[1], h)
        return utils.acreToSquaredFeet(S)
    
    def tailwater_level(self, q):
        l = utils.interpolate_linear(self.tailwater[0], self.tailwater[1], q)
        return l

    def muddyRunPumpTurb(self, day, hour, level_Co, level_MR) :
         # Determines the pumping and turbine release volumes in a day based on the hour and day of week for muddy run
        # Q: What is turbine volume?
        # fixed rule
        pt = []
        QP = 24800.0 #cfs # TODO: Sources for this
        QT = 32000.0 #cfs # TODO: Sources
        # Q: How did we get 470?
        qM = (self.levelToStorage(level_MR, 0) - self.levelToStorage(470.0,0))/3600 # active storage = sMR - deadStorage

        qp = 0.0
        qt = 0.0
        # Q: what is qp?
        # Q: Do we need the break statements here?
        if day == 0: #sunday
            if hour<5 or hour >=22:
                qp = QP
        #break
        elif day >= 1 and day <= 4: # monday to thursday
            if hour <=6 or hour >=21:
                qp = QP
            if (hour>=7 and hour <=11) or (hour>=17 and hour <=20):
                qt = min(QT, qM)

        #break
        elif day == 5: #friday
            if (hour>=7 and hour <=11) or (hour>=17 and hour <=20):
                qt = min(QT, qM)
        #break
        elif day ==6: #saturday
            if hour <=6 or hour >=22:
                qp = QP
        #break

        # water pumping stops to Muddy RUn beyond this point. 
        # However, according to the conowingo authorities 800 cfs will be released as emergency credits in order to keep the facilities from running
        if level_Co < self.min_level_MR: # if TRUE cavitation problems in pumping
            qp = 0.0
        
        if level_MR < 470.0:
            qt = 0.0
        
        pt.append(qp)
        # Turbine release
        pt.append(qt)

        return qp, qt

    def actual_release(self, uu, level_Co, day_of_year, n_sim, n_lat, evaporation_losses_Co, q_pump, q_rel, leak):
        # Check if it doesn't exceed the spillway capacity
        Tcap = 85412 # total turbine capacity (cfs)

        # TODO: Check where MaxSpill was supposed to be used. WHy is it not used now?
        maxSpill = 1242857.0 # total spillway combined (cfs)
        sim_step = 3600
       
        # minimum discharge values at APP, Balitomore, Chester and downstream
        qm_A = 0.0
        qm_B = 0.0
        qm_C = 0.0
        qm_D = 0.0

        # Additions from Kenji's model
        qM_A = self.w_atomic[day_of_year]
        qM_B = self.w_baltimore[day_of_year]
        qM_C = self.w_chester[day_of_year]
        qM_D = Tcap

        # 106 is close to the summer recreation level (106.5)
        s_crest = self.levelToStorage(106.0,1)
        s = self.levelToStorage(level_Co,1)

        if(level_Co<105.0):
            qM_D = 0.0

        if level_Co<self.min_level_conowingo:
            qM_C = 0.0
            qM_D = 0.0

        if level_Co< self.min_level_BCD:
            print("enters this condition")
            qM_B = 0.0
            qM_C = 0.0
            qM_D = 0.0

        #print(qM_D)
        # Activating spillways



        if level_Co > 109.2 and level_Co <110.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate0[0], self.gate0[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate0[0], self.gate0[1], level_Co) + Tcap # change to spillways[2]

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co > 110.0 and level_Co <112.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate1[0], self.gate1[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate1[0], self.gate1[1], level_Co) + Tcap # change to spillways[2]

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co > 112.0 and level_Co <114.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate2[0], self.gate2[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate2[0], self.gate2[1], level_Co) + Tcap # change to spillways[2]        

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co > 114.0 and level_Co <116.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate3[0], self.gate3[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate3[0], self.gate3[1], level_Co) + Tcap # change to spillways[2]        

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co > 116.0 and level_Co <118.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate4[0], self.gate4[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate4[0], self.gate4[1], level_Co) + Tcap # change to spillways[2]        

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co > 118.0 and level_Co <120.0:
            # spillways activated
            # Reservoir Decision Maker would typically take a call to activate spillways when the
            qM_D = utils.interpolate_linear(self.gate5[0], self.gate5[1], level_Co) + Tcap # Turbine capacity + spillways
            qm_D = utils.interpolate_linear(self.gate5[0], self.gate5[1], level_Co) + Tcap # change to spillways[2]        

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  

        if level_Co>=120.0:
            qM_D = maxSpill + Tcap
            qm_D = maxSpill + Tcap

            s_temp = s + sim_step*(n_sim - n_lat - evaporation_losses_Co - Tcap - q_pump + q_rel - leak - qM_A - qM_B - qM_C )
            s_temp2 = s_temp - sim_step*qM_D

            # Checking if downstream release is less than the summer recreation level height
            if s_temp2 < s_crest:
                qM_D = max((s_temp-s_crest)/sim_step+Tcap, Tcap)
                qm_D = Tcap  


        


        #print(s_temp2, level_Co)      


        # Physical contraints from old version of the model

        # # maximum discharge values. The max discharge can be as much as the demand in that area
        # if level_Co <= self.min_level_app:
        #     qM_A = 0.0
        # else:
        #     qM_A = self.w_atomic[day_of_year]
        
        # # if level_Co <= self.min_level_baltimore:
        # #     qM_B = 0.0
        # # else:
        # #     qM_B = self.w_baltimore[day_of_year]

        # if level_Co <= self.min_level_chester:
        #     qM_C = 0.0
        # else:
        #     qM_C = self.w_chester[day_of_year]

        # qM_B = self.w_baltimore[day_of_year]
        # qM_D = Tcap


        #NOTE: Kenji's model does not use this. Why?
        # getting the smallest of the demand and what the RBF policy returns
        # rr.append(min(qM_A, max(qm_A, uu[0])))
        # rr.append(min(qM_B, max(qm_B, uu[1])))
        # rr.append(min(qM_C, max(qm_C, uu[2])))
        # rr.append(min(qM_D, max(qm_D, uu[3])))

        #print(qM_D)
        #print(min(qM_D, max(qm_D, uu[0])))

        return min(qM_A, max(qm_A, uu[0])), min(qM_B, max(qm_B, uu[1])), min(qM_C, max(qm_C, uu[2])), min(qM_D, max(qm_D, uu[0]))

    def g_hydRevCo(self, r, h, day_of_year, hour0):
        Nturb = 13
        g_hyd = []
        g_rev = []
        pp = []
        c_hour = len(r)*hour0

        #print(r, "r length")

        for i in range(0, len(r)):
            deltaH = h[i] - self.tailwater_level(r[i])
            q_split = r[i]

            for j in range(0, Nturb):
                if q_split < self.turbines[1][j]:
                    qturb = 0.0
                elif q_split > self.turbines[0][j]:
                    qturb = self.turbines[0][j]
                else :
                    qturb = q_split
                q_split = q_split - qturb

                p = 0.79* self.GG * self.gammaH20 * utils.cubicFeetToCubicMeters(qturb) * utils.feetToMeters(deltaH) * 3600 / (3600*1000)
                pp.append(p)
            
            #print(len(pp), "pp length")
            g_hyd.append(sum(pp))            
            g_rev.append(sum(pp)/1000*self.energy_prices[c_hour][day_of_year])
            pp.clear()
            c_hour = c_hour + 1

        Gp = sum(g_hyd)
        Gr = sum(g_rev)

        return Gp, Gr

    def g_hydRevMR(self, qp, qr, hCo, hMR, day_of_year, hour0):
        Nturb = 8
        g_hyd = []
        g_pump = []
        g_rev = []
        g_revP = []
        #G = []
        c_hour = len(qp) * hour0
        pT = 0.0
        pP = 0.0

        for i in range(0, len(qp)):
            # net head
            deltaH = hMR[i] - hCo[i]
            # 8 turbines
            qp_split = qp[i] 
            qr_split = qr[i]

            # TODO: Vectorize this part?
            for j in range(0,Nturb):
                if qp_split < 0.0:
                    qpump = 0.0
                elif qp_split > self.turbines_Muddy[2]:
                    qpump = self.turbines_Muddy[2]
                else: 
                    qpump = qp_split

                p_ = self.turbines_Muddy[3] * self.GG * self.gammaH20 * utils.cubicFeetToCubicMeters(qpump) * utils.feetToMeters(deltaH) * 3600 / (3600 * 1000) # KWh/h
                pP = pP + p_
                qp_split = qp_split - qpump

                if qr_split < 0.0:
                    qturb = 0.0
                elif qr_split > self.turbines_Muddy[0]:
                    qturb = self.turbines_Muddy[0]
                else:
                    qturb = qr_split

                p = self.turbines_Muddy[1] * self.GG * self.gammaH20 * utils.cubicFeetToCubicMeters(qturb) * utils.feetToMeters(deltaH) * 3600 / (3600 * 1000) # kWh/h
                pT = pT + p
                qr_split = qr_split - qturb

            g_pump.append(pP)
            g_revP.append(pP / 1000 * self.energy_prices[c_hour][day_of_year])
            pP = 0.0
            g_hyd.append(pT)
            g_rev.append(pT / 1000 * self.energy_prices[c_hour][day_of_year])
            pT = 0.0
            c_hour = c_hour + 1

        return g_pump, g_hyd, g_revP, g_rev

    def g_Flood(self,h, hFlood):
        g = []
        #G = 0.0

        for i in range(0,len(h)):
            if h[i]>=hFlood:
                g.append(h[i]-hFlood)
            else:
                g.append(0.0)
        return max(g)
    
    def g_FloodDuration(self,flood):

        count = 0
        max_flood_duration = 0

        for i in range(0, len(flood)):
            if (flood[i]==0):
                count = 0
            else:
                count+=1
                max_flood_duration = max(max_flood_duration, count)
        
        # G = 0.0
        # fl_start = []
        # fl_end = []
        
        # for i in range(0, len(flood)-1):
        #     try:
        #         d = flood[i+1] - flood[i]
        #     except:
        #         print(i, " is out of length of flood")
        #         raise IndexError()
        #     if i == len(flood)-2 and d == 0:
        #         fl_end.append(len(flood))
        #     if d ==1:
        #         # Q: Shouldn't this be i+1?
        #         # Q: What happens if the last two time steps were flooded?
        #         fl_start.append(i)
        #     if d==-1:
        #         fl_end.append(i)
            
        
        # fl_length = []
        # if len(fl_start) and len(fl_end) > 0:
        #     for j in range(0, len(fl_start)):
        #         try:
        #             ev = fl_end[j] - fl_start[j]
        #         except:
        #             print(j, " is out of length of the array")
        #             print(flood, " flood")
        #             fl_length.append(0.0)
        #             #raise IndexError()
        #         fl_length.append(ev)
        # else:
        #     fl_length.append(0.0)
        
        # # This should be the decision step?
        # G = max(fl_length)/self.day_fraction

        return max_flood_duration/self.day_fraction


    def res_transition_h(self, s0, uu, n_sim, n_lat, ev, s0_mr, n_sim_mr, ev_mr, day_of_year, day_of_week, hour0):

        HH = self.dec_step
        # Q: What is s?
        sim_step = 3600 # s/hour 
        # Q: What is this?
        leak = 800 #cfs
        Tcap = 85412

        # Storages and levels of Conowingo and Muddy Run
        storage_Co = [-999.0]*(HH + 1)
        level_Co = [-999.0]*(HH + 1)
        storage_mr = [-999.0]*(HH + 1)
        level_mr = [-999.0]*(HH + 1)

        #Actual releases (Atomic Power plant, Baltimore, Chester, Dowstream)
        release_A = [-999.0]*(HH)
        release_B = [-999.0]*(HH)
        release_C = [-999.0]*(HH)
        release_D = [-999.0]*(HH)
        q_pump = [-999.0]*(HH)
        q_rel = [-999.0]*(HH)
        s_rr = []
        #pt_muddy = []
        rr = []

        # initial conditions
        storage_Co[0] = s0
        #if s0<0:
        #print(s0, day_of_year, HH)
        #print(s0, "s0")

        storage_mr[0] = s0_mr
        c_hour = hour0*HH


        # Check: Should this be HH or HH-1?
        for i in range(0, HH):

            level_Co[i] = self.storageToLevel(storage_Co[i], 1)

            level_mr[i] = self.storageToLevel(storage_mr[i], 0)

            # Muddy Run operation
            q_pump[i], q_rel[i] = self.muddyRunPumpTurb(day_of_week, int(c_hour), level_Co[i], level_mr[i])

            # Compute surface level and evaporation losses
            #NOTE: This is placed after evaporation_losses_Co in Kenji's model which means in the first instance evap losses will be zero
            surface_Co = self.levelToSurface(level_Co[i], 1)
            surface_MR = self.levelToSurface(level_mr[i], 0)
            evaporation_losses_MR = utils.inchesToFeet(ev_mr) * surface_MR / 86400            
            evaporation_losses_Co = utils.inchesToFeet(ev) * surface_Co / 86400

            # Compute actual release 
            rr = self.actual_release(uu, level_Co[i], day_of_year, n_sim, n_lat, evaporation_losses_Co, q_pump[i], q_rel[i], leak)

            release_A[i] = rr[0]
            release_B[i] = rr[1]
            release_C[i] = rr[2]
            release_D[i] = rr[3]
            # Q: Why is this being added?
            WS = release_A[i] + release_B[i] + release_C[i]

            


            # System Transition
            storage_mr[i+1] = storage_mr[i] + sim_step*(q_pump[i] - q_rel[i] + n_sim_mr - evaporation_losses_MR)
            storage_Co[i+1] = storage_Co[i] + sim_step*(n_sim + n_lat - release_D[i] - WS - evaporation_losses_Co - q_pump[i] + q_rel[i] - leak)
            
            c_hour = c_hour + 1

            #k = (n_sim + n_lat - WS - evaporation_losses_Co - q_pump[i] + q_rel[i] - leak - release_D[i])
            #if k<0:
                #print(k, "k value", release_D[i], "release D", storage_Co[i], storage_Co[i+1])


        s_rr.append(storage_Co[HH])
        #print(storage_Co[HH], "storgae Co HH")
        s_rr.append(storage_mr[HH])
        s_rr.append(utils.computeMean(release_A))
        s_rr.append(utils.computeMean(release_B))
        s_rr.append(utils.computeMean(release_C))
        s_rr.append(utils.computeMean(release_D))

        # rDTurb = []
        # for i in range(0, len(release_D)):
        #     rDTurb.append(min(release_D[i], Tcap))
        
        # array of the minimum of release D or the cap release
        rDTurb = np.minimum(release_D, Tcap)
            
        #print(level_Co)
        hp = self.g_hydRevCo(rDTurb, level_Co, day_of_year, hour0)
        hp_mr = self.g_hydRevMR(q_pump, q_rel, level_Co, level_mr, day_of_year, hour0)

        # Revenue
        s_rr.append(hp[1])
        s_rr.append(hp_mr[2])
        s_rr.append(hp_mr[3])

        # Production
        s_rr.append(hp[0])
        s_rr.append(hp_mr[0])
        s_rr.append(hp_mr[1])

        return s_rr


    def g_VolRel(self, q, qTarget):

        g = []
        delta = 24*3600
        #shortage = []

        for i in range(0, len(q)):
            tt = i % self.n_days_one_year
            g.append((q[i]*delta)/(qTarget[tt]*delta))

            #shortage.append(qTarget[tt] - q[i])
 
        return (sum(g)/len(g))
    
    def g_ShortageIndex(self, q, qTarget):
        delta = 24*3600
        g = []
        for i in range(0,len(q)):
            tt = i % self.n_days_one_year
            gg = max((qTarget[tt]*delta) - (q[i]*delta), 0.0) / (qTarget[tt]*delta)
            g.append(gg*gg)

        return sum(g)/len(g)

    def g_StorageReliability(self, h, hTarget):
        c = 0
        Nw = 0
        for i in range(0, len(h )-1):
            tt = i% self.n_days_one_year
            if h[i+1] < hTarget[tt]:
                #print(h[i+1])
                c = c+1
            if hTarget[tt] >0:
                Nw = Nw +1
        
        stor_rel = 1 - (c/Nw)
        # print(stor_rel, "stor_rel")
        # print(c, "c")
        # print(Nw, "Nw")
        return stor_rel

    def evaluateMC(self, **kwargs):

        #print("opt_met",opt_met)

        opt_met = 1

        center = [v for k,v in kwargs.items() if 'c' in k]
        radius = [v for k,v in kwargs.items() if 'r' in k]
        weights = [v for k,v in kwargs.items() if 'w' in k]
        phaseshift = [v for k,v in kwargs.items() if 'ps' in k]

        #Jcoalitionappchester, Jcoalitiondischarge, Jcoalitionwatersupply, 
        Jhydropower, Jatomicpowerplant, Jbaltimore, Jchester, Jrecreation, Jenvironment, Jfloodrisk, JFloodDuration   = self.simulate( center, radius, weights, phaseshift,self.inflow_MC, self.inflowLat_MC, self.inflow_Muddy_MC, self.evap_CO_MC, self.evap_Muddy_MC, opt_met)
        
        # Picking out the worst case outcomes from all the outcomes. The worst outcome is too sensitive and extreme to changes in input values. 99th percentile worst case serves as a better example

        hydropower_revenue = utils.computePercentile(Jhydropower, 99)
        atomic_power_plant_discharge = utils.computePercentile(Jatomicpowerplant, 99)
        baltimore_discharge = utils.computePercentile(Jbaltimore, 99)
        chester_discharge = utils.computePercentile(Jchester, 99)
        recreation = utils.computePercentile(Jrecreation, 99)
        environment = utils.computePercentile(Jenvironment, 99)
        flood_risk = utils.computePercentile(Jfloodrisk, 99)
        flood_duration = utils.computePercentile(JFloodDuration, 99)

        outcomes = {
                     'hydropower_revenue' : hydropower_revenue
                    , 'atomic_power_plant_discharge' : atomic_power_plant_discharge
                    , 'baltimore_discharge' : baltimore_discharge
                    , 'chester_discharge' : chester_discharge
                    , 'recreation' : recreation
                    , 'environment' : environment
                    , 'flood_risk' : flood_risk
                    , 'flood_duration' : flood_duration
                    }

        return outcomes

        

    
    def simulate(self, center, radius, weights,phaseshift, inflow_MC_n_sim, inflowLateral_MC_n_lat, inflow_Muddy_MC_n_mr, evap_CO_MC_e_co, evap_Muddy_MC_e_mr, opt_met):
        # Initializing daily variables
        # storages and levels
        storage_Co = [-999.0] * (self.time_horizon_H + 1)
        level_Co = [-999.0] * (self.n_days_in_year + 1)
        storage_MR = [-999.0] * (self.time_horizon_H + 1)
        level_MR = [-999.0] * (self.time_horizon_H + 1)
        # Conowingo actual releases
        release_A = [-999.0] * self.n_days_in_year
        release_B = [-999.0] * self.n_days_in_year
        release_C = [-999.0] * self.n_days_in_year
        release_D = [-999.0] * self.n_days_in_year

        release2_A = []
        release2_B = []
        release2_C = []
        release2_D = []
        control_policy_release_decisions = []

        hydropowerProduction_Co = []
        hydropowerProduction_MR = []
        hydroPump_MR = []
        hydropowerRevenue_Co = []
        hydropowerRevenue_MR = []
        hydropowerPumpRevenue_MR = []

        # TODO: change it to numpy array
        # subdaily variables
        storage2_Co = [-999.0] * (self.day_fraction +1)
        level2_Co = [-999.0] * (self.day_fraction +1)
        storage2_MR = [-999.0] * (self.day_fraction +1)
        level2_MR = [-999.0] * (self.day_fraction +1)

        # TODO: change it to numpy array
        # objectives
        Jhydropower = [-999.0] * self.n_years
        Jhydropowerproduction = [-999.0] * self.n_years
        Jatomicpowerplant = [-999.0] * self.n_years
        Jbaltimore = [-999.0] * self.n_years
        Jchester = [-999.0] * self.n_years
        Jenvironment = [-999.0] * self.n_years
        Jrecreation = [-999.0] * self.n_years
        Jfloodrisk = [-999.0] * self.n_years
        JFloodDuration = [-999.0] * self.n_years

        # Q: What are these?
        #JMRprod = [-999.0] * self.n_years
        #JMRcons = [-999.0] * self.n_years

        if opt_met ==1:
            #print("opt_met",opt_met)
            control_law = RBF(self.RBFs, self.inputs, self.outputs, center, radius, weights)

        # initial condition
        level_Co[0] = (self.init_level)
        storage_Co[0] = self.levelToStorage(level_Co[0], 1)
        #print(storage_Co[0], "initial storage")
        level_MR[0] = (self.init_level_MR)
        storage_MR[0] = self.levelToStorage(level_MR[0], 0)

        # identification of the periodicity (365 x fdays)
        # TODO: Better names
        count = 0
        total_decision_steps = self.n_days_in_year*self.day_fraction
        day_of_week = 0
        uu = []
        ss_rr_hp = []
        isFlooding = []
        input =[]

        # TODO: find sources for this
        self.input_max.append(120) # max reservoir level
        self.input_max.append(1400000) # max inflowMC

        self.input_max.append(1) # max sin() function
        self.input_max.append(1) # max cos() function
        self.input_min.append(0)# min reservoir level
        self.input_min.append(0)# min infloWMC
        self.input_min.append(-1) # min sin() function
        self.input_min.append(-1) # min cos() function

        # standardization of the input-output of the RBF release curve
        self.output_max.append(max(self.w_atomic))
        self.output_max.append(max(self.w_baltimore))
        self.output_max.append(max(self.w_chester))
        self.output_max.append(1328269) # max spillway release (cfs) @ max storage(109.2 ft) +turbine capacity
        year = 0

        # Run simulation
        # begin for loop for entire time horizon

        for t in range(0, self.time_horizon_H):
            day_of_week = (self.day0 + t)%7
            day_of_year = t%self.n_days_in_year

            if day_of_year%self.n_days_in_year == 0 and t !=0:
                year = year + 1



            level2_Co[0] = level_Co[day_of_year]

            storage2_Co[0] = storage_Co[t]
            #if storage2_Co[0]<0:
                #print(storage2_Co[0], day_of_year, level_Co[day_of_year])
            level2_MR[0] = level_MR[t]
            storage2_MR[0] = storage_MR[t]


            # subdaily cycle
            # Q: Why do we use only 4 of these inputs and not the lateral inflow and inflow into muddy run?
            for j in range(0,self.day_fraction):

                jj = count%total_decision_steps

                # compute decision
                if opt_met==0: # fixed release
                    uu.append(uu[0])
                if opt_met == 1: #RBF - PSO
                    input.append(level2_Co[j]) # reservoir level

                    if t>0:
 
                        input.append(self.inflow_MC[year][day_of_year-1])
                    else:
                         input.append(self.inflow_MC[0][0])
                        
                    input.append(np.sin(2*np.pi*jj/total_decision_steps - phaseshift[0])) # var[30] = phase shift for sin() function  //second last
                    input.append(np.sin(2*np.pi*jj/total_decision_steps - phaseshift[1])) # var[31] = phase shift for cos() function //last variable

                    uu = self.RBFs_policy(control_law, input)
                    #print(uu)
                    input.clear()
                
                    
                try:
                    
                    ss_rr_hp = self.res_transition_h(storage2_Co[j], uu, inflow_MC_n_sim[year][day_of_year], inflowLateral_MC_n_lat[year][day_of_year], 
                evap_CO_MC_e_co[year][day_of_year], storage2_MR[j], inflow_Muddy_MC_n_mr[year][day_of_year], evap_Muddy_MC_e_mr[year][day_of_year], day_of_year, day_of_week, j)

                except:
                    print(t, " day")
                    raise IndexError()

                storage2_Co[j+1] = ss_rr_hp[0]
                #print(ss_rr_hp[0], "ss_rr_hp")
                storage2_MR[j+1] = ss_rr_hp[1]
                #print(storage2_Co[j+1], "storage")
                level2_Co[j+1] = self.storageToLevel(storage2_Co[j+1], 1)
                #print(level2_Co[j+1], "level")
                level2_MR[j+1] = self.storageToLevel(storage2_MR[j+1], 0)
                
                release2_A.append(ss_rr_hp[2])
                release2_B.append(ss_rr_hp[3])
                release2_C.append(ss_rr_hp[4])
                release2_D.append(ss_rr_hp[5])

                #print(ss_rr_hp[6])

                hydropowerRevenue_Co.append(ss_rr_hp[6])

                hydropowerPumpRevenue_MR.append(ss_rr_hp[7])
                hydropowerRevenue_MR.append(ss_rr_hp[8])

                hydropowerProduction_Co.append(ss_rr_hp[9])
                hydroPump_MR.append(ss_rr_hp[10])
                hydropowerProduction_MR.append(ss_rr_hp[11])

                input = []
                control_policy_release_decisions.append(uu)
                uu = []
                ss_rr_hp = []
                count = count + 1

            maxR = 0

            for k in range(0, self.day_fraction):
                if release2_D[k] > maxR:
                    maxR = release2_D[k]
                if maxR > self.rFlood:
                    isFlooding.append(1)
                else:
                    isFlooding.append(0)
            
                
            #for k in range(0, self.day_fraction):

            
            # level_Co only has as many values as the days in a year and the list gets reused after the year. 
            # Q: Is this because of limited data?
            level_Co[day_of_year + 1] = level2_Co[self.day_fraction]
            #print(level_Co[day_of_year +1])
            storage_Co[t+1] = storage2_Co[self.day_fraction]

            # taking mean release per day
            release_A[day_of_year] = sum(release2_A)/len(release2_A)
            release_B[day_of_year] = sum(release2_B)/len(release2_B)
            release_C[day_of_year] = sum(release2_C)/len(release2_C)
            release_D[day_of_year] = sum(release2_D)/len(release2_D)

            
            level_MR[t+1] = level2_MR[self.day_fraction]
            storage_MR[t+1] = storage2_MR[self.day_fraction]

            level2_Co = [-999.0] * (self.day_fraction +1) 
            storage2_Co = [-999.0] * (self.day_fraction +1) 
            release2_A.clear()  
            release2_B.clear() 
            release2_C.clear() 
            release2_D.clear() 
            level2_MR = [-999.0] * (self.day_fraction +1) 
            storage2_MR = [-999.0] * (self.day_fraction +1) 

            if day_of_year == 364 :
                #Jhydropowerproduction[int((t+1)/365) - 1] = -sum(hydropowerProduction_Co)/pow(10,6)
                #print(hydropowerRevenue_Co)
                Jhydropower[int((t+1)/365) - 1] = -sum(hydropowerRevenue_Co)/ pow(10,6)
                Jatomicpowerplant[int((t+1)/365) - 1] = -self.g_VolRel(release_A, self.w_atomic)
                Jbaltimore[int((t+1)/365) - 1] = -self.g_VolRel(release_B, self.w_baltimore)
                Jchester[int((t+1)/365) - 1] = -self.g_VolRel(release_C, self.w_chester)
                #print(level_Co)
                Jrecreation[int((t+1)/365) - 1] = -self.g_StorageReliability(level_Co, self.h_ref_rec)
                Jenvironment[int((t+1)/365) - 1] = self.g_ShortageIndex(release_D, self.min_flow)
                Jfloodrisk[int((t+1)/365) - 1] = self.g_Flood(level_Co, self.hFlood)
                JFloodDuration[int((t+1)/365) - 1] = self.g_FloodDuration(isFlooding)


                level_Co[0] = level_Co[365]
                hydropowerRevenue_Co = []
                isFlooding = []

       

        #return Jcoalitionappchester, Jcoalitiondischarge, Jcoalitionwatersupply, 
        return Jhydropower, Jatomicpowerplant, Jbaltimore, Jchester, Jrecreation, Jenvironment, Jfloodrisk, JFloodDuration 
        




