import streamlit as st 
from streamlit.ReportThread import add_report_ctx
import SessionState
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import subprocess
import threading
from glob import glob
from random import randint

from datetime import datetime
import sys
from collections import OrderedDict


def process_input():

        # Input number of criteria and their values
        session_state.criteria_value = []
        session_state.criteria_name = []
        # criteria = 0
        session_state.product = st.text_input("Product", key="productname")
        criteriaCount = st.number_input("No. of criteria", min_value=1, value=1, step=1, key="criteriaCount")


        for criteria in range(criteriaCount):
                vars()['criteria_' + str(criteria + 1)] = st.text_input("Criteria " + str(criteria + 1), key="criteria_name" + str(criteria + 1))
                # if vars()['criteria_values' + str(criteria + 1)] is not None:
                session_state.criteria_name.append(vars()['criteria_' + str(criteria + 1)])

        if st.checkbox("Add values", False):
                n = len(session_state.criteria_name)
                st.sidebar.subheader("Key in " + str(len(session_state.criteria_name)) + " X " + str(len(session_state.criteria_name)) + " Matrix for all criterias")


                criteriaMatrix = st.sidebar.text_area(
                                        label = "Criteria Matrix",
                                        value = 0.0,
                                        key = "cm"
                                )
                
                new = [float(s) for s in criteriaMatrix.split(',')]
                print (new)

                if len(new) == n*n:
                        session_state.criteria_value = [new[i:i+n] for i in range(0, len(new), n)]
                else:
                        st.sidebar.error("The entered input values doesn't meet the size of matrix")


        
        # for criteria in range(criteriaCount):
        #         vars()['criteria_' + str(criteria + 1)] = st.text_input("Criteria " + str(criteria + 1), key="criteria_name" + str(criteria + 1))
        #         vars()['criteria_values' + str(criteria + 1)] = st.text_area(
        #                 label = "Criteria Values " + str(criteria + 1),
        #                 value = 0.0,
        #                 key = "criteriaValue" + str(criteria + 1)
        #         )

        #         if vars()['criteria_values' + str(criteria + 1)] is not None:
        #                 vars()['criteria_values' + str(criteria + 1)] = [float(s) for s in vars()['criteria_values' + str(criteria + 1)].split(',')]

        #         if vars()['criteria_' + str(criteria + 1)] is not None and vars()['criteria_values' + str(criteria + 1)] is not None:
        #                 session_state.criteria_value.append(vars()['criteria_values' + str(criteria + 1)])
        #                 session_state.criteria_name.append(vars()['criteria_' + str(criteria + 1)])
        


        return session_state.product, session_state.criteria_name, session_state.criteria_value

class AHP:
        RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
        consistency = False
        priority_vec = None
        compete = False
        normal = False
        sublayer = None
        
        def __init__(self, name, size):
                self.name = name
                self.size = size
                self.matrix = np.zeros([size,size])
                self.criteria = [None] * size
               
        def update_matrix(self, mat, automated=True):
                if not ((mat.shape[0] == mat.shape[1]) and (mat.ndim == 2)):
                        raise Exception('Input matrix must be squared.')
                if self.size != len(self.criteria):
                        self.criteria = [None] * self.size
                self.matrix = mat
                self.size = mat.shape[0]
                self.consistency = False
                self.normal = False
                self.priority_vec = None
                if automated:
                        self.rank()
        
        def input_prioriry_vec(self, vec):
                if not (vec.shape[1] == 1) and (vec.shape[0] == self.size) and (vec.ndim == 2):
                        raise Exception('The size of input priority vector is not compatable.')
                self.priority_vec = vec
                self.output = self.priority_vec / self.priority_vec.sum()
                self.consistency = True
                self.normal = True
        
        def rename(self, name):
                self.name = name
        
        def update_criteria(self, criteria):
                if len(criteria) == self.size:
                        self.criteria = criteria
                else:
                        raise Exception('Input doesn\'t match the number of criteria.')
            
        def add_layer(self, alternative):
                if not self.criteria:
                        raise Exception('Please input criterias before adding new layer.')
                self.compete  = False
                self.sublayer = OrderedDict()
                self.alternative = alternative
                for i in range(self.size):
                        self.sublayer[self.criteria[i]] = AHP(self.criteria[i], len(alternative))
                        self.sublayer[self.criteria[i]].update_criteria(self.alternative)
            
        def normalize(self):
                if self.normal:
                        pass
                col_sum = self.matrix.sum(axis = 0)
                try:
                        self.matrix = self.matrix / col_sum
                except:
                        raise Exception('Error when normalize on columns.')
                else:
                        self.nomral = True
                        self.priority_vec = self.matrix.sum(axis = 1).reshape(-1,1)
    
        def rank(self):
                if self.consistency:
                        df = pd.DataFrame(data = self.output, index = self.criteria, columns=[self.name])
                        return df
                if not self.normal:
                        self.normalize()
                Ax = self.matrix.dot(self.priority_vec)
                eigen_val = (Ax / self.priority_vec).mean()
                CI = (eigen_val - self.size) / (self.size - 1)
                CR = CI / self.RI[self.size]
                if CR < 0.1:
                        self.consistency = True
                        self.output = self.priority_vec / self.priority_vec.sum()
                        self.df_out = pd.DataFrame(data = self.output, index = self.criteria, columns=[self.name])
                        return self.df_out
                else:
                        raise Exception('The consistency for desicion is not sufficient.')
            
        def make_decision(self):

                if not self.consistency:
                        self.rank()
                if not self.compete:
                        temp = True
                        arrays = []
                        interresults = []
                        colnames = []
                        for item in self.sublayer.values():
                                itemdf = item.rank()
                                temp = temp and item.consistency
                                if temp:
                                        arrays.append(item.output)
                                        interresults.append(itemdf.values)
                                        colnames.append(list(itemdf.columns))
                                else:
                                        raise Exception('Please check the AHP for {}'.format(item.name))
                        if temp:
                                self.compete = True
                        else:
                                pass

                        self.recommendation = np.concatenate(arrays, axis = 1).dot(self.output)
                        self.inter = np.concatenate(interresults,axis=1)
                        self.interfinal = np.concatenate((self.inter, self.recommendation),axis=1)
                        self.collist = [item for sublist in colnames for item in sublist]
                        self.collist.append('AHP Score')
                       
                self.df_decision = pd.DataFrame(data = self.interfinal, index = self.alternative, columns = self.collist)
                self.df_decision.index.name = 'Alternative'
                self.df_decision['rank'] = self.df_decision['AHP Score'].rank(ascending = False)
                return self.df_decision

def initSession():

        sessionstate_init = SessionState.get(

                criteria_value = [],
                criteria_name = [],
                ranking = None,
                product = None


        )

        return sessionstate_init


if __name__ == "__main__":

        st.sidebar.image("Apps/imgs/siemens_logo.png", width = 300)

        st.title("Ranking of interventions")

        # maintain session variables
        session_state = initSession()

        # set the datetime
        dt_now = datetime.now().strftime("%Y%m%d")

        # This is your Project Root
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        
        # PAGES
        PAGES = [
                "Setup Comparison Matrix",
                "Make Decision",
                "Adhoc Decision Criteria"
        ]

        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", PAGES)

        if selection.lower().replace(" ", "")  == "setupcomparisonmatrix":

                st.header("Setup Comparison Matrix")

                product, criteria_name, criteria_value = process_input()

                criteriaCount = len(criteria_value)

                if st.button("submit"):
                        st.info(criteria_name)
                        session_state.ranking = AHP(product, criteriaCount)
                        session_state.ranking.update_matrix(np.array(
                                        criteria_value
                        ))

                        session_state.ranking.update_criteria(criteria_name)
                        df = session_state.ranking.rank()
                        st.dataframe(df)

        elif selection.lower().replace(" ", "")  == "makedecision":

                st.header("Make Decision")
                alternates = 0
                criteria_name = session_state.criteria_name
                alternate_name= []
                alt_matrix = {}
                session_state.alternatesCount = st.number_input("No. of alternates", min_value=1, value=1, step=1, key="alternatesCount")
                for alternates in range(session_state.alternatesCount):
                        vars()['Alternate_' + str(alternates + 1)] = st.text_input("Alternate " + str(alternates + 1), key="alternate_name" + str(alternates + 1))
                        if vars()['Alternate_' + str(alternates + 1)] is not None:
                                alternate_name.append(vars()['Alternate_' + str(alternates + 1)])

                st.subheader("Input Alternate matrix values for the following criterias")
                st.info(criteria_name)
                if st.checkbox("Add values", False):
                        n = len(alternate_name)
                        st.sidebar.subheader("Key in " + str(len(alternate_name)) + " X " + str(len(alternate_name)) + " Matrix for all criterias")
                        
                        for am in range(len(criteria_name)):
                                vars()['Alternate_Matrix_' + criteria_name[am]] = st.sidebar.text_area(
                                                        label = "Alternate Matrix " + criteria_name[am],
                                                        value = 0.0,
                                                        key = "am_" + criteria_name[am]
                                                )
                                


                                new = [float(s) for s in vars()['Alternate_Matrix_' + criteria_name[am]].split(',')]

                                if len(new) == n*n:
                                        vars()['Alternate_Matrix_' + criteria_name[am]] = [new[i:i+n] for i in range(0, len(new), n)]
                                elif len(new) == n:
                                        vars()['Alternate_Matrix_' + criteria_name[am]] = [new[i:i+1] for i in range(0, len(new), 1)]
                                else:
                                        st.sidebar.error("The entered input values doesn't meet the size of matrix")

                                alt_matrix.update({criteria_name[am]: vars()['Alternate_Matrix_' + criteria_name[am]]})


                if st.button("submit"):


                        ## mat.shape[0] == mat.shape[1]) and (mat.ndim == 2)

                        session_state.ranking.add_layer(alternate_name)

                        for i in range(len(criteria_name)):

                                session_state.ranking.sublayer[criteria_name[i]].update_matrix(
                                        np.array(alt_matrix[criteria_name[i]]))


                        df = session_state.ranking.make_decision()
                        st.dataframe(df)
                        

        elif selection.lower().replace(" ", "")  == "adhocdecisioncriteria":
                st.header("Adhoc Decision Criteria")
                # Input number of criteria and their values
                session_state.criteria_value = []
                session_state.criteria_name = []
                
                # criterianame = st.text_input("Criteria", key="criterianame")
                criteriaCount = 1

                for criteria in range(criteriaCount):
                        vars()['criteria_' + str(criteria + 1)] = st.text_input("Criteria " + str(criteria + 1), key="criteria_name" + str(criteria + 1))
                        # if vars()['criteria_values' + str(criteria + 1)] is not None:
                        session_state.criteria_name.append(vars()['criteria_' + str(criteria + 1)])

                if st.checkbox("Add values", False):
                        n = criteriaCount
                        st.sidebar.subheader("Key in " + str(n) + " X " + str(session_state.alternatesCount) + " Matrix for all criterias")


                        criteriaMatrix = st.sidebar.text_area(
                                                label = "Criteria Matrix",
                                                value = 1.0,
                                                key = "cm"
                                        )
                        
                        new = [float(s) for s in criteriaMatrix.split(',')]
                        new = [float(i)/sum(new) for i in new]

                        if len(new) == n*session_state.alternatesCount:
                                # session_state.criteria_value = [new[i:i+n] for i in range(0, len(new), n)]
                                session_state.criteria_value = new

                        else:
                                st.sidebar.error("The entered input values doesn't meet the size of matrix")

                        if st.button("submit"):

                                df = session_state.ranking.make_decision()
                                df['Normalized_' + session_state.criteria_name[0]] = np.array(session_state.criteria_value)
                                df['New_AHP_score'] =  df['Normalized_' + session_state.criteria_name[0]] / df['AHP Score']
                                df['New_Rank'] = df['New_AHP_score'].rank(ascending=False)

                                st.dataframe(df)          
                        
                
             
 
        st.sidebar.markdown("#### **Copyright &copy; 2020 DA REAMS, Siemens Mobility**")





