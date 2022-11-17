from collections import OrderedDict
from itertools import combinations

import pandas as pd
from sklearn.metrics import r2_score
from typing import List, Tuple, Union

from .gd_regressor import GeoDetectorRegressor


class GeoDetector():
    """ GeoDetector proveid an easy to use interface to calculate the q values
    of the covariates.
    """

    def __init__(self,
                 x_names: List[str],
                 y_name: str,
                 data,
                 *,
                 method=None,
                 random_state=42):
        self.gd_regressor = GeoDetectorRegressor(method=method,
                                                 random_state=random_state)
        self.x_names = x_names
        self.y_name = y_name
        self.data = data
        self._check_valid_data(data, x_names, y_name)  #  type: ignore

    def q_values(self):
        """_q_values calculate the q values of the covariates

        Returns
        -------
        dict, the key is the covariate name, and the value is the q value
        """
        # key is the name of x, value is the q value of x
        dic = OrderedDict()
        for x_name in self.x_names:
            self.gd_regressor.fit(self.data[[x_name]], self.data[self.y_name])
            y_true = self.data[self.y_name]
            # the mean of y in each group is the prediction of y
            y_pred = self.gd_regressor.predict(self.data[[x_name]])
            # 1 - SSW/SST = 1 - MSE/VAR = R2
            q = r2_score(y_true, y_pred)
            dic[x_name] = q
        return dic

    def inter_q_values(self):
        """ factor_detect detect the interaction between covariates
        
        Returns
        -------
        dict, the key is the tuple covariate name, and the value is the q value
        
        Examples
        --------
        
        """
        dic = OrderedDict()
        for x1, x2 in combinations(self.x_names, 2):
            self.gd_regressor.fit(self.data[[x1, x2]], self.data[self.y_name])
            y_true = self.data[self.y_name]
            # the mean of y in each group is the prediction of y
            y_pred = self.gd_regressor.predict(self.data[[x1, x2]])
            # 1 - SSW/SST = 1 - MSE/VAR = R2
            q = r2_score(y_true, y_pred)
            dic[(x1, x2)] = q
        return dic

    def interaction_detect(
        self,
        interaction_type=False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ factor_detect detect the interaction between covariates
        Parameters
        ----------
        return_union: bool, default False
        whether return the union of the covariates

        Returns
        -------
        a dataframe, the index and columns are the covariate names, and the 
        value is the q value
        """
        df = pd.DataFrame(index=self.x_names, columns=self.x_names)

        q_dic = self.q_values()
        for x_name in self.x_names:
            df.loc[x_name, x_name] = q_dic[x_name]

        inter_q_dic = self.inter_q_values()
        for cols, q in inter_q_dic.items():
            df.loc[cols[0], cols[1]] = q
            df.loc[cols[1], cols[0]] = q

        if not interaction_type:
            return df
        elif interaction_type:
            interaction_type = pd.DataFrame(index=self.x_names,
                                            columns=self.x_names)
            for cols, q in inter_q_dic.items():
                new_q = df.loc[cols[0], cols[1]]
                q_0 = q_dic[cols[0]]
                q_1 = q_dic[cols[1]]
                if new_q < min(q_0, q_1):
                    interaction_type.loc[cols[0], cols[1]] = "no_linear-"
                elif min(q_0, q_1) < new_q < max(q_0, q_1):
                    interaction_type.loc[cols[0],
                                         cols[1]] = "single_no_linear-"
                elif max(q_0, q_1) < new_q < q_0 + q_1:
                    interaction_type.loc[cols[0], cols[1]] = "bi+"
                elif new_q == q_0 + q_1:
                    interaction_type.loc[cols[0], cols[1]] = "alone"
                elif new_q > q_0 + q_1:
                    interaction_type.loc[cols[0], cols[1]] = "no_linear+"
                else:
                    raise ValueError("new_q is not in the range")
            return df, interaction_type
        else:
            raise ValueError("interaction_type must be bool")

    def _check_valid_data(self, data: pd.DataFrame, x_names: List[str],
                          y_names: str):
        for x_name in x_names:
            assert x_name in data.columns, f"{x_name} is not in data."

        assert y_names in data.columns, f"{y_names} is not in data."
