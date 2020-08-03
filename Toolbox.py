import pandas as pd


class ParameterGenerator:
    @staticmethod
    def create_combination_matrix(params: dict) -> pd.DataFrame:
        combination_list = pd.DataFrame({'dummy': [1]})
        for key, values in params.items():
            combination_list = pd.merge(combination_list, pd.DataFrame({key: values, 'dummy': [1] * len(values)}))
        combination_list.drop('dummy', axis=1, inplace=True)
        return combination_list
