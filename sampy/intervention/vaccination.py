import numpy as np
from ..pandas_xs.pandas_xs import DataFrameXS
from .jit_compiled_functions import vaccination_apply_vaccine_from_array_condition


class BaseVaccinationSingleSpeciesDisease:
    def __init__(self, disease=None, **kwargs):
        if disease is None:
            raise ValueError(
                "No disease object provided for the vaccination. Should be provided using kwarg 'disease'.")
        self.disease = disease
        self.target_species = self.disease.host
        self.target_species.df_population['vaccinated_' + self.disease.disease_name] = False
        self.target_species.dict_default_val['vaccinated_' + self.disease.disease_name] = False


class VaccinationSingleSpeciesDiseaseFixedDuration:
    def __init__(self, duration_vaccine=None, **kwargs):
        if duration_vaccine is None:
            raise ValueError(
                "No duration provided for the vaccination duration. Should be provided using kwarg 'duration_vaccine'.")
        self.duration_vaccine = int(duration_vaccine)
        self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name] = 0
        self.target_species.dict_default_val['cnt_vaccinated_' + self.disease.disease_name] = 0

    def update_vaccine_status(self):
        """
        Should be call at each time-step of the simulation. Update the attribute that count how many day each individual
        has been vaccinated, and remove the vaccinated status of the individual which recieved their dose for more than
        'duration_vaccine' time-steps.
        """
        self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name] += 1

        arr_lose_vaccine = self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name] >= \
                           self.duration_vaccine
        not_arr_lose_vaccine = ~arr_lose_vaccine

        self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name] = \
            self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name] * not_arr_lose_vaccine
        self.target_species.df_population['vaccinated_' + self.disease.disease_name] = \
            self.target_species.df_population['vaccinated_' + self.disease.disease_name] * not_arr_lose_vaccine
        self.target_species.df_population['imm_' + self.disease.disease_name] = \
            self.target_species.df_population['imm_' + self.disease.disease_name] * not_arr_lose_vaccine

    def apply_vaccine_from_array(self, array_vaccine_level, condition=None, position_attribute='position'):
        """
        Apply vaccine to the agents based on the 1D array 'array_vaccine_level'. array_vaccine_level[i] is the
        probability for an agent on the vertex of index i to get vaccinated.

        Note that, by default, infected and contagious agents can get vaccinated. They can be excluded using the
        kwarg 'condition'

        :param array_vaccine_level: 1D array of float. Floats between 0 and 1.
        :param condition: optional, 1D array of bool, default None.
        :param position_attribute: optional, string, default 'position'.
        """
        if condition is None:
            condition = np.full((self.target_species.df_population.nb_rows,), True, dtype=np.bool_)

        rand = np.random.uniform(0, 1, (condition.sum(),))

        newly_vaccinated = \
            vaccination_apply_vaccine_from_array_condition(
                self.target_species.df_population['vaccinated_' + self.disease.disease_name],
                self.target_species.df_population['cnt_vaccinated_' + self.disease.disease_name],
                self.target_species.df_population['imm_' + self.disease.disease_name],
                array_vaccine_level, self.target_species.df_population[position_attribute], rand, condition)

        not_newly_vaccinated = ~newly_vaccinated
        self.target_species.df_population['inf_' + self.disease.disease_name] *= not_newly_vaccinated
        self.target_species.df_population['con_' + self.disease.disease_name] *= not_newly_vaccinated

    def apply_vaccine_from_dict(self, graph, dict_vertex_id_to_level, condition=None, position_attribute='position'):
        """
        same as apply_vaccine_from_array, but the 1D array is replaced by a dictionary whose keys are vertices ID and
        values is the vaccination level on each cell.

        :param graph: graph object on which the vaccine is applied
        :param dict_vertex_id_to_level: dictionnary-like object with vaccine level
        :param condition: optional, 1D array of bool, default None.
        :param position_attribute: optional, string, default 'position'.
        """
        array_vac_level = np.full((graph.number_vertices,), 0., dtype=float)
        for id_vertex, level in dict_vertex_id_to_level.items():
            array_vac_level[graph.dict_cell_id_to_ind[id_vertex]] = level

        self.apply_vaccine_from_array(array_vac_level, condition=condition, position_attribute=position_attribute)
