
import csv
from datetime import datetime

from servicecommon.persistence.interface.persistor import Persistor

from experimenthost.util.candidate_util import CandidateUtil

from framework.util.experiment_filer import ExperimentFiler


class FitnessPersistor(Persistor):
    """
    This implementation of the Persistor interface creates the
    fitness.csv file.
    """

    def __init__(self, experiment_dir, generation, fitness_objectives):
        """
        Constructor.

        """
        self.filer = ExperimentFiler(experiment_dir)
        self.generation = generation
        self.fitness_objectives = fitness_objectives
        self.candidate_util = CandidateUtil(fitness_objectives)
        self.basename = 'fitness.csv'
        self.time_format = '%Y-%m-%d-%H:%M:%S'


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
                In this case we are expecting an advanced stats dictionary
                from the SoftOrderPersistor
        """
        advanced_stats = obj

        filename = self.filer.experiment_file(self.basename)
        self.write_csv_file(filename, advanced_stats)



    def write_csv_file(self, filename, advanced_stats):
        """
        Writes out the fitness.csv file

        :param filename: The filename to write to
        :param advanced_stats: The advanced_stats dict gathered by the
                            SoftOrderPersistor
        :return: Nothing
        """
        with open(filename, 'w') as csv_file:

            # Prepare dynamic column names
            primary_objective = self.fitness_objectives.get_fitness_objective(0)
            fitness_name = primary_objective.get_metric_name()

            best_fitness_field_name = 'Best ' + fitness_name
            best_fitness_id_field_name = best_fitness_field_name + ' id'
            avg_fitness_field_name = 'Avg ' + fitness_name

            field_names = ['Generation',
                           'Timestamp',
                           best_fitness_id_field_name,
                           best_fitness_field_name,
                           avg_fitness_field_name]
            csv_writer = csv.DictWriter(csv_file,
                                        fieldnames=field_names,
                                        quoting=csv.QUOTE_MINIMAL,
                                        lineterminator="\n")
            csv_writer.writeheader()
            for gen in range(self.generation + 1):

                # Get timestamp in human-readable format
                timestamp = advanced_stats['time'][gen]
                ts_datetime = datetime.fromtimestamp(timestamp)
                time_string = ts_datetime.strftime(self.time_format)

                # Get best candidate
                # XXX multi-objective
                best_id = None
                best_fitness = None
                candidate = advanced_stats['best_candidate'][gen]
                if candidate is not None:
                    best_id = self.candidate_util.get_candidate_id(candidate)
                    best_fitness = self.candidate_util.get_candidate_fitness(candidate)

                # Get average fitness
                # XXX multi-objective
                avg_fitness = advanced_stats['avg_fitness'][gen]

                row = {
                    'Generation': gen,
                    'Timestamp': time_string,
                    best_fitness_id_field_name: best_id,
                    best_fitness_field_name: best_fitness,
                    avg_fitness_field_name: avg_fitness
                }
                csv_writer.writerow(row)
