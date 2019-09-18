
import time
import copy

from experimenthost.persistence import visualize
from experimenthost.persistence.fitness_persistor import FitnessPersistor
from experimenthost.persistence.best_fitness_candidate_persistence \
    import BestFitnessCandidatePersistence
from experimenthost.util.candidate_util import CandidateUtil


class SoftOrderPersistor():
    """
    A LEAF-ier ploy to separate out the files that are persisted by
    softorder_coevolution -- all except for checkpointing.

    We do this so that the session_server can be the one to do the
    persistence and the files can persist on the session_server machine.
    """

    def __init__(self, experiment_dir, fitness_objectives,
                 save_best=True, draw=True, logger=None):

        self.experiment_dir = experiment_dir
        self.save_best = save_best
        self.draw = draw
        self.fitness_objectives = fitness_objectives
        self.candidate_util = CandidateUtil(fitness_objectives)
        self.advanced_stats = {
            'best_candidate': [],
            'avg_fitness': [],
            'time': []
        }
        self.logger = logger


    def persist(self, population, generation):
        """
        Gather statistics and persist what we want to files
        """

        best_candidate = self.gather_advanced_stats(population)
        self.do_save(generation, best_candidate)
        self.do_draw(generation)

        fitness_persistence = FitnessPersistor(self.experiment_dir, generation,
                                             self.fitness_objectives)
        fitness_persistence.persist(self.advanced_stats)


    def get_candidate_fitness(self, candidate):
        return self.candidate_util.get_candidate_fitness(candidate)


    def average_fitness(self, population):
        """
        Returns the average raw fitness of population
        """
        my_sum = 0.0
        counter = 1e-308
        for candidate in population:
            fitness = self.get_candidate_fitness(candidate)
            if fitness is not None:
                my_sum += fitness
                counter += 1
        return my_sum / counter

    def find_best_candidate(self, population):

        if population is None or len(population) == 0:
            return None

        one = population[0]
        best = None

        if isinstance(one, dict):
            # Candidates are dictionaries
            best_fitness = None
            for candidate in population:
                fitness = self.get_candidate_fitness(candidate)
                if best_fitness is None:
                    best_fitness = fitness
                    best = candidate
                elif fitness > best_fitness:
                    best_fitness = fitness
                    best = candidate
        else:
            # Candidates are ChromosomeData
            best = max(population)

        return best

    def gather_advanced_stats(self, population):
        """
        Populates the advanced_stats member dictionary
        with info about the generation just evaluated.
        """
        best_candidate = self.find_best_candidate(population)
        self.advanced_stats['best_candidate'].append(copy.deepcopy(best_candidate))
        self.advanced_stats['avg_fitness'].append(self.average_fitness(population))
        self.advanced_stats['time'].append(time.time())
        return best_candidate

    def do_save(self, generation, best_candidate):

        # saves the best candidate from the current generation
        if not self.save_best:
            return

        if best_candidate is not None:
            candidate_id = self.candidate_util.get_candidate_id(best_candidate)
            best_persistence = BestFitnessCandidatePersistence(self.experiment_dir,
                                                candidate_id,
                                                generation,
                                                logger=self.logger)
            best_persistence.persist(best_candidate)


    def do_draw(self, generation):

        if self.draw:
            if generation >= 2:
                stats = (self.advanced_stats['best_candidate'],
                         self.advanced_stats['avg_fitness'])
                visualize.plot_stats(stats, self.candidate_util,
                                     self.experiment_dir)
