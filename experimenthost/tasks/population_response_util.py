

class PopulationResponseUtil():
    """
    Utility class with stateless methods that assist in packing
    and unpacking population responses that go over the wire.
    """

    def unpack_response(self, population_response, unpack_obj):
        """
        :param population_response: The population response to unpack
        :param unpack_obj:  The object onto which unpacked data is assigned
            It is expected this object has the following fields:
            * persistor.advanced_stats
            * server_stats
            * seen_checkpoint_ids
            * generation
            * checkpoint_id
        :return: The population from the population response
        """

        if population_response is None:
            return None

        # Disassemble the returned population response
        population = population_response["population"]
        unpack_obj.generation = population_response["generation_count"]
        evaluation_stats = population_response["evaluation_stats"]
        unpack_obj.checkpoint_id = population_response["checkpoint_id"]

        # Disassemble evaluation stats
        if evaluation_stats is not None:
            # Pick up where we left off, if there is something to pick up
            # Restore stats before calling evaluate
            unpack_obj.persistor.advanced_stats = \
                evaluation_stats.get("advanced_stats",
                                     unpack_obj.persistor.advanced_stats)
            unpack_obj.server_stats = evaluation_stats.get("server_stats", {})
            unpack_obj.seen_checkpoint_ids = \
                evaluation_stats.get("seen_checkpoint_ids", [])

        # Append the new checkpoint id to the list of what we have seen
        if unpack_obj.checkpoint_id is not None:
            unpack_obj.seen_checkpoint_ids.append(unpack_obj.checkpoint_id)

        return population


    def pack_response(self, population, pack_obj):
        """
        Populates a population response with various fields.
        :param population: The list of candidates with results metrics
            to send over the wire in the population response.
        :param pack_obj:  The object from which data is taken to pack
            It is expected this object has the following fields:
            * persistor.advanced_stats
            * server_stats
            * seen_checkpoint_ids
            * generation
            * checkpoint_id
        :return: a properly populated population response
        """

        # Assemble new evaluation stats
        evaluation_stats = {
            "advanced_stats": pack_obj.persistor.advanced_stats,
            "server_stats": pack_obj.server_stats,
            "seen_checkpoint_ids": pack_obj.seen_checkpoint_ids
        }

        # Set up the population response for next time
        population_response = {
            "population": population,
            "generation_count": pack_obj.generation,
            "evaluation_stats": evaluation_stats,
            "checkpoint_id": pack_obj.checkpoint_id,
        }

        return population_response
