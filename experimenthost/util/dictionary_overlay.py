

class DictionaryOverlay():
    """
    Policy class assisting with deep dictionary updates.
    """


    def overlay(self, basis, overlay):
        """
        :param basis: The basis dictionary to be overlayed.
                        Unmodified on exit.
        :param overlay: The overlay dictionary.
                        Unmodified on exit.
        :return: A new dictionary whose keys are the union of all
                keys between the two dictionaries and whose values
                favor the contents of the overlay dictionary.
                In the case where values in both the basis and overlay
                are both dictionaries, this method recurses.
        """

        # Preliminary argument checking
        if basis is None and overlay is None:
            return None

        if basis is None:
            basis = {}

        if not isinstance(basis, dict):
            raise ValueError("basis is not a dictionary")

        if overlay is None:
            overlay = {}

        if not isinstance(overlay, dict):
            raise ValueError("overlay is not a dictionary")

        # Do not modify any incoming arguments
        result = {}
        result.update(basis)

        for key in overlay.keys():

            # Any key we do not have, we just copy over the value from overlay.
            if key not in basis:
                result[key] = overlay[key]
                continue

            basis_value = basis.get(key)
            overlay_value = overlay.get(key)

            # By default, the result value for the key will be the overlay
            # value itself.
            result_value = overlay_value

            # ... except if both values are dictionaries.
            # In that case, recurse.
            if isinstance(basis_value, dict) and \
               isinstance(overlay_value, dict):
                result_value = self.overlay(basis_value, overlay_value)

            result[key] = result_value

        return result
