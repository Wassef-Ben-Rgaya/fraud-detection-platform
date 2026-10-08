# ============================================================
# API SCHEMAS / VALIDATION
# ============================================================

MODEL_FEATURES = 32


# ============================================================
# SINGLE PREDICTION
# ============================================================

def validate_features(features):
    """
    Validate a single transaction feature vector.
    """

    if features is None:

        return (
            False,
            "Le champ 'features' est obligatoire."
        )

    if not isinstance(features, list):

        return (
            False,
            "Le champ 'features' doit être une liste."
        )

    if len(features) != MODEL_FEATURES:

        return (
            False,
            f"{MODEL_FEATURES} features sont nécessaires. "
            f"{len(features)} reçues."
        )

    for i, value in enumerate(features):

        try:

            float(value)

        except (TypeError, ValueError):

            return (
                False,
                f"La feature {i + 1} n'est pas numérique."
            )

    return True, None


# ============================================================
# BATCH PREDICTION
# ============================================================

def validate_batch_features(features):
    """
    Validate a batch of transaction feature vectors.
    """

    if features is None:

        return (
            False,
            "Le champ 'features' est obligatoire."
        )

    if not isinstance(features, list):

        return (
            False,
            "Le champ 'features' doit être une liste."
        )

    if len(features) == 0:

        return (
            False,
            "La liste 'features' est vide."
        )

    for row_index, row in enumerate(features):

        if not isinstance(row, list):

            return (
                False,
                f"La transaction {row_index} doit être une liste."
            )

        if len(row) != MODEL_FEATURES:

            return (
                False,
                f"La transaction {row_index} doit contenir "
                f"{MODEL_FEATURES} features. "
                f"{len(row)} reçues."
            )

        for feature_index, value in enumerate(row):

            try:

                float(value)

            except (TypeError, ValueError):

                return (
                    False,
                    f"La feature {feature_index + 1} "
                    f"de la transaction {row_index} "
                    f"n'est pas numérique."
                )

    return True, None