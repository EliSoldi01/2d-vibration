import re
import pandas as pd

### TODO:
# - Modifica validazione aggiungendo anche i nuovi campi: experiment_id, protocol_id, arm, initial_angle


HEX_COLOR_REGEX = r"^#(?:[0-9a-fA-F]{3}){1,2}$"

import re


HEX_COLOR_REGEX = r"^#(?:[0-9a-fA-F]{3}){1,2}$"


def validate_protocol(protocol):
    """
    Validate the protocol configuration.

    Returns:
        bool: True if valid, False otherwise.
    """

    errors = []

    # --------------------------------------------------------
    # Protocol metadata
    # --------------------------------------------------------

    required_fields = [
        "experiment_id",
        "protocol_id",
        "arm",
        "initial_angle",
        "blocks",
        "patterns",
        "grid",
        "scales",
    ]

    for field in required_fields:
        if field not in protocol:
            errors.append(f"Missing protocol field: {field}")

    # Stop if the main structure is missing
    if errors:
        print("[ERROR] Protocol validation failed:")
        for e in errors:
            print(" -", e)
        return False

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    if "durations" not in protocol["blocks"]:
        errors.append("Missing blocks.durations")

    if "order" not in protocol["blocks"]:
        errors.append("Missing blocks.order")

    # --------------------------------------------------------
    # Patterns
    # --------------------------------------------------------

    patterns = protocol["patterns"]

    if "sequence_length" not in patterns:
        errors.append("Missing patterns.sequence_length")

    if "repetitions" not in patterns:
        errors.append("Missing patterns.repetitions")

    if "order" not in patterns:
        errors.append("Missing patterns.order")

    if "definitions" not in patterns:
        errors.append("Missing patterns.definitions")
        definitions = []
    else:
        definitions = patterns["definitions"]

    required_pattern_fields = [
        "id",
        "biceps",
        "triceps",
        "text",
        "effect",
        "color",
        "legend_position",
    ]

    for pattern in definitions:

        pattern_id = pattern.get("id", "unknown")

        # Required fields
        for field in required_pattern_fields:
            if field not in pattern:
                errors.append(
                    f"Pattern {pattern_id} missing field: {field}"
                )

        # Biceps
        if "biceps" in pattern:
            if (
                not isinstance(pattern["biceps"], list)
                or len(pattern["biceps"]) != patterns["sequence_length"]
            ):
                errors.append(
                    f"Pattern {pattern_id} has invalid biceps: "
                    f"{pattern['biceps']}"
                )

        # Triceps
        if "triceps" in pattern:
            if (
                not isinstance(pattern["triceps"], list)
                or len(pattern["triceps"]) != patterns["sequence_length"]
            ):
                errors.append(
                    f"Pattern {pattern_id} has invalid triceps: "
                    f"{pattern['triceps']}"
                )

        # Color
        if "color" in pattern:
            if not re.match(HEX_COLOR_REGEX, pattern["color"]):
                errors.append(
                    f"Pattern {pattern_id} has invalid color: "
                    f"{pattern['color']}"
                )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid = protocol["grid"]

    for field in ["x", "y", "cell_size_cm", "start_cell"]:
        if field not in grid:
            errors.append(f"Missing grid field: {field}")

    # --------------------------------------------------------
    # Validation result
    # --------------------------------------------------------

    if errors:
        print("[ERROR] Protocol validation failed:")
        for e in errors:
            print(" -", e)
        return False

    print("[OK] Protocol validated successfully")
    return True


def validate_main_data(df, protocol):
    """
    Validate experimental data against the protocol.

    Returns:
        bool: True if valid, False otherwise.
    """

    required_cols = [
        "subject",
        "duration",
        "rep",
        "x",
        "y",
        "pattern_biceps",
        "pattern_triceps",
        "vividness",
    ]

    errors = []

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    # If required columns are missing, stop here
    if errors:
        print("[ERROR] Subject data validation failed:")
        for e in errors:
            print(" -", e)
        return False

    # --------------------------------------------------------
    # Grid coordinates
    # --------------------------------------------------------

    NX = protocol["grid"]["x"]
    NY = protocol["grid"]["y"]

    if not df["x"].between(1, NX).all():
        errors.append(
            f"Some x coordinates are out of range (1-{NX})"
        )

    if not df["y"].between(1, NY).all():
        errors.append(
            f"Some y coordinates are out of range (1-{NY})"
        )

    # --------------------------------------------------------
    # Pattern pairs
    # --------------------------------------------------------

    protocol_pairs = [
        p["text"]
        for p in protocol["patterns"]["definitions"]
    ]

    for _, row in df.iterrows():

        pair = f"{row['pattern_pair']}"

        if pair not in protocol_pairs:
            errors.append(
                f"Unknown pattern pair: {pair} "
                f"(subject {row['subject']})"
            )

    # --------------------------------------------------------
    # Validation result
    # --------------------------------------------------------

    if errors:
        print("[ERROR] Subject data validation failed:")
        for e in errors:
            print(" -", e)
        return False

    print("[OK] Subject data validated successfully")
    return True

def validate_subjects_data(df):
    """
    Validate the Subjects worksheet.

    Required columns:
        subject
        forearm_cm
        forearm_angle_deg
        upper_cm
        block_order
    """

    required_columns = {
        "subject",
        "forearm_cm",
        "forearm_angle_deg",
        "upper_cm",
        "block_order",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(
            f"Missing columns in Subjects sheet: "
            f"{sorted(missing_columns)}"
        )
        return False

    if df["subject"].isna().any():
        print("Subjects sheet contains missing subject IDs.")
        return False

    if df["subject"].duplicated().any():
        print("Subjects sheet contains duplicate subject IDs.")
        return False

    numeric_columns = [
        "forearm_cm",
        "forearm_angle_deg",
        "upper_cm",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(
                f"Column '{column}' in Subjects sheet "
                f"must be numeric."
            )
            return False

    if df["block_order"].isna().any():
        print("Subjects sheet contains missing block_order values.")
        return False

    return True