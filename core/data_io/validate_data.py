import json
import pandas as pd
import re

HEX_COLOR_REGEX = r"^#(?:[0-9a-fA-F]{3}){1,2}$"

def validate_protocol(protocol):
    """
    Validate the JSON protocol.
    Args:
        protocol_path (str): Path to the protocol JSON file.
    Returns:
        bool: True if valid, False otherwise.
    """
    required_fields = ["id", "biceps", "triceps", "text", "color"]
    errors = []

    for pattern in protocol["patterns"]["definitions"]:
        for field in required_fields:
            if field not in pattern:
                errors.append(f"Pattern {pattern.get('id', 'unknown')} missing field: {field}")
        if not isinstance(pattern["biceps"], list) or len(pattern["biceps"]) != 3:
            errors.append(f"Pattern {pattern['id']} has invalid biceps: {pattern['biceps']}")
        if not isinstance(pattern["triceps"], list) or len(pattern["triceps"]) != 3:
            errors.append(f"Pattern {pattern['id']} has invalid triceps: {pattern['triceps']}")
        if not re.match(HEX_COLOR_REGEX, pattern["color"]):
            errors.append(f"Pattern {pattern['id']} has invalid color: {pattern['color']}")
    
    if errors:
        print("[ERROR] Protocol validation failed:")
        for e in errors:
            print(" -", e)
        return False
    print("[OK] Protocol validated successfully")
    return True

def validate_subject_data(df, protocol):
    """
    Validate Excel subject data against protocol.
    Args:
        df (pd.DataFrame): Subject data.
        protocol (dict): Protocol configuration.
    Returns:
        bool: True if valid, False otherwise.
    """
    required_cols = ["subject", "duration", "rep", "x", "y", "pattern_biceps", "pattern_triceps", "vividness"]
    errors = []

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    # Check x, y ranges
    NX, NY = 21, 9
    if protocol["name"] == "Protocol4": NY = 10
    if not df["x"].between(1, NX).all():
        errors.append("Some x coordinates are out of range")
    if not df["y"].between(1, NY).all():
        errors.append("Some y coordinates are out of range")

    # Check pattern pairs exist in protocol
    protocol_pairs = [p["text"] for p in protocol["patterns"]["definitions"]]
    for _, row in df.iterrows():
        pair = f"{row['pattern_pair']}"
        if pair not in protocol_pairs:
            errors.append(f"Unknown pattern pair: {pair} (subject {row['subject']})")

    if errors:
        print("[ERROR] Subject data validation failed:")
        for e in errors:
            print(" -", e)
        return False
    print("[OK] Subject data validated successfully")
    return True
